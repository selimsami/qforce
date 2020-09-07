import subprocess
import numpy as np
import pulp
from scipy.optimize import curve_fit
from itertools import combinations_with_replacement
#
from .. import qforce_data
from ..elements import ATOM_SYM


class NonBonded():
    def __init__(self, inp, n_atoms, q, lj_types, lj_pairs, lj_1_4, exclusions, n_excl, alpha):
        self.n_atoms = n_atoms
        self.q = q
        self.lj_types = lj_types
        self.lj_pairs = lj_pairs
        self.lj_1_4 = lj_1_4
        self.fudge_lj = inp.fudge_lj
        self.fudge_q = inp.fudge_q
        self.exclusions = exclusions
        self.n_excl = n_excl
        self.alpha = {key: alpha[key] for key in sorted(alpha.keys())}  # sort the dictionary
        self.alpha_map = {key: i+self.n_atoms for i, key in enumerate(self.alpha.keys())}

    @classmethod
    def from_topology(cls, inp, qm, topo):
        # RUN D4 IF NECESSARY
        if 'd4' in [inp.point_charges, inp.lennard_jones]:
            q, lj_a, lj_b = handle_d4(inp, qm, topo)

        # CHARGES - esp and resp
        if inp.point_charges == 'ext':
            q = np.loadtxt(f'{inp.job_dir}/ext_q', comments=['#', ';'])
        elif inp.point_charges == 'cm5':
            q = qm.cm5
        elif inp.point_charges == 'esp':
            q = qm.esp
        q = average_equivalent_terms(topo, [q])[0]
        q = sum_charges_to_qtotal(topo, q)

        # LENNARD-JONES
        if inp.lennard_jones != 'd4':
            if inp.lennard_jones == 'gromos_auto':
                lj_types = determine_atom_types(topo, q)
            else:
                lj_types = np.loadtxt(f'{inp.job_dir}/ext_lj', dtype='str', comments=['#', ';'])
            lj_pairs, lj_1_4 = set_external_lennard_jones(inp, lj_types)
        else:
            lj_types, lj_pairs = set_qforce_lennard_jones(topo, inp, lj_a, lj_b)
            lj_1_4 = lj_pairs
            print('WARNING: You are using Q-Force Lennard-Jones parameters. This is not finished.',
                  '\nYou are advised to provide external LJ parameters for production runs.\n')

        alpha = set_polar(q, topo, inp)

        return cls(inp, topo.n_atoms, q, lj_types, lj_pairs, lj_1_4, inp.exclusions, inp.n_excl,
                   alpha)

    @classmethod
    def subset(cls, inp, non_bonded, mapping):
        n_atoms = len(mapping)
        rev_map = {v: k for k, v in mapping.items()}

        q = np.array([non_bonded.q[rev_map[i]] for i in range(n_atoms)])
        lj_types = [non_bonded.lj_types[rev_map[i]] for i in range(n_atoms)]
        lj_pairs = {key: val for key, val in list(non_bonded.lj_pairs.items())
                    if key[0] in lj_types and key[1] in lj_types}
        lj_1_4 = {key: val for key, val in list(non_bonded.lj_1_4.items())
                  if key[0] in lj_types and key[1] in lj_types}
        exclusions = [(mapping[excl[0]], mapping[excl[1]]) for excl in inp.exclusions if
                      excl[0] in mapping.keys() and excl[1] in mapping.keys()]

        alpha = {mapping[key]: val for key, val in list(non_bonded.alpha.items())
                 if key in mapping.keys()}

        return cls(inp, n_atoms, q, lj_types, lj_pairs, lj_1_4, exclusions, inp.n_excl, alpha)


def determine_atom_types(topo, q):
    print('NOTE: Automatic atom-type determination (used only for LJ interactions) is new. \n'
          '      Double check your atom types or enter them manually.\n')
    a_types = []
    for i, elem in enumerate(topo.elements):
        elem_neigh = [topo.elements[atom] for atom in topo.neighbors[0][i]]

        if elem == 1:
            if topo.elements[topo.neighbors[0][i]] == 6:
                a_type = 'HC'  # bound to C
            else:
                a_type = 'HS14'

        elif elem == 6:
            in_ring_and_conj = [all([topo.edge(*edge)['order'] > 1, topo.edge(*edge)['in_ring']])
                                for edge in topo.graph.edges(i)]
            united_charge = q[i] + sum([q[j] for j in topo.neighbors[0][i] if
                                        topo.elements[j] == 1])

            if any(in_ring_and_conj):
                a_type = 'CAro'
            elif elem_neigh.count(6) == 4:  # CH4
                a_type = 'CH0'
            elif united_charge > 0.15:
                a_type = 'CPos'
            else:
                a_type = 'C'

        elif elem == 7:

            if len(elem_neigh) == 4 or elem_neigh.count(1) == 3:  # 4 bonds or NH3
                a_type = 'NL'
            elif elem_neigh.count(6) == 3:
                a_type = 'NTer'  # Bound to three C
            elif elem_neigh.count(1) == 2:
                a_type = 'NPri'  # Bound to two H
            else:
                a_type = 'NOpt'

        elif elem == 8:
            if elem_neigh.count(6) == 2:
                a_type = 'OE'
            elif len(elem_neigh) == 1:
                a_type = 'OEOpt'  # has 1 bond and it is to C
            # if 1 in topo.elements[topo.neighbors[0][i]]:
            else:
                a_type = 'OAlc'

        else:
            a_type = ATOM_SYM[elem].upper()

        a_types.append(a_type)

    return a_types


def set_polar(q, topo, inp):
    polar_dict = {1: 0.45330, 6: 1.30300, 7: 0.98840, 8: 0.83690, 16: 2.47400}
    # polar_dict = { 1: 0.000413835,  6: 0.00145,  7: 0.000971573,
    #               8: 0.000851973,  9: 0.000444747, 16: 0.002474448,
    #               17: 0.002400281, 35: 0.003492921, 53: 0.005481056}
    # polar_dict = { 1: 0.000413835,  6: 0.001288599,  7: 0.000971573,
    #               8: 0.000851973,  9: 0.000444747, 16: 0.002474448,
    #               17: 0.002400281, 35: 0.003492921, 53: 0.005481056}
    # polar_dict = { 1: 0.000205221,  6: 0.000974759,  7: 0.000442405,
    #               8: 0.000343551,  9: 0.000220884, 16: 0.001610042,
    #               17: 0.000994749, 35: 0.001828362, 53: 0.002964895}
    alpha_dict, alpha = {}, []

    if inp.polar:
        if inp.ext_alpha:
            atoms, alpha = np.loadtxt(f'{inp.job_dir}/ext_alpha', unpack=True, comments=['#', ';'])
            atoms = atoms.astype(dtype='int') - 1
            alpha *= 1000  # convert from nm3 to ang3
        else:
            atoms = np.arange(topo.n_atoms)
            for elem in topo.elements:
                alpha.append(polar_dict[elem])

        for i, a in zip(atoms, alpha):
            alpha_dict[i] = a

    # EPS0 = 1389.35458  # kJ*ang/mol/e2
    # for q, elem in zip(q, topo.elements):
    #     polar_fcs.append(64.0 * EPS0 / polar_dict[elem])

    # for i in range(topo.n_atoms):
    #     for j in range(i+1, topo.n_atoms):
    #         close_neighbor = any([j in topo.neighbors[c][i] for c in range(inp.n_excl)])
    #         if not close_neighbor and (i, j) not in inp.exclusions:
    #             polar_pairs.append([i, j])

    return alpha_dict


def set_qforce_lennard_jones(topo, inp, lj_a, lj_b):
    lj_type_dict = {}
    lj_pairs = {}
    lj_types = topo.types
    for i, atype in enumerate(lj_types[topo.unique_atomids]):
        lj_type_dict[atype] = [lj_a[i], lj_b[i]]

    for comb in combinations_with_replacement(lj_type_dict.keys(), 2):
        comb = tuple(sorted(comb))
        params = use_combination_rule(lj_type_dict[comb[0]], lj_type_dict[comb[1]], inp)
        lj_pairs[comb] = get_c6_c12_for_diff_comb_rules(inp, params)
    return lj_types, lj_pairs


def set_external_lennard_jones(inp, lj_types):
    lj_pairs, lj_1_4 = {}, {}

    atom_types, nonbond_params, nonbond_1_4 = read_ext_nonbonded_file(inp)

    for comb in combinations_with_replacement(set(lj_types), 2):
        comb = tuple(sorted(comb))

        if comb in nonbond_params.keys():
            params = nonbond_params[comb]
        elif comb[0] == comb[1]:
            params = atom_types[comb[0]]
        else:
            params = use_combination_rule(atom_types[comb[0]], atom_types[comb[1]], inp)

        if comb in nonbond_1_4.keys():
            params_1_4 = nonbond_1_4[comb]
            lj_1_4[comb] = get_c6_c12_for_diff_comb_rules(inp, params_1_4)

        lj_pairs[comb] = get_c6_c12_for_diff_comb_rules(inp, params)

    if inp.polar:
        for key, val in lj_pairs.items():
            if key[0] not in inp.polar_not_scale_c6 and key[1] not in inp.polar_not_scale_c6:
                val[0] *= inp.polar_c6_scale

    return lj_pairs, lj_1_4


def get_c6_c12_for_diff_comb_rules(inp, params):
    if inp.comb_rule == 1:
        c6 = params[0] * 1e6
        c12 = params[1] * 1e12
    else:
        sigma = params[0] * 10
        epsilon = params[1]
        sigma6 = sigma**6
        c6 = 4 * epsilon * sigma6
        c12 = c6 * sigma6
    return [c6, c12]


def use_combination_rule(param1, param2, inp):
    b = (param1[1] * param2[1])**0.5
    if inp.comb_rule in [1, 3]:
        a = (param1[0] * param2[0])**0.5
    else:
        a = (param1[0] + param2[0]) / 2
    return a, b


def calc_sigma_epsilon(c6, c12):
    sigma = (c12/c6)**(1/6)
    epsilon = c6 / (4*sigma**6)
    return sigma, epsilon


def read_ext_nonbonded_file(inp):
    atom_types, nonbond_params, nonbond_1_4 = {}, {}, {}

    if inp.lennard_jones == 'gromos_auto':
        lj_lib = 'gromos'
    else:
        lj_lib = inp.lennard_jones

    with open(f'{qforce_data}/{lj_lib}.itp', 'r') as file:
        for line in file:
            line = line.partition('#')[0].partition(';')[0].strip()
            if line == '':
                continue
            elif "[" in line and "]" in line:
                no_space = line.lower().replace(" ", "")
                in_section = no_space[no_space.index("[")+1:no_space.index("]")]
            else:
                line = line.split()
                if in_section == 'atomtypes':
                    atype, a, b = line[0], float(line[-2]), float(line[-1])
                    atom_types[atype] = [a, b]
                elif in_section == 'nonbond_params':
                    atype1, atype2, a, b = line[0], line[1], float(line[-2]), float(line[-1])
                    nonbond_params[tuple(sorted([atype1, atype2]))] = [a, b]
                elif in_section == 'pairtypes':
                    atype1, atype2, a, b = line[0], line[1], float(line[-2]), float(line[-1])
                    nonbond_1_4[tuple(sorted([atype1, atype2]))] = [a, b]

        return atom_types, nonbond_params, nonbond_1_4


def average_equivalent_terms(topo, terms):
    avg_terms = []
    for term in terms:
        term = np.array(term)
        for l in topo.list:
            term[l] = term[l].mean().round(5)
        avg_terms.append(term)
    return avg_terms


def sum_charges_to_qtotal(topo, q):
    total = q.sum()
    q_integer = int(round(total))

    extra = int(round(100000 * round(total - q_integer, 5)))

    if extra != 0:
        if extra > 0:
            sign = 1
        else:
            sign = -1
            extra = - extra

        n_eq = [len(l) for l in topo.list]
        eq_no = [f"{i:05d}" for i, _ in enumerate(n_eq)]

        var = pulp.LpVariable.dicts("x", eq_no, lowBound=0, cat='Integer')
        prob = pulp.LpProblem('prob', pulp.LpMinimize)
        prob += pulp.lpSum([var[n] for n in eq_no])
        prob += pulp.lpSum([eq * var[eq_no[i]] for i, eq in enumerate(n_eq)]) == extra
        prob.solve()

        if prob.status == 1:
            for i, v in enumerate(prob.variables()):
                q[topo.list[i]] -= sign * v.varValue / 100000
        else:
            print('Failed to equate total of charges to the total charge of '
                  'the system. Do so manually')
    return q


def handle_d4(inp, qm, topo):
    n_more = 0
    c6, c8, alpha, r_rel, q = [], [], [], [], []
    lj_a, lj_b = None, None

    d4_out = run_d4(inp)

    with open(f'{inp.job_dir}/dftd4_results', 'w') as dftd4_file:
        dftd4_file.write(d4_out)

    for line in d4_out.split('\n'):
        if 'number of atoms' in line:
            n_atoms = int(line.split()[4])
        elif 'covCN                  q              C6A' in line:
            n_more = n_atoms
        elif n_more > 0:
            line = line.split()
            q.append(float(line[4]))
            c6.append(float(line[5]))
            c8.append(float(line[6]))
            alpha.append(float(line[7]))
            r_rel.append(float(line[8])**(1/3))
            n_more -= 1

    if inp.lennard_jones == 'd4':
        q, c6, c8, alpha, r_rel = average_equivalent_terms(topo, [q, c6, c8, alpha, r_rel])
        c6, c8, alpha, r_rel = [term[topo.unique_atomids] for term in [c6, c8, alpha, r_rel]]
        lj_a, lj_b = calc_c6_c12(inp, qm, topo, c6, c8, r_rel)
    return q, lj_a, lj_b


def run_d4(inp):
    dftd4 = subprocess.Popen(['dftd4', '-c', str(inp.charge), inp.xyz_file],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dftd4.wait()
    check_termination(dftd4)
    out = dftd4.communicate()[0].decode("utf-8")
    return out


def check_termination(process):
    if process.returncode != 0:
        print(process.communicate()[0].decode("utf-8"))
        raise RuntimeError({"DFTD4 run has terminated unsuccessfully"})


def calc_c6_c12(inp, qm, topo, c6s, c8s, r_rels):
    hartree2kjmol = 2625.499638
    bohr2ang = 0.52917721067
    bohr2nm = 0.052917721067
    new_ljs = []

    r_ref = {1: 1.986, 6: 2.083, 7: 1.641, 8: 1.452, 9: 1.58, 16: 1.5}
    s8_scale = {1: 0.133, 6: 0.133, 7: 0.683, 8: 0.683, 9: 0.683, 16: 0.5}

    for i, (c6, c8, r_rel) in enumerate(zip(c6s, c8s, r_rels)):
        elem = qm.elements[topo.unique_atomids[i]]
        c8 *= s8_scale[elem]
        c10 = 40/49*(c8**2)/c6

        r_vdw = 2*r_ref[elem]*r_rel/bohr2ang
        r = np.arange(r_vdw*0.5, 20, 0.01)
        c12 = (c6 + c8/r_vdw**2 + c10/r_vdw**4) * r_vdw**6 / 2
        lj = c12/r**12 - c6/r**6 - c8/r**8 - c10/r**10
        weight = 10*(1-lj / min(lj))+1
        popt, _ = curve_fit(calc_lj, r, lj, absolute_sigma=False, sigma=weight)
        new_ljs.append(popt)

    new_ljs = np.array(new_ljs)*hartree2kjmol
    new_c6 = new_ljs[:, 0]*bohr2nm**6
    new_c12 = new_ljs[:, 1]*bohr2nm**12

    if inp.comb_rule != 1:
        new_a, new_b = calc_sigma_epsilon(new_c6, new_c12)
    else:
        new_a, new_b = new_c6, new_c12
    return new_a, new_b


def calc_lj(r, c6, c12):
    return c12/r**12 - c6/r**6


# def move_polarizability_from_hydrogens(alpha, mol):
#     new_alpha = np.zeros(mol.n_atoms)
#     for i, a_id in enumerate(mol.atomids):
#         if a_id == 1:
#             new_alpha[mol.neighbors[0][i][0]] += alpha[mol.atoms[i]]
#         else:
#             new_alpha[i] += alpha[mol.atoms[i]]
#     return new_alpha
