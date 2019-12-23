import subprocess
import numpy as np
import pulp
from ase.io import write
from ase import Atoms
from scipy.optimize import curve_fit


def run_dftd4(inp, mol, qm):
    n_more = 0
    q, c6, c8, alpha, r_rel = [], [], [], [], []
    write_optimized_xyz(inp, qm)
    dftd4 = subprocess.Popen(['dftd4', '-c', str(inp.charge), inp.xyz_file],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dftd4.wait()
    check_termination(dftd4)
    out = dftd4.communicate()[0].decode("utf-8")

    with open(f'{inp.job_dir}/dftd4_results', 'w') as dftd4_file:
        dftd4_file.write(out)

    for line in out.split('\n'):
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

    q = average_equivalent_terms(q, mol.list)
    alpha = average_equivalent_terms(alpha, mol.list)
    c6 = average_equivalent_terms(c6, mol.list)
    c8 = average_equivalent_terms(c8, mol.list)
    r_rel = average_equivalent_terms(r_rel, mol.list)

    qm.q = sum_charges_to_qtotal(mol, q, inp.charge)
    qm.alpha = move_polarizability_from_hydrogens(alpha, mol)
    calc_c6_c12(qm, mol, c6, c8, r_rel, inp.param)
    calc_pair_list(mol, qm, inp.nrexcl)


def calc_pair_list(mol, qm, nrexcl):
    eps0 = 1389.35458
    qm.sigma = (qm.c12/qm.c6)**(1/6)
    qm.epsilon = qm.c6 / (4*qm.sigma**6)

    for i, a1 in enumerate(mol.atoms):
        for j, a2 in enumerate(mol.atoms):
            if i < j and all([j not in mol.neighbors[c][i]
                              for c in range(nrexcl)]):
                sigma = 0.5 * (qm.sigma[a1] + qm.sigma[a2])
                epsilon = (qm.epsilon[a1] * qm.epsilon[a2])**0.5
                sigma6 = sigma**6
                c6 = 4 * epsilon * sigma6
                c12 = c6 * sigma6
                qq = qm.q[a1] * qm.q[a2] * eps0

                mol.pair_list.append([i, j, c6, c12, qq])


def write_optimized_xyz(inp, qm):
    inp.xyz_file = f'{inp.job_dir}/opt.xyz'
    mol = Atoms(numbers=qm.atomids, positions=qm.coords)
    write(inp.xyz_file, mol, plain=True, comment=f'{inp.job_name} - optimized '
          'geometry')


def check_termination(process):
    if process.returncode != 0:
        print(process.communicate()[0].decode("utf-8"))
        raise RuntimeError({"DFTD4 run has terminated unsuccessfully"})


def move_polarizability_from_hydrogens(alpha, mol):
    new_alpha = np.zeros(mol.n_atoms)
    for i, a_id in enumerate(mol.atomids):
        if a_id == 1:
            new_alpha[mol.neighbors[0][i][0]] += alpha[mol.atoms[i]]
        else:
            new_alpha[i] += alpha[mol.atoms[i]]
    return new_alpha


def average_equivalent_terms(term, eq):
    avg_term = []
    term = np.array(term)
    for l in eq:
        total = 0
        for a in l:
            total += term[a]
        avg_term.append(round(total/len(l), 5))
    return avg_term


def sum_charges_to_qtotal(mol, charges, q_total):
    total = sum([round(charges[i], 5)*len(l) for i, l in enumerate(mol.list)])
    extra = int(100000 * round(total - q_total, 5))
    if extra != 0:
        if extra > 0:
            sign = 1
        else:
            sign = -1
            extra = - extra

        n_eq = [len(l) for l in mol.list]
        no = [f"{i:05d}" for i, _ in enumerate(n_eq)]

        var = pulp.LpVariable.dicts("x", no, lowBound=0, cat='Integer')
        prob = pulp.LpProblem('prob', pulp.LpMinimize)
        prob += pulp.lpSum([var[n] for n in no])
        prob += pulp.lpSum([eq * var[no[i]] for i, eq
                            in enumerate(n_eq)]) == extra
        prob.solve()

        if prob.status == 1:
            for i, v in enumerate(prob.variables()):
                charges[i] -= sign * v.varValue / 100000
        else:
            print('Failed to equate total of charges to the total charge of '
                  'the system. Do so manually')
    return charges


def calc_c6_c12(qm, mol, c6s, c8s, r_rels, param):
    hartree2kjmol = 2625.499638
    bohr2ang = 0.52917721067
    new_ljs = []

    order2elem = [6, 1, 8, 7]
    r_ref = {1: 1.9860558, 6: 2.08254094, 7: 1.64052717, 8: 1.45226652}
    s8_scale = {1: 0.13335551, 6: 0.13335551, 7: 0.68314538, 8: 0.68314538}

    for i, s8 in enumerate(param[::2]):
        s8_scale[order2elem[i]] = s8
    for i, r in enumerate(param[1::2]):
        r_ref[order2elem[i]] = r

    for i, (c6, c8, r_rel) in enumerate(zip(c6s, c8s, r_rels)):
        elem = mol.atomids[mol.list[i][0]]
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
    qm.c6 = new_ljs[:, 0]*bohr2ang**6
    qm.c12 = new_ljs[:, 1]*bohr2ang**12


def calc_lj(r, c6, c12):
    return c12/r**12 - c6/r**6
