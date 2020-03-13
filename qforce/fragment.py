import networkx as nx
import os
import hashlib
import sys
import networkx.algorithms.isomorphism as iso
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np
#
from ase.optimize.sciopt import SciPyFminBFGS
from ase.optimize import BFGS
from ase import Atoms
from ase.constraints import FixInternals
#
from .elements import elements
from .read_qm_out import QM
from .make_qm_input import make_qm_input
from .calculator import QForce
from .forces import get_dihed


def fragment(inp, mol, qm):
    unique_terms = []
    params = {}
    n_missing, n_have = 0, 0
    missing_path, have_path = reset_data_files(inp)

    for term in mol.terms['dihedral/flexible']:
        if str(term) in unique_terms:
            continue
        else:
            unique_terms.append(str(term))

        frag_name, have_data, G, terms, elems, scanned = check_one_fragment(inp, mol, qm,
                                                                            term.atomids)

#        if frag_name != 'CO_H8C3O2_0_1_PBEPBE-GD3BJ_6-31+G-D_0fa89da172948014ae3527354858c0aa~1':
#            continue

        if have_data:
            print(f'Fitting dihedral number {len(unique_terms)}\n')
            n_have = write_data(have_path, frag_name, inp, n_have)
            param = calc_dihedral_function(inp, mol, frag_name, terms, elems, scanned)
            params[str(term)] = param
        else:
            n_missing = write_data(missing_path, frag_name, inp, n_missing)
            make_qm_input(inp, G, frag_name)

    check_and_notify(inp, n_missing, n_have)

    for term in mol.terms['dihedral/flexible']:
        term.equ = params[str(term)]


def calc_dihedral_function(inp, mol, frag_name, terms, elems, scanned):
    md_energies = []
    frag_id, id_no = frag_name.split('~')
    frag_dir = f'{inp.frag_lib}/{frag_id}'

    angles, qm_energies = np.loadtxt(f'{frag_dir}/scandata_{id_no}', unpack=True)
    coords = np.load(f'{frag_dir}/scancoords_{id_no}.npy')
    angles_radians = np.radians(angles)
    coords = coords[:, :len(elems)]  # ignore the capping hydrogens during the minimization

    for angle, qm_energy, coord in zip(angles_radians, qm_energies, coords):
        frag = Atoms(elems, positions=coord, calculator=QForce(terms))
        dihedral_constraints = []
        for dihed in terms['dihedral/flexible']:
            angle_const = get_dihed(coord[dihed.atomids])[0]
            dihedral_constraints.append([angle_const, dihed.atomids])
            print(np.degrees(angle_const), dihed.atomids)
        constraints = FixInternals(dihedrals=dihedral_constraints)
        frag.set_constraint(constraints)

        # e_minimiz = SciPyFminBFGS(frag, logfile=f'{inp.frag_dir}/opt_{frag_name}.log')
        # try:
        #     e_minimiz.run(fmax=0.05, steps=1000)
        # except:
    #    ignores = ['dihedral/flexible', 'dihedral/constr']
    #    with terms.add_ignore(ignores):

        e_minimiz = BFGS(frag, trajectory=f'{inp.frag_dir}/{frag_name}_{np.degrees(angle)}.traj',
                         logfile=f'{inp.frag_dir}/opt_{frag_name}.log')
        try:
            e_minimiz.run(fmax=0.01, steps=1000)
        except Exception:
            print('WARNING: Possible convergence problem in fragment the optimization procedure.')

        md_energies.append(frag.get_potential_energy())

    md_energies = np.array(md_energies)
    energy_diff = qm_energies - md_energies
    energy_diff -= energy_diff.min()

    weights = 1/np.exp(-0.2 * np.sqrt(qm_energies))
    np.save('angles', angles)
    np.save('diff', energy_diff)
    np.save('qm', qm_energies)
    np.save('md', md_energies)
    popt, _ = curve_fit(calc_rb, angles_radians, energy_diff, absolute_sigma=False, sigma=weights)
    r_squared = calc_r_squared(calc_rb, angles_radians, energy_diff, popt)

    md_energies += calc_rb(angles_radians, *popt)
    md_energies -= md_energies.min()
    plot_fragment_results(inp, angles, qm_energies, md_energies, frag_name, r_squared)

    return popt


def plot_fragment_results(inp, angles, qm_energies, md_energies, frag_name, r_squared):
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width, height), dpi=300)
    sns.set(font_scale=1.3)
    plt.title(f'R-squared = {round(r_squared, 3)}', loc='left')
    plt.xlabel('Angle')
    plt.ylabel('Energy (kJ/mol)')
    plt.plot(angles, qm_energies, linewidth=4, label='QM')
    plt.plot(angles, md_energies, linewidth=4, label='Q-Force')
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
    f.savefig(f"{inp.frag_dir}/scan_data_{frag_name}.pdf", bbox_inches='tight')


def check_one_fragment(inp, mol, qm, dihed_atoms):
    e = elements()
    frag_atomids, capping_h = identify_fragment(dihed_atoms, mol.topo, e)
    G, terms, elems, scanned = make_fragment(qm, mol, frag_atomids, capping_h, dihed_atoms)
    frag_id, G = make_fragment_identifier(G, mol.topo, inp, e, dihed_atoms)
    have_data, frag_name, mapping_db_to_current = check_for_fragment(G, inp, frag_id)

    return frag_name, have_data, G, terms, elems, scanned


def identify_fragment(dihed_atoms, topo, e):
    capping_h = []
    n_neigh = 0
    frag_atomids = list(dihed_atoms[1:3])
    next_neigh = [[a, n] for a in frag_atomids for n
                  in topo.neighbors[0][a] if n not in frag_atomids]
    while next_neigh != []:
        new = []
        for a, n in next_neigh:
            if n in frag_atomids:
                pass
            elif (n_neigh < 2 or not topo.edge(a, n)['breakable']
                  or not topo.node(n)['breakable']):
                new.append(n)
                frag_atomids.append(n)
            else:
                bl = topo.edge(a, n)['length']
                new_bl = e.cov[topo.atomids[a]] + e.cov[1]
                vec = topo.node(a)['coords'] - topo.node(n)['coords']
                capping_h.append((a, topo.coords[a] - vec/bl*new_bl))
        next_neigh = [[a, n] for a in new for n in topo.neighbors[0][a] if n not in frag_atomids]
        n_neigh += 1
    return frag_atomids, capping_h


def make_fragment(qm, mol, frag_atomids, capping_h, dihed_atoms):
    G = mol.topo.graph.subgraph(frag_atomids)
    G.graph['n_atoms'] = len(frag_atomids)
    mapping = {frag_atomids[i]: i for i in range(G.graph['n_atoms'])}

    terms = mol.terms.subset(frag_atomids, mapping)

    elems = np.array([qm.atomids[i] for i in frag_atomids])

    G = nx.relabel_nodes(G, mapping)
    scanned = [mapping[a] for a in dihed_atoms]
    G.graph['scan'] = [s+1 for s in scanned]
    G.edges[scanned[1:3]]['scan'] = True

    for _, _, d in G.edges(data=True):
        for att in ['vector', 'length', 'order', 'breakable', 'vers', 'in_ring3', 'in_ring']:
            d.pop(att, None)
    for _, d in G.nodes(data=True):
        for att in ['breakable', 'q', 'n_ring']:
            d.pop(att, None)
    for i, h in enumerate(capping_h):
        G.add_node(G.graph['n_atoms']+i, elem=1, n_bonds=1, lone_e=0, coords=h[1])
        G.add_edge(G.graph['n_atoms']+i, mapping[h[0]], type=f'1(1.0){mol.topo.atomids[h[0]]}')
    G.graph['n_atoms'] += len(capping_h)
    return G, terms, elems, scanned


def make_fragment_identifier(G, topo, inp, e, dihed_atoms):
    atom_ids = [[], []]
    comp_dict = {i: 0 for i in topo.atomids}
    mult = 1
    composition = ""
    for a in range(2):
        neighbors = nx.bfs_tree(G, a, depth_limit=4).nodes
        for n in neighbors:
            paths = nx.all_simple_paths(G, a, n, cutoff=4)
            for path in map(nx.utils.pairwise, paths):
                types = [G.edges[edge]['type'] for edge in path]
                atom_ids[a].append("-".join(types))
        atom_ids[a] = "_".join(sorted(atom_ids[a]))
    frag_hash = "=".join(sorted(atom_ids))
    frag_hash = hashlib.md5(frag_hash.encode()).hexdigest()

    charge = int(round(sum(nx.get_node_attributes(G, 'q').values())))
    s1, s2 = sorted([e.sym[topo.atomids[elem]] for elem in dihed_atoms[1:3]])
    G.graph['charge'] = charge
    if (sum(topo.atomids) + charge) % 2 == 1:
        mult = 2
    G.graph['mult'] = mult
    for elem in nx.get_node_attributes(G, 'elem').values():
        comp_dict[elem] += 1
    for elem in sorted(comp_dict):
        composition += f"{e.sym[elem]}{comp_dict[elem]}"
    if inp.disp == '':
        disp = ''
    else:
        disp = f'-{inp.disp}'
    frag_id = (f"{s1}{s2}_{composition}_{charge}_{mult}_{inp.method}"
               f"{disp}_{inp.basis}_{frag_hash}")
    frag_id = frag_id.replace('(', '-').replace(',', '').replace(')', '')
    return frag_id, G


def check_for_fragment(G, inp, frag_id):
    """
    Check if fragment exists in the database
    If not, check current fragment directory if new data is there
    """
    frag_dir = f'{inp.frag_lib}/{frag_id}'

    mapping_db_to_current = {i: i for i in range(G.graph['n_atoms'])}
    have_data, have_match = False, False

    nm = iso.categorical_node_match(['elem', 'n_bonds', 'lone_e'], [0, 0, 0])
    em = iso.categorical_edge_match(['type', 'scan'], [0, False])

    os.makedirs(frag_dir, exist_ok=True)
    identifiers = [i for i in sorted(os.listdir(f'{frag_dir}')) if i.startswith('identifier')]

    for id_no, id_file in enumerate(identifiers, start=1):
        compared = nx.read_gpickle(f"{frag_dir}/{id_file}")
        GM = iso.GraphMatcher(G, compared, node_match=nm, edge_match=em)
        if GM.is_isomorphic():
            if os.path.isfile(f'{frag_dir}/scandata_{id_no}'):
                have_data = True
                mapping_db_to_current = GM.mapping
            have_match = True
            break

    if not have_match:
        id_no = len(identifiers)+1

    if not have_data:
        have_data = check_new_scan_data(inp, frag_dir, frag_id, id_no)
        nx.write_gpickle(G, f"{frag_dir}/identifier_{id_no}")
        write_xyz(G, frag_dir, frag_id, id_no)

    frag_name = f'{frag_id}~{id_no}'

    return have_data, frag_name, mapping_db_to_current


def check_new_scan_data(inp, frag_dir, frag_id, id_no):
    frag_name = f'{frag_id}~{id_no}'
    found = False
    outs = [f for f in os.listdir(inp.frag_dir) if f.startswith(frag_name) and
            f.endswith(('log', 'out'))]
    for out in outs:
        qm = QM(inp, 'scan', out_file=f'{inp.frag_dir}/{out}')
        if qm.normal_term:
            found = True
            with open(f'{frag_dir}/scandata_{id_no}', 'w') as data:
                for angle, energy in zip(qm.angles, qm.energies):
                    data.write(f'{angle:>10.3f} {energy:>20.8f}\n')
            np.save(f'{frag_dir}/scancoords_{id_no}.npy', qm.coords)
        else:
            print(f'WARNING: Scan output file "{out}" has not'
                  'terminated sucessfully. Skipping it...\n\n')
    return found


def reset_data_files(inp):
    missing_path = f'{inp.frag_dir}/missing_data'
    have_path = f'{inp.frag_dir}/have_data'

    for data in [missing_path, have_path]:
        if os.path.exists(data):
            os.remove(data)
    return missing_path, have_path


def write_data(data, frag_name, inp, n):
    with open(data, 'a+') as data_f:
        data_f.write(f'{frag_name}\n')
        n += 1
    return n


def write_xyz(G, frag_dir, frag_id, id_no):
    e = elements()
    with open(f'{frag_dir}/coords_{id_no}.xyz', 'w') as xyz:
        xyz.write(f'{G.graph["n_atoms"]}\n')
        xyz.write('Scanned atoms: {} {} {} {}\n'.format(*G.graph["scan"]))
        for data in sorted(G.nodes.data()):
            atom_name, [c1, c2, c3] = e.sym[data[1]['elem']], data[1]['coords']
            xyz.write(f'{atom_name:>3s} {c1:>12.6f} {c2:>12.6f} {c3:>12.6f}\n')


def check_and_notify(inp, n_missing, n_have):
    if n_missing+n_have == 0:
        print('There are no flexible dihedrals.')
    else:
        print(f"There are {n_missing+n_have} unique flexible dihedrals.")
    if n_missing == 0:
        print(f"\nAll scan data is available. Continuing with the fitting...\n")
    else:
        print(f"{n_missing} of them are missing the scan data.")
        print(f"QM input files for them are created in: {inp.frag_dir}\n\n")
        sys.exit()


def calc_r_squared(funct, x, y, params):
    residuals = y - funct(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - (ss_res / ss_tot)


def calc_rb(angles, c0, c1, c2, c3, c4, c5):
    params = [c0, c1, c2, c3, c4, c5]

    rb = np.full(len(angles), c0)

    for i in range(1, 6):
        rb += params[i] * np.cos(angles-np.pi)**i
    return rb
