import networkx as nx
import os
import hashlib
import sys
import networkx.algorithms.isomorphism as iso
from scipy.optimize import curve_fit
import numpy as np
from .elements import elements
from .read_qm_out import QM
from .make_qm_input import make_qm_input


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

        frag_name, have_data, G, pairs = check_one_fragment(inp, mol.topo,
                                                            term.atomids)
        if have_data:
            n_have = write_data(have_path, frag_name, inp, n_have)
            param = calc_dihedral_function(inp, frag_name, pairs)
            params[str(term)] = param
        else:
            n_missing = write_data(missing_path, frag_name, inp, n_missing)
            make_qm_input(inp, G, frag_name)

    check_and_notify(inp, n_missing, n_have)

    for term in mol.terms['dihedral/flexible']:
        term.equ = params[str(term)]


def check_and_notify(inp, n_missing, n_have):
    if n_missing+n_have == 0:
        print('There are no flexible dihedrals.')
    else:
        print(f"There are {n_missing+n_have} unique flexible dihedrals.")
    if n_missing == 0:
        print(f"All scan data is available. Continuing with the fitting...\n")
    else:
        print(f"{n_missing} of them are missing the scan data.")
        print(f"QM input files for them are created in: {inp.frag_dir}\n\n")
        sys.exit()


def calc_dihedral_function(inp, frag_name, pairs):
    energy_diff = []
    frag_id, id_no = frag_name.split('~')
    frag_dir = f'{inp.frag_lib}/{frag_id}'

    angles, energies = np.loadtxt(f'{frag_dir}/scandata_{id_no}', unpack=True)
    coords = np.load(f'{frag_dir}/scancoords_{id_no}.npy')

    for energy, coord in zip(energies, coords):
#        for i, j, c6, c12, qq in pairs:
#            energy = calc_pair_energies(coord, i, j, c6, c12, qq, energy)
        energy_diff.append(energy)

    energy_diff = np.array(energy_diff) - np.array(energy_diff).min()

    angles_radians = np.radians(angles)
    weights = np.exp(-0.2 * np.sqrt(energies))

    popt, _ = curve_fit(calc_rb, angles_radians, energy_diff,
                        absolute_sigma=False, sigma=weights)
    r_squared = calc_r_squared(calc_rb, angles_radians, energy_diff, popt)

    print(r_squared)
    # Plot with r_squared
    return popt


def calc_r_squared(funct, x, y, params):
    residuals = y - funct(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - (ss_res / ss_tot)


def calc_rb(angle, c0, c1, c2, c3, c4, c5):
    params = [c0, c1, c2, c3, c4, c5]
    rb = 0
    for i, p in enumerate(params):
        rb += p * np.cos(angle-np.pi)**i
    return rb


def check_one_fragment(inp, topo, atoms):
    e = elements()
    fragment, capping_h = identify_fragment(atoms, topo, e)
    G, pairs = make_frag_graph(fragment, capping_h, topo, atoms)
    frag_id, G = make_frag_identifier(G, topo, inp, e, atoms)
    have_data, id_no = check_frag_in_database(G, inp.frag_lib, frag_id)
    frag_name = f'{frag_id}~{id_no}'

    if not have_data:
        have_data = check_new_scan_data(inp, frag_name)

    return frag_name, have_data, G, pairs


def reset_data_files(inp):
    missing_path = f'{inp.frag_dir}/missing_data'
    have_path = f'{inp.frag_dir}/have_data'

    for data in [missing_path, have_path]:
        if os.path.exists(data):
            os.remove(data)
    return missing_path, have_path


def check_new_scan_data(inp, frag_name):
    found = False
    outs = [f for f in os.listdir(inp.frag_dir) if f.startswith(frag_name) and
            f.endswith(('log', 'out'))]
    for out in outs:
        qm = QM('scan', out_file=f'{inp.frag_dir}/{out}')
        if qm.normal_term:
            found = True
            frag_id, frag_no = frag_name.split('~')
            path = f'{inp.frag_lib}/{frag_id}/scandata_{frag_no}'
            if not os.path.isfile(path):
                with open(path, 'w') as data:
                    for angle, energy in zip(qm.angles, qm.energies):
                        data.write(f'{angle:>10.3f} {energy:>20.8f}\n')
            path = f'{inp.frag_lib}/{frag_id}/scancoords_{frag_no}.npy'
            if not os.path.isfile(path):
                np.save(path, qm.coords)
        else:
            print(f'WARNING: Scan output file "{out}" has not'
                  'terminated sucessfully. Skipping it...\n\n')
    return found


def write_data(data, frag_name, inp, n):
    with open(data, 'a+') as data_f:
        data_f.write(f'{frag_name}\n')
        n += 1
    return n


def identify_fragment(atoms, topo, e):
    capping_h = []
    n_neigh = 0
    fragment = list(atoms[1:3])
    next_neigh = [[a, n] for a in fragment for n
                  in topo.neighbors[0][a] if n not in fragment]
    while next_neigh != []:
        new = []
        for a, n in next_neigh:
            if n in fragment:
                pass
            elif (n_neigh < 2 or not topo.edge(a, n)['breakable']
                  or not topo.node(n)['breakable']):
                new.append(n)
                fragment.append(n)
            else:
                bl = topo.edge(a, n)['length']
                new_bl = e.cov[topo.atomids[a]] + e.cov[1]
                vec = topo.node(a)['coords'] - topo.node(n)['coords']
                capping_h.append((a, topo.coords[a] - vec/bl*new_bl))
        next_neigh = [[a, n] for a in new for n in topo.neighbors[0][a]
                      if n not in fragment]
        n_neigh += 1
    return fragment, capping_h


def make_frag_graph(fragment, capping_h, topo, atoms):
    G = topo.graph.subgraph(fragment)
    G.graph['n_atoms'] = len(fragment)
    mapping = {fragment[i]: i for i in range(G.graph['n_atoms'])}
    G = nx.relabel_nodes(G, mapping)
    scanned = [mapping[a] for a in atoms]
    G.graph['scan'] = [s+1 for s in scanned]
    G.edges[scanned[1:3]]['scan'] = True

    pairs = [[mapping[a[0]], mapping[a[1]], a[2], a[3], a[4]] for a in
             topo.pair_list if set(a[0:2]) <= set(mapping.keys())]

    for _, _, d in G.edges(data=True):
        for att in ['vector', 'length', 'order', 'breakable', 'vers',
                    'in_ring3', 'in_ring']:
            d.pop(att, None)
    for _, d in G.nodes(data=True):
        for att in ['breakable', 'q', 'n_ring']:
            d.pop(att, None)
    for i, h in enumerate(capping_h):
        G.add_node(G.graph['n_atoms']+i, elem=1, n_bonds=1, lone_e=0,
                   coords=h[1], c6=101, c12=31405)
        G.add_edge(G.graph['n_atoms']+i, mapping[h[0]],
                   type=f'1(1.0){topo.atomids[h[0]]}')
    G.graph['n_atoms'] += len(capping_h)
    return G, pairs


def make_frag_identifier(G, topo, inp, e, atoms):
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
    s1, s2 = sorted([e.sym[topo.atomids[elem]] for elem in atoms[1:3]])
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


def check_frag_in_database(G, frag_lib, frag_id):
    frag_dir = f'{frag_lib}/{frag_id}'
    have_data = False
    match = []
    nm = iso.categorical_node_match(['elem', 'n_bonds', 'lone_e'], [0, 0, 0])
    em = iso.categorical_edge_match(['type', 'scan'], [0, False])
    id_exists = any([f for f in os.listdir(frag_lib) if f == frag_id])
    # at the next step copy everything in jobname_qforce/scandata_* to here.
    if id_exists:
        identifiers = [i for i in sorted(os.listdir(f'{frag_dir}'))
                       if i.startswith('identifier')]
        for i in identifiers:
            compared = nx.read_gpickle(f"{frag_dir}/{i}")
            match.append(nx.is_isomorphic(G, compared, node_match=nm,
                                          edge_match=em))
        if any(match):
            id_no = match.index(True)+1
            if os.path.isfile(f'{frag_dir}/scandata_{id_no}'):
                have_data = True
        else:
            id_no = len(match)+1
            nx.write_gpickle(G, f"{frag_dir}/identifier_{id_no}")
            write_xyz(G, frag_dir, frag_id, id_no)
    else:
        os.makedirs(frag_dir, exist_ok=True)
        id_no = 1
        nx.write_gpickle(G, f"{frag_dir}/identifier_{id_no}")
        write_xyz(G, frag_dir, frag_id, id_no)
    return have_data, id_no


def write_xyz(G, frag_dir, frag_id, id_no):
    e = elements()
    with open(f'{frag_dir}/coords_{id_no}.xyz', 'w') as xyz:
        xyz.write(f'{G.graph["n_atoms"]}\n')
        xyz.write('Scanned atoms: {} {} {} {}\n'.format(*G.graph["scan"]))
        for data in sorted(G.nodes.data()):
            atom_name, [c1, c2, c3] = e.sym[data[1]['elem']], data[1]['coords']
            xyz.write(f'{atom_name:>3s} {c1:>12.6f} {c2:>12.6f} {c3:>12.6f}\n')
