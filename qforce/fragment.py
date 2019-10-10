import networkx as nx
import os, hashlib
import networkx.algorithms.isomorphism as iso
from .elements import elements
from .read_qm_out import QM
from .molecule import Molecule
from .make_qm_input import make_dihedral_scan_input

def fragment(inp): 
    missing_data, got_data = [], []
    qm = QM(fchk_file = inp.fchk_file, out_file = inp.qm_freq_out)
    mol = Molecule(qm.coords, qm.atomids, inp, qm = qm)
    e = elements()
    t = 0

    os.makedirs(inp.frag_lib, exist_ok=True)

    for atoms, term in zip(mol.dih.flex.atoms, mol.dih.flex.term_ids):
        if term != t:
            continue
        fragment, capping_h = identify_fragment(atoms, mol, e)
        G = make_frag_graph(fragment, capping_h, mol, atoms)
        frag_id, G = make_frag_identifier(G, mol, inp, e, atoms)
        has_data, id_no = check_frag_in_database(G, inp.frag_lib, frag_id)
        if has_data:
            got_data.append(f'{frag_id}~{id_no}')
        else:
            missing_data.append(f'{frag_id}~{id_no}')
            make_dihedral_scan_input(inp, frag_id, id_no)
        t += 1
    
    n_missing, n_got = len(missing_data), len(got_data)
    print(f"There are {n_missing+n_got} unique flexible dihedrals.")

    if n_missing == 0:
        print(f"All scan data is available. Continuing with the next step...")
    else:
        print(f"{n_missing} of them are missing the scan data.")
        print(f"QM input files for them are created in {inp.job_name}_qforce")

def identify_fragment(atoms, mol, e):
    capping_h = []
    n_neigh = 0
    fragment = atoms[1:3]
    next_neigh = [[a, n] for a in fragment for n 
                  in mol.neighbors[0][a] if n not in fragment]
    while next_neigh != []:
        new = []
        for a, n in next_neigh:
            if n in fragment:
                pass
            elif (n_neigh < 2 or not mol.edge(a, n)['breakable'] 
                 or not mol.node(n)['breakable']):
                new.append(n)
                fragment.append(n)
            else:   
                bl = mol.edge(a, n)['length']
                new_bl = e.cov[mol.atomids[a]] + e.cov[1]
                vec = mol.node(a)['coords'] - mol.node(n)['coords']
                bl * new_bl
                capping_h.append((a, mol.coords[a] - vec/bl*new_bl))
        next_neigh = [[a, n] for a in new for n in mol.neighbors[0][a] 
                      if n not in fragment]
        n_neigh += 1
    return fragment, capping_h

def make_frag_graph(fragment, capping_h, mol, atoms):
    G = mol.graph.subgraph(fragment)
    G.graph['n_atoms'] = len(fragment)
    mapping = {fragment[i]:i for i in range(G.graph['n_atoms'])}
    G = nx.relabel_nodes(G, mapping)
    G.graph['scanned'] = [mapping[a] for a in atoms]

    for _, _, d in G.edges(data=True):
        for att in ['vector', 'length', 'order', 'breakable', 'vers', 
                    'in_ring3', 'in_ring']:
            d.pop(att, None)
    for _, d in G.nodes(data=True):
        for att in ['breakable', 'q', 'n_ring']:                
            d.pop(att, None)

    for i, h in enumerate(capping_h):
        G.add_node(G.graph['n_atoms']+i, elem = 1 , n_bonds = 1, lone_e = 0,
                   coords = h[1])
        G.add_edge(G.graph['n_atoms']+i, h[0],
                   type = f'1(1.0){mol.atomids[h[0]]}')
    G.graph['n_atoms'] += len(capping_h)
    return G

def make_frag_identifier(G, mol, inp, e, atoms):
    atom_ids = [[], []]
    comp_dict = {i : 0 for i in mol.atomids}
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
    s1, s2 = sorted([e.sym[mol.atomids[elem]] for elem in atoms[1:3]])
    G.graph['charge'] = charge
    if (sum(mol.atomids) + charge) % 2 == 1:
        mult = 2
    G.graph['mult'] = mult
    for elem in nx.get_node_attributes(G, 'elem').values():
        comp_dict[elem] +=1
    for elem in sorted(comp_dict):
        composition += f"{e.sym[elem]}{comp_dict[elem]}"
    if inp.disp == '':
        disp = ''
    else:
        disp = f'-{inp.disp}'
    frag_id = (f"{s1}{s2}_{composition}_{charge}_{mult}_{inp.method}"
              f"{disp}_{inp.basis}_{frag_hash}")
    return frag_id, G

def check_frag_in_database(G, frag_lib, frag_id):
    frag_dir = f'{frag_lib}/{frag_id}'
    has_data = False
    match = []
    nm = iso.categorical_node_match(['elem', 'n_bonds', 'lone_e'], [0, 0, 0])
    em = iso.categorical_edge_match('type', 0)
    id_exists = any([f for f in os.listdir(frag_lib) if f == frag_id])
    # at the next step copy everything in jobname_qforce/scandata_* to here.
    if id_exists:
        identifiers = [i for i in sorted(os.listdir(f'{frag_dir}')) 
                       if i.startswith('identifier')]
        for i in identifiers:
            compared = nx.read_gpickle(f"{frag_dir}/{i}")
            match.append(nx.is_isomorphic(G, compared, node_match = nm,
                                          edge_match = em))
        if any(match):
            id_no = match.index(True)+1
            if os.path.isfile(f'scandata_{id_no}'):
                has_data = True
        else:
            id_no = len(match)+1
            nx.write_gpickle(G, f"{frag_dir}/identifier_{id_no}")
            write_xyz(G, frag_dir, frag_id, id_no)
    else:
        os.makedirs(frag_dir, exist_ok=True)
        id_no = 1
        nx.write_gpickle(G, f"{frag_dir}/identifier_{id_no}")
        write_xyz(G, frag_dir, frag_id, id_no)
    return has_data, id_no
        
def write_xyz(G, frag_dir, frag_id, id_no):
    e = elements()
    with open(f'{frag_dir}/coords_{id_no}.xyz', 'w') as xyz:
        xyz.write(f'{G.graph["n_atoms"]}\n')
        xyz.write(f'{frag_id}_{id_no}\n')
        for data in G.nodes.data():
            atom_name, [c1, c2, c3] = e.sym[data[1]['elem']], data[1]['coords']
            xyz.write(f'{atom_name:>3s} {c1:>12.6f} {c2:>12.6f} {c3:>12.6f}\n')
    
    