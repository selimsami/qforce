import networkx as nx
import os
import hashlib
import sys
import networkx.algorithms.isomorphism as iso
import numpy as np
import json
import pickle
#
from .elements import ELE_COV, ATOM_SYM, ELE_ENEG
from .forces import get_dihed

"""

Removing non-bonded: Only hydrogens excluded for the equi - make it more general
Problem with using capping atom info in the graph comparaison - avoid it

Prompt a warning when if any point has an error larger than 2kJ/mol.

"""


def fragment(mol, qm, job, config):
    fragments = []
    unique_dihedrals = {}

    os.makedirs(config.scan.frag_lib, exist_ok=True)
    os.makedirs(job.frag_dir, exist_ok=True)
    reset_data_files(job.frag_dir)

    for term in mol.terms['dihedral/flexible']:
        name = term.typename.partition('_')[0]

        if name not in unique_dihedrals:
            unique_dihedrals[name] = term.atomids

    generated = []  # Number of fragments generated but not computed
    for name, atomids in unique_dihedrals.items():
        frag = Fragment(job, config, mol, qm, atomids, name)
        if frag.has_data:
            fragments.append(frag)
        elif config.scan.batch_run and frag.has_inp:
            generated.append(frag)

    check_and_notify(job, config.scan, len(unique_dihedrals), len(fragments), len(generated))

    return fragments


def reset_data_files(frag_dir):
    for data in ['missing', 'have']:
        data_path = f'{frag_dir}/{data}'
        if os.path.exists(data_path):
            os.remove(data_path)


def check_and_notify(job, config, n_unique, n_have, n_generated):
    n_missing = n_unique - n_have
    if n_unique == 0:
        print('There are no flexible dihedrals.')
    else:
        print(f"There are {n_unique} unique flexible dihedrals.")
        if n_missing == 0:
            print("All scan data is available. Fitting the dihedrals...\n")
        else:
            print(f"{n_missing} of them are missing the scan data.")
            if n_generated > 0:
                print(
                    f"{n_generated} of them generated previously (Batch run enabled).")
            if n_missing - n_generated > 0:
                print(f"QM input files for them are created in: {job.frag_dir}")

            if config.avail_only:
                print('Continuing without the missing dihedrals...\n')
            else:
                print('Exiting...\n')
                raise SystemExit


class Fragment():
    """
    Issue: using capping categorization is not ideal - different capped fragments can be identical
    For now necessary because of the mapping - but should be fixed at some point
    """

    def __init__(self, job, config, mol, qm, scanned_atomids, name):
        self.central_atoms = tuple(scanned_atomids[1:3])
        self.scanned_atomids = scanned_atomids
        self.atomids = list(scanned_atomids[1:3])
        self.name = name
        self.caps = []
        self.n_atoms = 0
        self.n_atoms_without_cap = 0
        self.hash = ''
        self.hash_idx = 0
        self.id = ''
        self.has_data = False
        self.has_inp = False
        self.map_frag_to_db = {}
        self.map_mol_to_frag = {}
        self.elements = []
        self.terms = None
        self.non_bonded = None
        self.remove_non_bonded = []
        self.qm_energies = []
        self.qm_coords = []
        self.qm_angles = []
        self.fit_terms = []
        self.coords = []
        self.frag_charges = []
        self.charge_scaling = config.ff.charge_scaling
        self.ext_charges = config.ff.ext_charges
        self.use_ext_charges_for_frags = config.ff.use_ext_charges_for_frags
        self.charge_method = config.qm.charge_method

        self.check_fragment(job, config.scan, mol, qm)

    def check_fragment(self, job, config, mol, qm):
        self.identify_fragment(mol, config)
        self.make_fragment_graph(mol)
        self.make_fragment_identifier(config, mol, qm)
        self.check_for_fragment(job, config, qm)
        self.check_for_qm_data(job, config, mol, qm)
        self.make_fragment_terms(mol)

    def check_single_ring_rules(self, mol, bond, a1, a2):
        ring = [ring for ring in mol.topo.rings if {a1, a2}.issubset(ring)][0]
        n_atoms_in_ring = sum([atom in self.atomids for atom in ring])

        conj_bonds_in_ring = np.any(mol.topo.b_order_matrix[ring][:, ring] > 1.25)

        if n_atoms_in_ring != 1 or conj_bonds_in_ring:
            not_breakable = True
        else:
            not_breakable = False
        return not_breakable

    def identify_fragment(self, mol, config):
        n_neigh, n_cap = 0, 0
        possible_h_caps = {i: [] for i in range(mol.n_atoms)}
        next_neigh = [[a, n] for a in self.atomids for n
                      in mol.topo.neighbors[0][a] if n not in self.atomids]
        while next_neigh != []:
            new = []
            for a, n in next_neigh:
                bond = mol.topo.edge(a, n)
                if n in self.atomids:
                    pass
                elif (config.frag_threshold < 1 or  # fragmentation turned off
                      n_neigh < config.frag_threshold  # don't break first n neighbors
                      or bond['order'] >= 1.4  # don't break bonds conjugated more than 1.4
                      or ELE_ENEG[mol.elements[a]] > 3  # don't break if very electronegative
                      or (config.break_co_bond and ELE_ENEG[mol.elements[n]] > 3)
                      or mol.topo.n_neighbors[n] == 1  # don't break terminal atoms
                      or bond['n_rings'] > 1  # don't break a bond that is in multiple rings
                      or (bond['n_rings'] == 1 and self.check_single_ring_rules(mol, bond, a, n))):

                    new.append(n)
                    self.atomids.append(n)
                    if mol.topo.node(n)['elem'] == 1:
                        possible_h_caps[a].append(n)
                else:
                    bl = mol.topo.edge(a, n)['length']
                    new_bl = ELE_COV[mol.topo.elements[a]] + ELE_COV[1]
                    vec = mol.topo.node(a)['coords'] - mol.topo.node(n)['coords']
                    coord = mol.topo.coords[a] - vec/bl*new_bl
                    self.caps.append({'connected': a, 'idx': n, 'n_cap': n_cap, 'coord': coord,
                                      'b_length': bl})
                    n_cap += 1
            next_neigh = [[a, n] for a in new for n in mol.topo.neighbors[0][a] if n not in
                          self.atomids]
            n_neigh += 1

        self.n_atoms_without_cap = len(self.atomids)
        self.n_atoms = self.n_atoms_without_cap + len(self.caps)

        # self.remove_non_bonded = [cap['idx'] for cap in self.caps]
        # for cap in self.caps:
        #     hydrogens = possible_h_caps[cap['connected']]
        #     for h in hydrogens:
        #         if h not in self.remove_non_bonded:
        #             self.remove_non_bonded.append(h)

    def make_fragment_graph(self, mol):
        self.map_mol_to_frag = {self.atomids[i]: i for i in range(self.n_atoms_without_cap)}
        self.scanned_atomids = [self.map_mol_to_frag[a] for a in self.scanned_atomids]
        self.elements = [mol.elements[idx] for idx in self.atomids+[cap['idx'] for cap in
                                                                    self.caps]]
        self.graph = mol.topo.graph.subgraph(self.atomids)
        self.graph = nx.relabel_nodes(self.graph, self.map_mol_to_frag)

        for atom in self.scanned_atomids:
            self.graph.nodes[atom]['scan'] = True
        self.graph.graph['n_atoms'] = self.n_atoms
        self.graph.graph['scan'] = [scanned+1 for scanned in self.scanned_atomids]

        for _, _, d in self.graph.edges(data=True):
            for att in ['vector', 'length', 'order', 'vers', 'in_ring3', 'in_ring']:
                d.pop(att, None)
        for _, d in self.graph.nodes(data=True):
            for att in ['n_ring']:
                d.pop(att, None)

        for cap in self.caps:
            self.atomids.append(cap['idx'])
            self.map_mol_to_frag[cap['idx']] = self.n_atoms_without_cap + cap['n_cap']
            h_type = f'1(1.0){mol.topo.elements[cap["connected"]]}'
            self.graph.add_node(self.n_atoms_without_cap + cap['n_cap'], elem=1, n_bonds=1,
                                coords=cap['coord'], capping=True)
            self.graph.add_edge(self.n_atoms_without_cap + cap['n_cap'],
                                self.map_mol_to_frag[cap["connected"]], type=h_type)

            cap['idx'] = self.map_mol_to_frag[cap['idx']]
            cap['connected'] = self.map_mol_to_frag[cap['connected']]

    def make_fragment_identifier(self, config, mol, qm):
        atom_ids = [[], []]
        comp_dict = {i: 0 for i in set(self.elements[:self.n_atoms_without_cap])}
        if 1 not in comp_dict.keys() and len(self.caps) > 0:
            comp_dict[1] = 0

        for a in range(2):
            neighbors = nx.bfs_tree(self.graph, a, depth_limit=4).nodes
            for n in neighbors:
                paths = nx.all_simple_paths(self.graph, a, n, cutoff=4)
                for path in map(nx.utils.pairwise, paths):
                    types = [self.graph.edges[edge]['type'] for edge in path]
                    atom_ids[a].append("-".join(types))
            atom_ids[a] = "_".join(sorted(atom_ids[a]))
        frag_hash = "=".join(sorted(atom_ids))
        frag_hash = hashlib.md5(frag_hash.encode()).hexdigest()

        multiplicity = 1
        charge = int(round(sum(nx.get_node_attributes(self.graph, 'q').values())))
        n_electrons = sum(self.elements[:self.n_atoms_without_cap])+len(self.caps)
        if config.frag_threshold < 1:  # If fragmentation is off - take molecule's charge&multi
            charge = qm.config.charge
            multiplicity = qm.config.multiplicity
        elif (n_electrons + charge) % 2 == 1:
            multiplicity = 2

        qm_method = qm.method.copy()
        qm_method.update({'charge': charge, 'multiplicity': multiplicity})
        self.graph.graph['qm_method'] = qm_method

        composition = ""
        s1, s2 = sorted([ATOM_SYM[elem] for elem in self.elements[:2]])
        for elem in nx.get_node_attributes(self.graph, 'elem').values():
            comp_dict[elem] += 1
        for elem in sorted(comp_dict):
            composition += f"{ATOM_SYM[elem]}{comp_dict[elem]}"
        frag_id = f"{s1}{s2}_{composition}_{frag_hash}"
        self.hash = frag_id = frag_id.replace('(', '-').replace(',', '').replace(')', '')

    def check_for_fragment(self, job, config, qm):
        """
        Check if fragment exists in the database
        If not, check current fragment directory if new data is there
        """
        self.dir = f'{config.frag_lib}/{self.hash}'
        self.map_frag_to_db = {i: i for i in range(self.n_atoms)}
        have_match = False

        nm = iso.categorical_node_match(['elem', 'n_bonds', 'capping', 'scan'],
                                        [0, 0, 0, False, False])
        em = iso.categorical_edge_match(['type'], [0])

        os.makedirs(self.dir, exist_ok=True)
        identifiers = [i for i in sorted(os.listdir(f'{self.dir}')) if i.startswith('ident')]

        for id_no, id_file in enumerate(identifiers, start=1):
            with open(f"{self.dir}/{id_file}", 'rb') as f:
                compared = pickle.load(f)

            GM = iso.GraphMatcher(self.graph, compared, node_match=nm, edge_match=em)
            if self.graph.graph['qm_method'] == compared.graph['qm_method'] and GM.is_isomorphic():
                if os.path.isfile(f'{self.dir}/scandata_{id_no}'):
                    self.has_data = True
                    self.map_frag_to_db = GM.mapping
                else:
                    # has_inp to mean that the input file has been generated
                    # But the scan data has not been collected yet.
                    # This variable is set such that the same input file will
                    # not be generated twice when batch_run = True
                    self.has_inp = True
                have_match = True
                self.hash_idx = id_no
                break

        if not have_match:
            self.hash_idx = len(identifiers)+1
        self.id = f'{self.hash}~{self.hash_idx}'

    def check_new_scan_data(self, job, mol, config, qm):
        files = [f for f in os.listdir(job.frag_dir) if f.startswith(self.id) and
                 f.endswith(('log', 'out'))]

        if files:
            self.has_data = True
            qm_out = qm.read_scan(files)
            self.qm_energies = qm_out.energies
            self.qm_coords = qm_out.coords
            self.assign_frag_charge(mol, qm_out.charges)

            if qm_out.mismatch:
                if config.avail_only:
                    print('"\navail_only" requested, attempting to continue with the missing '
                          'points...\n\n')
                else:
                    sys.exit('Exiting...\n\n')
            else:
                with open(f'{self.dir}/scandata_{self.hash_idx}', 'w') as data:
                    for angle, energy in zip(qm_out.angles, qm_out.energies):
                        data.write(f'{angle:>10.3f} {energy:>20.8f}\n')
                with open(f"{self.dir}/charges_{self.hash_idx}", 'w') as file:
                    json.dump(qm_out.charges, file, sort_keys=True, indent=4)
                np.save(f'{self.dir}/scancoords_{self.hash_idx}.npy', qm_out.coords)

    def write_xyz(self):
        atomids = [atomid+1 for atomid in self.scanned_atomids]
        with open(f'{self.dir}/coords_{self.hash_idx}.xyz', 'w') as xyz:
            xyz.write(f'{self.n_atoms}\n')
            xyz.write('Scanned atoms: {} {} {} {}\n'.format(*atomids))
            for data in sorted(self.graph.nodes.data()):
                atom_name, [c1, c2, c3] = ATOM_SYM[data[1]['elem']], data[1]['coords']
                xyz.write(f'{atom_name:>3s} {c1:>12.6f} {c2:>12.6f} {c3:>12.6f}\n')

    def make_fragment_terms(self, mol):
        map_mol_to_db = {}
        map_db_to_frag = {v: k for k, v in self.map_frag_to_db.items()}

        self.elements = [self.elements[map_db_to_frag[i]] for i in range(self.n_atoms)]
        self.scanned_atomids = [self.map_frag_to_db[s] for s in self.scanned_atomids]

        for id_mol, id_frag in self.map_mol_to_frag.items():
            map_mol_to_db[id_mol] = self.map_frag_to_db[id_frag]

        map_db_to_mol = {v: k for k, v in map_mol_to_db.items()}
        self.terms = mol.terms.subset(self.atomids, map_mol_to_db,
                                      remove_non_bonded=self.remove_non_bonded)
        self.non_bonded = mol.non_bonded.subset(mol.non_bonded, self.frag_charges, map_mol_to_db)
        self.remove_non_bonded = [map_mol_to_db[i] for i in self.remove_non_bonded]

        for cap in self.caps:
            cap['idx'] = self.map_frag_to_db[cap['idx']]
            cap['connected'] = self.map_frag_to_db[cap['connected']]

            self.non_bonded.lj_types[cap['idx']] = self.non_bonded.h_cap
            for term in self.terms['bond']:
                if cap['idx'] in term.atomids and cap['connected'] in term.atomids:
                    term.equ = 1.1
            # for term in self.terms['dihedral/flexible']:
            #     if cap['idx'] in term.atomids:
            #         self.terms.remove_terms_by_name(str(term), atomids=term.atomids)

        # Reorder neighbors
        self.neighbors = [[] for _ in range(3)]
        for n in range(3):
            for i in range(self.n_atoms):
                neighs = mol.topo.neighbors[n][map_db_to_mol[i]]
                self.neighbors[n].append([map_mol_to_db[neigh] for neigh in neighs if neigh in
                                          map_mol_to_db.keys() and
                                          map_mol_to_db[neigh] < self.n_atoms])

    def check_for_qm_data(self, job, config, mol, qm):
        if self.has_data:
            self.qm_energies = np.loadtxt(f'{self.dir}/scandata_{self.hash_idx}', unpack=True)[1]
            self.qm_coords = np.load(f'{self.dir}/scancoords_{self.hash_idx}.npy')

            if os.path.isfile(f'{self.dir}/charges_{self.hash_idx}'):
                with open(f'{self.dir}/charges_{self.hash_idx}') as file:
                    charges = json.load(file)
                self.assign_frag_charge(mol, charges)

        else:
            self.check_new_scan_data(job, mol, config, qm)
            self.write_have_or_missing(job, config)

            with open(f"{self.dir}/identifier_{self.hash_idx}", 'wb') as f:
                pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

            self.write_xyz()
            with open(f"{self.dir}/qm_method_{self.hash_idx}", 'w') as file:
                json.dump(self.graph.graph['qm_method'], file, sort_keys=True, indent=4)

            if not (self.has_data or (config.batch_run and self.has_inp)):
                self.make_qm_input(job, qm)

    def assign_frag_charge(self, mol, charges):
        if (self.charge_method in charges.keys() and
                not (self.ext_charges and self.use_ext_charges_for_frags)):
            self.frag_charges = np.array(charges[self.charge_method])
            if mol.charge == 0:
                self.frag_charges *= self.charge_scaling

    def write_have_or_missing(self, job, config):
        if self.has_data:
            status = 'have'
        elif config.batch_run and self.has_inp:
            status = 'generated'
        else:
            status = 'missing'

        data_path = f'{job.frag_dir}/{status}'
        with open(data_path, 'a+') as data_file:
            data_file.write(f'{self.id}\n')

    def make_qm_input(self, job, qm):
        coords, atnums = [], []
        for data in sorted(self.graph.nodes.data()):
            coords.append(data[1]['coords'])
            atnums.append(data[1]['elem'])

        coords = np.array(coords)
        start_angle = np.degrees(get_dihed(coords[self.scanned_atomids])[0])

        with open(f'{job.frag_dir}/{self.id}.inp', 'w') as file:
            qm.write_scan(file, self.id, coords, atnums, self.graph.graph['scan'], start_angle,
                          self.graph.graph['qm_method']['charge'],
                          self.graph.graph['qm_method']['multiplicity'])
