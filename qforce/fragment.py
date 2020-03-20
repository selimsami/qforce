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
from ase.optimize import BFGS
from ase import Atoms
#
from .elements import ELE_COV, ATOM_SYM
from .read_qm_out import QM
from .make_qm_input import make_qm_input
from .calculator import QForce
from .forces import get_dihed
from .forcefield import ForceField
from .dihedral_scan import scan_dihedral


def fragment(inp, mol, qm):
    fragments = []
    unique_dihedrals = {}

    reset_data_files(inp)

    for term in mol.terms['dihedral/flexible']:
        if str(term) not in unique_dihedrals:
            unique_dihedrals[str(term)] = term.atomids

    for name, atomids in unique_dihedrals.items():
        frag = Fragment(inp, mol, qm, atomids, name)

        if frag.has_data:
            fragments.append(frag)
        else:
            mol.terms.remove_terms_by_name(name=frag.name)

    check_and_notify(inp, len(unique_dihedrals), len(fragments))

    n_runs = 2
    ignore = ['dihedral/flexible', 'dihedral/constr']
    for n_run in range(n_runs):
        fit_dihedrals(inp, mol, fragments, n_run, ignores=ignore, nsteps=1000)
        ignore = []

    for frag in fragments:
        ff = ForceField(inp, frag, frag.coords[0], f'{inp.frag_dir}/{frag.id}')
        ff.write_gromacs(inp, frag)
        scan_dihedral(inp, frag.scanned_atomids, frag.id, frag=True)

    return fragments


def reset_data_files(inp):
    for data in ['missing', 'have']:
        data_path = f'{inp.frag_dir}/{data}'
        if os.path.exists(data_path):
            os.remove(data_path)


def check_and_notify(inp, n_unique, n_have):
    n_missing = n_unique - n_have
    if n_unique == 0:
        print('There are no flexible dihedrals.')
    else:
        print(f"There are {n_unique} unique flexible dihedrals.")
    if n_missing == 0:
        print(f"All scan data is available. Fitting the dihedrals...\n")
    else:
        print(f"{n_missing} of them are missing the scan data.")
        print(f"QM input files for them are created in: {inp.frag_dir}")

        if inp.fragment == 'available':
            print('Continuing without the missing dihedrals...\n')
        else:
            print('Exiting...\n')
            sys.exit()


def fit_dihedrals(inp, mol, fragments, n_run, ignores=[], nsteps=50):
    n_fitted = 1
    for frag in fragments:
        print(f'\nFitting dihedral {n_fitted}: {frag.id} \n')

        scanned = list(frag.terms.get_terms_from_name(name=frag.name,
                                                      atomids=frag.scanned_atomids))[0]
        scanned.equ = np.zeros(6)

        frag.calc_dihedral_function(inp, mol, ignores, nsteps, n_run)
        n_fitted += 1

        for frag2 in fragments:
            for term in frag2.terms.get_terms_from_name(frag.name):
                term.equ = frag.params

        for term in mol.terms.get_terms_from_name(frag.name):
            term.equ = frag.params

        print(frag.params)


class Fragment():
    def __init__(self, inp, mol, qm, scanned_atomids, name):
        self.scanned_atomids = scanned_atomids
        self.atomids = list(self.scanned_atomids[1:3])
        self.name = name
        self.capping_h = []
        self.n_atoms = 0
        self.n_atoms_with_capping = 0
        self.hash = ''
        self.hash_idx = 0
        self.id = ''
        self.has_data = False
        self.mapping_frag_to_db = {}
        self.mapping_mol_to_frag = {}
        self.elems = []
        self.terms = None
        self.params = None
        self.r_squared = None
        self.capping_atoms = []

        self.check_fragment(inp, mol, qm)

    def check_fragment(self, inp, mol, qm):
        self.identify_fragment(mol)
        self.make_fragment_graph(qm, mol)
        self.make_fragment_identifier(inp, mol)
        self.check_for_fragment(inp)
        self.make_fragment_terms(inp, qm, mol)
        self.write_have_or_missing(inp)

        if not self.has_data:
            make_qm_input(inp, self.graph, self.id)

    def identify_fragment(self, mol):
        n_neigh = 0
        next_neigh = [[a, n] for a in self.atomids for n
                      in mol.topo.neighbors[0][a] if n not in self.atomids]
        while next_neigh != []:
            new = []
            for a, n in next_neigh:
                if n in self.atomids:
                    pass
                elif (n_neigh < 3 or not mol.topo.edge(a, n)['breakable']
                      or not mol.topo.node(n)['breakable']):
                    new.append(n)
                    self.atomids.append(n)
                else:
                    bl = mol.topo.edge(a, n)['length']
                    new_bl = ELE_COV[mol.topo.atomids[a]] + ELE_COV[1]
                    vec = mol.topo.node(a)['coords'] - mol.topo.node(n)['coords']
                    self.capping_atoms.append(n)
                    self.capping_h.append((a, mol.topo.coords[a] - vec/bl*new_bl))
            next_neigh = [[a, n] for a in new for n in mol.topo.neighbors[0][a] if n not in
                          self.atomids]
            n_neigh += 1
        self.n_atoms = len(self.atomids)
        self.n_atoms_with_capping = self.n_atoms + len(self.capping_h)

    def make_fragment_graph(self, qm, mol):
        self.mapping_mol_to_frag = {self.atomids[i]: i for i in range(self.n_atoms)}
        self.scanned_atomids = [self.mapping_mol_to_frag[a] for a in self.scanned_atomids]
        self.elems = [qm.atomids[idx] for idx in self.atomids+self.capping_atoms]
        self.graph = mol.topo.graph.subgraph(self.atomids)
        self.graph = nx.relabel_nodes(self.graph, self.mapping_mol_to_frag)
        self.graph.edges[[0, 1]]['scan'] = True
        self.graph.graph['n_atoms'] = self.n_atoms
        self.graph.graph['scan'] = [scanned+1 for scanned in self.scanned_atomids]

        for _, _, d in self.graph.edges(data=True):
            for att in ['vector', 'length', 'order', 'breakable', 'vers', 'in_ring3', 'in_ring']:
                d.pop(att, None)
        for _, d in self.graph.nodes(data=True):
            for att in ['breakable', 'q', 'n_ring']:
                d.pop(att, None)

        for i, cap in enumerate(self.capping_atoms):
            self.atomids.append(cap)
            self.mapping_mol_to_frag[cap] = i + self.n_atoms

        for i, h in enumerate(self.capping_h):
            h_type = f'1(1.0){mol.topo.atomids[h[0]]}'
            self.graph.add_node(self.n_atoms+i, elem=1, n_bonds=1, lone_e=0, coords=h[1],
                                capping=True)
            self.graph.add_edge(self.n_atoms+i, self.mapping_mol_to_frag[h[0]], type=h_type)
        self.graph.graph['n_atoms'] = self.n_atoms_with_capping

    def make_fragment_identifier(self, inp, mol):
        atom_ids = [[], []]
        comp_dict = {i: 0 for i in mol.topo.atomids}
        mult = 1
        composition = ""
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

        charge = int(round(sum(nx.get_node_attributes(self.graph, 'q').values())))
        s1, s2 = sorted([ATOM_SYM[elem] for elem in self.elems[:2]])
        self.graph.graph['charge'] = charge
        if (sum(mol.topo.atomids) + charge) % 2 == 1:
            mult = 2
        self.graph.graph['mult'] = mult
        for elem in nx.get_node_attributes(self.graph, 'elem').values():
            comp_dict[elem] += 1
        for elem in sorted(comp_dict):
            composition += f"{ATOM_SYM[elem]}{comp_dict[elem]}"
        if inp.disp == '':
            disp = ''
        else:
            disp = f'-{inp.disp}'
        frag_id = (f"{s1}{s2}_{composition}_{charge}_{mult}_{inp.method}"
                   f"{disp}_{inp.basis}_{frag_hash}")
        self.hash = frag_id = frag_id.replace('(', '-').replace(',', '').replace(')', '')

    def check_for_fragment(self, inp):
        """
        Check if fragment exists in the database
        If not, check current fragment directory if new data is there
        """
        self.dir = f'{inp.frag_lib}/{self.hash}'
        self.mapping_frag_to_db = {i: i for i in range(self.n_atoms)}
        have_match = False

        nm = iso.categorical_node_match(['elem', 'n_bonds', 'lone_e', 'capping'], [0, 0, 0, False])
        em = iso.categorical_edge_match(['type', 'scan'], [0, False])

        os.makedirs(self.dir, exist_ok=True)
        identifiers = [i for i in sorted(os.listdir(f'{self.dir}')) if i.startswith('ident')]

        for id_no, id_file in enumerate(identifiers, start=1):
            compared = nx.read_gpickle(f"{self.dir}/{id_file}")
            GM = iso.GraphMatcher(self.graph, compared, node_match=nm, edge_match=em)
            if GM.is_isomorphic():
                if os.path.isfile(f'{self.dir}/scandata_{id_no}'):
                    self.has_data = True
                    self.mapping_frag_to_db = GM.mapping
                have_match = True
                self.hash_idx = id_no
                break

        if not have_match:
            self.hash_idx = len(identifiers)+1

        self.id = f'{self.hash}~{self.hash_idx}'

        if not self.has_data:
            self.check_new_scan_data(inp)
            nx.write_gpickle(self.graph, f"{self.dir}/identifier_{self.hash_idx}")
            self.write_xyz()

    def check_new_scan_data(self, inp):
        outs = [f for f in os.listdir(inp.frag_dir) if f.startswith(self.id) and
                f.endswith(('log', 'out'))]
        for out in outs:
            try:
                qm = QM(inp, 'scan', out_file=f'{inp.frag_dir}/{out}')
                sucessfully_read = True
            except Exception:
                sucessfully_read = False
            if sucessfully_read and qm.normal_term:
                self.has_data = True
                with open(f'{self.dir}/scandata_{self.hash_idx}', 'w') as data:
                    for angle, energy in zip(qm.angles, qm.energies):
                        data.write(f'{angle:>10.3f} {energy:>20.8f}\n')
                np.save(f'{self.dir}/scancoords_{self.hash_idx}.npy', qm.coords)
            else:
                print(f'WARNING: Following scan has not been terminated sucessfully:\n{out}\n'
                      'Skipping it...\n')

    def write_xyz(self):
        with open(f'{self.dir}/coords_{self.hash_idx}.xyz', 'w') as xyz:
            xyz.write(f'{self.n_atoms}\n')
            xyz.write('Scanned atoms: {} {} {} {}\n'.format(*self.scanned_atomids))
            for data in sorted(self.graph.nodes.data()):
                atom_name, [c1, c2, c3] = ATOM_SYM[data[1]['elem']], data[1]['coords']
                xyz.write(f'{atom_name:>3s} {c1:>12.6f} {c2:>12.6f} {c3:>12.6f}\n')

    def make_fragment_terms(self, inp, qm, mol):
        mapping_mol_to_db = {}

        for i in range(self.n_atoms, self.n_atoms_with_capping+1):
            self.mapping_frag_to_db[i] = i

        mapping_db_to_frag = {v: k for k, v in self.mapping_frag_to_db.items()}

        self.elems = [self.elems[mapping_db_to_frag[i]] for i in range(self.n_atoms_with_capping)]

        self.scanned_atomids = [self.mapping_frag_to_db[s] for s in self.scanned_atomids]

        for id_mol, id_frag in self.mapping_mol_to_frag.items():
            mapping_mol_to_db[id_mol] = self.mapping_frag_to_db[id_frag]

        self.terms = mol.terms.subset(self.atomids, mapping_mol_to_db)
        self.non_bonded = mol.non_bonded.subset(inp, self.atomids, mol.non_bonded,
                                                mapping_mol_to_db)

    def write_have_or_missing(self, inp):
        if self.has_data:
            status = 'have'
        else:
            status = 'missing'
        data_path = f'{inp.frag_dir}/{status}'
        with open(data_path, 'a+') as data_file:
            data_file.write(f'{self.id}\n')

    def calc_dihedral_function(self, inp, mol, ignores, nsteps, n_run):
        md_energies = []

        angles, qm_energies = np.loadtxt(f'{self.dir}/scandata_{self.hash_idx}', unpack=True)

        if n_run == 0:
            self.coords = np.load(f'{self.dir}/scancoords_{self.hash_idx}.npy')

        angles_rad = np.radians(angles)

        for i, (angle, qm_energy, coord) in enumerate(zip(angles_rad, qm_energies, self.coords)):
            restraints = []
            for term in self.terms['dihedral/flexible']:
                phi0 = get_dihed(coord[term.atomids])[0]
                restraints.append([term.atomids, phi0])
            restraints.append([np.array(self.scanned_atomids), angle])

            frag = Atoms(self.elems, positions=coord,
                         calculator=QForce(self.terms, ignores, dihedral_restraints=restraints))

            # if n_run != 0:
            traj_name = f'{inp.frag_dir}/{self.id}_{np.degrees(angle).round()}.traj'
            log_name = f'{inp.frag_dir}/opt_{self.id}.log'
            e_minimiz = BFGS(frag, trajectory=traj_name, logfile=log_name)
            e_minimiz.run(fmax=0.01, steps=nsteps)
            self.coords[i] = frag.get_positions()

            md_energies.append(frag.get_potential_energy())

        md_energies = np.array(md_energies)
        energy_diff = qm_energies - md_energies
        energy_diff -= energy_diff.min()

        weights = 1/np.exp(-0.2 * np.sqrt(qm_energies))
        self.params = curve_fit(calc_rb, angles_rad, energy_diff, absolute_sigma=False,
                                sigma=weights)[0]
        self.calc_r_squared(calc_rb, angles_rad, energy_diff)

        md_energies += calc_rb(angles_rad, *self.params)
        md_energies -= md_energies.min()
        self.plot_results(inp, angles, qm_energies, md_energies)

    def plot_results(self, inp, angles, qm_energies, md_energies):
        width, height = plt.figaspect(0.6)
        f = plt.figure(figsize=(width, height), dpi=300)
        sns.set(font_scale=1.3)
        plt.title(f'R-squared = {round(self.r_squared, 3)}', loc='left')
        plt.xlabel('Angle')
        plt.ylabel('Energy (kJ/mol)')
        plt.plot(angles, qm_energies, linewidth=4, label='QM')
        plt.plot(angles, md_energies, linewidth=4, label='Q-Force')
        plt.xticks(np.arange(0, 361, 60))
        plt.tight_layout()
        plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
        f.savefig(f"{inp.frag_dir}/scan_data_{self.id}.pdf", bbox_inches='tight')
        plt.close()

    def calc_r_squared(self, funct, x, y):
        residuals = y - funct(x, *self.params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        self.r_squared = 1 - (ss_res / ss_tot)


def calc_rb(angles, c0, c1, c2, c3, c4, c5):
    params = [c0, c1, c2, c3, c4, c5]

    rb = np.full(len(angles), c0)
    for i in range(1, 6):
        rb += params[i] * np.cos(angles-np.pi)**i
    return rb
