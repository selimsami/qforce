import numpy as np
import networkx as nx
import itertools
from .elements import elements
from .forces import get_dist, get_angle, get_dihed


class Terms():
    def __init__(self, scan=False):
        self.atoms = []
        self.types = []
        self.minima = []
        self.term_ids = []
        self.n_terms = 0

    def add_term(self, atoms, minimum, t_type):
        if t_type not in self.types:
            self.term_ids.append(self.n_terms)
            self.types.append(t_type)
            self.n_terms += 1
        else:
            self.term_ids.append(self.types.index(t_type))
        self.atoms.append(atoms)
        self.minima.append(minimum)

# class Atoms():
#    def add(self, atom, atom_id):
#        self.graph.add_node(atom, atom_id)
#        atoms.types , atoms.ids, atoms, elements, etc


class Angles(Terms):
    def __init__(self):
        Terms.__init__(self)
        self.urey = Terms()
        self.cross = Terms()


class Dihedrals(Terms):
    def __init__(self):
        Terms.__init__(self)
        self.rigid = Terms()
        self.flex = Terms(scan=True)
        self.imp = Terms()
        self.constr = Terms()

    def add_rigid(self, mol, atoms):
        phi = get_dihed(mol.coords[atoms])[0]
        d_type = self.get_type(mol, *atoms)
        self.rigid.add_term(atoms, phi, d_type)

    def add_constr(self, atoms, phi, d_type):
        # find multiplicity - QM - MD (LJ/charges) profile to get minima ?
        self.constr.add_term(atoms, phi, d_type)

    def add_flex(self, mol, atoms_combin):
        e = elements()
        heaviest = 0
        for a1, a2, a3, a4 in atoms_combin:
            mass = e.mass[mol.atomids[a1]] + e.mass[mol.atomids[a4]]
            if mass > heaviest:
                atoms = np.array([a1, a2, a3, a4])
                heaviest = mass
        self.flex.add_term(atoms, 0, mol.edge(a2, a3)['vers'])

    def get_type(self, mol, a1, a2, a3, a4):
        b12 = mol.edge(a1, a2)["vers"]
        b23 = mol.edge(a2, a3)["vers"]
        b43 = mol.edge(a4, a3)["vers"]
        t23 = [mol.types[a2], mol.types[a3]]
        t12 = f"{mol.types[a1]}({b12}){mol.types[a2]}"
        t43 = f"{mol.types[a4]}({b43}){mol.types[a3]}"
        d_type = [t12, t43]

        if t12 > t43:
            d_type.reverse()
            t23.reverse()
        d_type = f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"
        return d_type


class Molecule():
    """
    Scope:
    ------
    self.list : atom numbers of unique atoms grouped together
    self.atoms : unique atom numbers of each atom
    self.types : atom types of each atom
    self.neighbors : First 3 neighbors of each atom


    To do:
    -----
    - 180 degree angle = NO DIHEDRAL! (can/should avoid this in impropers?)

    - Since equivalent rigid dihedrals not necessarily have the same angle,
      can't average angles? Like in C60. But is it a bug or a feature? :)

    - Improper scan requires removal of relevant proper dihedrals from
      Redundant Coordinates in Gaussian

    - Think about cis/trans and enantiomers

    """
    def __init__(self, coords, atomids, inp, qm=None):
        self.n_atoms = len(atomids)
        self.atomids = atomids
        self.coords = coords
        self.atoms = np.zeros(self.n_atoms, dtype='int8')
        self.list = []
        self.conj = []
        self.pair_list = []
        self.pair_int = []
        self.double = []
        self.n_types = 0
        self.n_terms = 0
        self.neighbors = [[[] for j in range(self.n_atoms)] for i in range(3)]
        self.thole = []
        self.polar = []
        self.fragments = []

        self.bonds = Terms()
        self.angles = Angles()
        self.dih = Dihedrals()

        self.find_bonds_and_rings(qm)
        self.find_atom_types(inp.n_equiv)
        self.find_neighbors()
        self.find_parameter_types(inp)
        self.prepare()
        self.polarize()

    def find_bonds_and_rings(self, qm):
        e = elements()
        self.graph = nx.Graph()
        for i, i_id in enumerate(self.atomids):
            self.graph.add_node(i, elem=i_id, n_bonds=qm.n_bonds[i],
                                lone_e=qm.lone_e[i], q=qm.cm5[i],
                                coords=self.coords[i])
            # Check electronegativity difference to H to see if breakable
            eneg_diff = abs(e.eneg[i_id] - e.eneg[1])
            if eneg_diff > 0.5 or eneg_diff == 0:
                self.node(i)['breakable'] = False
            else:
                self.node(i)['breakable'] = True
            # add bonds
            for j, j_id in enumerate(self.atomids):
                order = qm.b_orders[i, j]
                if order > 0:
                    id1, id2 = sorted([i_id, j_id])
                    vec = self.coords[i] - self.coords[j]
                    dist = np.sqrt((vec**2).sum())
                    self.graph.add_edge(i, j, vector=vec, length=dist,
                                        order=order,
                                        type=f'{id1}({order}){id2}')
            if qm.n_bonds[i] > e.maxb[i_id]:
                print(f"WARNING: Atom {i+1} ({e.sym[i_id]}) has too many",
                      " ({qm.n_bonds[i]}) bonds?")
            elif qm.n_bonds[i] == 0:
                print(f"WARNING: Atom {i+1} ({e.sym[i_id]}) has no bonds")
        # add rings
        self.rings = nx.minimum_cycle_basis(self.graph)
        self.rings3 = [r for r in self.rings if len(r) == 3]
        for i in range(self.n_atoms):
            self.node(i)['n_ring'] = sum([i in ring for ring in self.rings])

        for atoms in self.graph.edges:
            self.edge(*atoms)['in_ring'] = any(set(atoms).issubset(set(ring))
                                               for ring in self.rings)
            self.edge(*atoms)['in_ring3'] = any(set(atoms).issubset(set(ring))
                                                for ring in self.rings3)

    def find_atom_types(self, n_eq):
        e = elements()
        atom_ids = [[] for _ in range(self.n_atoms)]

        for i in range(self.n_atoms):
            neighbors = nx.bfs_tree(self.graph, i, depth_limit=n_eq).nodes
            for n in neighbors:
                paths = nx.all_simple_paths(self.graph, i, n, cutoff=n_eq)
                for path in map(nx.utils.pairwise, paths):
                    types = [self.edge(*edge)['type'] for edge in path]
                    atom_ids[i].append("-".join(types))
                atom_ids[i].sort()
#            print('\n\n\n')
#            print(f'-----{i+1}-----')
#            print(atom_ids[i])

        if n_eq < 0:
            atom_ids = [i for i in range(self.n_atoms)]

        for n in range(self.n_atoms):
            if n in [item for sub in self.list for item in sub]:
                continue
            eq = [i for i, a_id in enumerate(atom_ids) if atom_ids[n] == a_id]
            self.list.append(eq)
            self.atoms[eq] = self.n_types
            self.n_types += 1

        types = {i: 1 for i in set(self.atomids)}
        self.types = [None for _ in self.atomids]
        for eq in self.list:
            for i in eq:
                self.types[i] = "{}{}".format(e.sym[self.atomids[i]],
                                              types[self.atomids[i]])
            types[self.atomids[eq[0]]] += 1

    def find_parameter_types(self, inp):
        bonds, angles, dihedrals = [], [], []

        for i in range(self.n_atoms):
            neighbors = nx.bfs_tree(self.graph, source=i,  depth_limit=3).nodes
            for n in neighbors:
                paths = nx.all_simple_paths(self.graph, i, n, cutoff=3)
                for path in paths:
                    if len(path) == 2 and path[0] < path[1]:
                        bonds.append(path)
                    elif len(path) == 3 and path[0] < path[2]:
                        angles.append(path)
                    elif len(path) == 4 and path[1] < path[2]:
                        dihedrals.append(path)
        dihedrals.sort(key=lambda x: [x[1], x[2], x[0], x[3]])

        for a1, a2 in bonds:
            bond = self.edge(a1, a2)
            dist = bond['length']
            type1, type2 = sorted([self.types[a1], self.types[a2]])
            bond['vers'] = f"{type1}({bond['order']}){type2}"
            if bond['order'] > 1.5 or bond["in_ring"]:
                bond['breakable'] = False
            else:
                bond['breakable'] = True
            self.bonds.add_term([a1, a2], dist, bond['vers'])

        for a1, a2, a3 in angles:
            vec12, dist12 = get_dist(self.coords[a1], self.coords[a2])
            vec32, dist32 = get_dist(self.coords[a3], self.coords[a2])
            theta = get_angle(vec12, vec32)

            b21 = self.edge(a2, a1)['vers']
            b23 = self.edge(a2, a3)['vers']
            a_type = sorted([f"{self.types[a2]}({b21}){self.types[a1]}",
                             f"{self.types[a2]}({b23}){self.types[a3]}"])
            a_type = f"{a_type[0]}_{a_type[1]}"
            self.angles.add_term([a1, a2, a3], theta, a_type)

            dist13 = get_dist(self.coords[a1], self.coords[a3])[1]

            if inp.urey:
                self.angles.urey.add_term([a1, a3], dist13, a_type)
            if inp.cross:
                self.angles.cross.add_term([a1, a2, a3], np.array([dist12,
                                           dist32, dist13]), a_type)

        # dihedrals
        for a2, a3 in bonds:
            central = self.edge(a2, a3)
            a1s = [a1 for a1 in self.neighbors[0][a2] if a1 != a3]
            a4s = [a4 for a4 in self.neighbors[0][a3] if a4 != a2]

            if a1s == [] or a4s == []:
                continue

            atoms_comb = [list(d) for d in itertools.product(a1s, [a2], [a3],
                          a4s) if d[0] != d[-1]]
            if (central['order'] > 1.5 or central["in_ring3"]
                    or (central['in_ring'] and central['order'] > 1)
                    or all([self.node(a)['n_ring'] > 2 for a in [a2, a3]])):
                for atoms in atoms_comb:
                    self.dih.add_rigid(self, atoms)
            elif central['in_ring']:
                atoms_r = [a for a in atoms_comb if any(set(a).issubset(set(r))
                           for r in self.rings)][0]
                phi = get_dihed(self.coords[atoms_r])[0]
                if abs(phi) < 0.07:
                    for atoms in atoms_comb:
                        self.dih.add_rigid(self, atoms)
                else:
                    d_type = self.dih.get_type(self, *atoms_r)
                    self.dih.add_constr(atoms_r, phi, d_type)
            else:
                self.dih.add_flex(self, atoms_comb)

        # find improper dihedrals
        for i in range(self.n_atoms):
            bonds = list(self.graph.neighbors(i))
            if len(bonds) != 3:
                continue
            atoms = [i, -1, -1, -1]
            n_bond = [len(list(self.graph.neighbors(b))) for b in bonds]
            non_ring = [a for a in bonds if not self.edge(i, a)['in_ring']]

            if len(non_ring) == 1:
                atoms[3] = non_ring[0]
            else:
                atoms[3] = bonds[n_bond.index(min(n_bond))]

            for b in bonds:
                if b not in atoms:
                    atoms[atoms.index(-1)] = b

            phi = get_dihed(self.coords[atoms])[0]

            # Only add improper dihedrals if there is no stiff dihedral
            # on the central improper atom and one of the neighbors
            bonds = [sorted([b, i]) for b in bonds]
            if any(b == a[1:3] for a in self.dih.rigid.atoms
                   for b in bonds):
                continue
            imp_type = f"ki_{self.types[i]}"
            if abs(phi) < 0.07:  # check planarity <4 degrees
                self.dih.imp.add_term(atoms, phi, imp_type)
            else:
                self.dih.add_constr(atoms, phi, imp_type)

    def find_neighbors(self):
        for i in range(self.n_atoms):
            neighbors = nx.bfs_tree(self.graph, source=i,  depth_limit=3).nodes
            for n in neighbors:
                paths = nx.all_shortest_paths(self.graph, i, n)
                for path in paths:
                    if len(path) == 2:
                        self.neighbors[0][i].append(path[-1])
                    elif len(path) == 3:
                        if path[-1] not in self.neighbors[1][i]:
                            self.neighbors[1][i].append(path[-1])
                    elif len(path) == 4:
                        if path[-1] not in self.neighbors[2][i]:
                            self.neighbors[2][i].append(path[-1])

    def prepare(self):
        for term in [self.bonds, self.angles.urey, self.angles.cross,
                     self.angles, self.dih.rigid, self.dih.imp]:
            term.term_ids = [i + self.n_terms for i in term.term_ids]
            self.n_terms += term.n_terms

    def polarize(self):
        """
        TEST for relax_drude.
        """
        a_const = 2.6
        polar_dict = {1: 0.45330, 6: 1.30300, 7: 0.98840, 8: 0.83690,
                      16: 2.47400}
#        polar_dict = { 1: 0.41383,  6: 1.45000,  7: 0.971573,
#                       8: 0.851973,  9: 0.444747, 16: 2.474448,
#                      17: 2.400281, 35: 3.492921, 53: 5.481056}  # 1: 0.413835

        for i in range(self.n_atoms):
            # add polarizability
            self.polar.append(polar_dict[self.atomids[i]])
            # add thole
            for a in self.neighbors[0][i]+self.neighbors[1][i]:
                # +self.neighbors[2][i]
                if i < a:
                    a1 = polar_dict[self.atomids[i]]
                    a2 = polar_dict[self.atomids[a]]
                    self.thole.append([i, a, a_const / (a1*a2)**(1./6.)])

    def node(self, i):
        return self.graph.nodes[i]

    def edge(self, i, j):
        return self.graph.edges[i, j]
