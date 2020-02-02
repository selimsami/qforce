import networkx as nx
import numpy as np
import pulp
#
from ..elements import ATOM_SYM, ELE_ENEG, ELE_MAXB


class Topology(object):

    """"

    Contains all bonding etc. information of the system

    self.list : atom numbers of unique atoms grouped together
    self.atoms : unique atom numbers of each atom
    self.types : atom types of each atom
    self.neighbors : First 3 neighbors of each atom

    TODO:
        * include parts of qm, related to graph info

    """

    def __init__(self, inp, qm):
        self.n_equiv = inp.n_equiv
        self.n_excl = inp.n_excl
        self.atomids = qm.atomids
        self.n_atoms = len(self.atomids)
        self.coords = qm.coords
        #
        self.n_types = 0
        self.n_terms = 0
        #
        self.neighbors = [[[] for j in range(self.n_atoms)] for i in range(3)]
        self.list = []
        self.atoms = np.zeros(self.n_atoms, dtype='int8')
        self.pair_list = []
        #
        self._setup(qm)

    def _setup(self, qm):
        self._find_bonds_and_rings(qm)
        self._find_atom_types()
        self._find_neighbors()
        self._find_bonds_angles_dihedrals()
        self._handle_non_bonded_terms(qm)

    def _handle_non_bonded_terms(self, qm):
        self.q, self.c6, self.c12 = self._average_equivalent_terms([qm.q, qm.c6, qm.c12])
        self._calc_sigma_epsilon()
        self._sum_charges_to_qtotal()

    def _calc_sigma_epsilon(self):
        self.sigma = (self.c12/self.c6)**(1/6)
        self.epsilon = self.c6 / (4*self.sigma**6)

    def _average_equivalent_terms(self, terms):
        avg_terms = []
        for term in terms:
            avg_term = []
            term = np.array(term)
            for l in self.list:
                total = 0
                for a in l:
                    total += term[a]
                avg_term.append(round(total/len(l), 5))
            avg_terms.append(avg_term)
        return np.array(avg_terms)

    def _sum_charges_to_qtotal(self):
        total = sum([round(self.q[i], 5)*len(l) for i, l in enumerate(self.list)])
        q_integer = round(total)
        extra = int(100000 * round(total - q_integer, 5))
        if extra != 0:
            if extra > 0:
                sign = 1
            else:
                sign = -1
                extra = - extra

            n_eq = [len(l) for l in self.list]
            no = [f"{i:05d}" for i, _ in enumerate(n_eq)]

            var = pulp.LpVariable.dicts("x", no, lowBound=0, cat='Integer')
            prob = pulp.LpProblem('prob', pulp.LpMinimize)
            prob += pulp.lpSum([var[n] for n in no])
            prob += pulp.lpSum([eq * var[no[i]] for i, eq
                                in enumerate(n_eq)]) == extra
            prob.solve()

            if prob.status == 1:
                for i, v in enumerate(prob.variables()):
                    self.q[i] -= sign * v.varValue / 100000
            else:
                print('Failed to equate total of charges to the total charge of '
                      'the system. Do so manually')

    def _find_bonds_and_rings(self, qm):
        """Setup networkx graph """
        self.graph = nx.Graph()
        for iatom, iidx in enumerate(self.atomids):
            self.graph.add_node(iatom, elem=iidx, n_bonds=qm.n_bonds[iatom],
                                lone_e=qm.lone_e[iatom], q=qm.cm5[iatom],
                                coords=self.coords[iatom])
            # Check electronegativity difference to H to see if breakable
            eneg_diff = abs(ELE_ENEG[iidx] - ELE_ENEG[1])
            #
            if eneg_diff > 0.5 or eneg_diff == 0:
                self.node(iatom)['breakable'] = False
            else:
                self.node(iatom)['breakable'] = True
            # add bonds
            for jatom, jidx in enumerate(self.atomids):
                order = qm.b_orders[iatom, jatom]
                if order > 0:
                    id1, id2 = sorted([iidx, jidx])
                    vec = self.coords[iatom] - self.coords[jatom]
                    dist = np.sqrt((vec**2).sum())
                    self.graph.add_edge(iatom, jatom, vector=vec, length=dist,
                                        order=order,
                                        type=f'{id1}({order}){id2}')
            if qm.n_bonds[iatom] > ELE_MAXB[iidx]:
                print(f"WARNING: Atom {iatom+1} ({ATOM_SYM[iidx]}) has too many",
                      " ({qm.n_bonds[iatom]}) bonds?")
            elif qm.n_bonds[iatom] == 0:
                print(f"WARNING: Atom {iatom+1} ({ATOM_SYM[iidx]}) has no bonds")
        # add rings
        self.rings = nx.minimum_cycle_basis(self.graph)
        self.rings3 = [r for r in self.rings if len(r) == 3]
        for i in range(self.n_atoms):
            self.node(i)['n_ring'] = sum([i in ring for ring in self.rings])
        #
        for atoms in self.graph.edges:
            self.edge(*atoms)['in_ring'] = any(set(atoms).issubset(set(ring))
                                               for ring in self.rings)
            self.edge(*atoms)['in_ring3'] = any(set(atoms).issubset(set(ring))
                                                for ring in self.rings3)

    def _find_atom_types(self):
        atom_ids = [[] for _ in range(self.n_atoms)]

        for i in range(self.n_atoms):
            neighbors = nx.bfs_tree(self.graph, i, depth_limit=self.n_equiv).nodes
            for n in neighbors:
                paths = nx.all_simple_paths(self.graph, i, n, cutoff=self.n_equiv)
                for path in map(nx.utils.pairwise, paths):
                    types = [self.edge(*edge)['type'] for edge in path]
                    atom_ids[i].append("-".join(types))
                atom_ids[i].sort()

        if self.n_equiv < 0:
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
                self.types[i] = "{}{}".format(ATOM_SYM[self.atomids[i]],
                                              types[self.atomids[i]])
            types[self.atomids[eq[0]]] += 1

    def _find_neighbors(self):
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

    def _find_bonds_angles_dihedrals(self):

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
        #
        dihedrals.sort(key=lambda x: [x[1], x[2], x[0], x[3]])
        #
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def node(self, i):
        return self.graph.nodes[i]

    def edge(self, i, j):
        return self.graph.edges[i, j]
