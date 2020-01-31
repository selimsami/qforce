import networkx as nx
import numpy as np
#
from ..elements import ATOM_SYM, ELE_ENEG, ELE_MAXB


class Topology(object):

    """"

    Contains all bonding etc. information of the system

    TODO:
        * include parts of qm, related to graph info

    """

    def __init__(self, atomids, coords, qm, n_equiv):
        self.n_atoms = len(atomids)
        self.atomids = atomids
        self.coords = coords
        #
        self.n_types = 0
        self.n_terms = 0
        #
        self.neighbors = [[[] for j in range(self.n_atoms)] for i in range(3)]
        self.list = []
        self.atoms = np.zeros(self.n_atoms, dtype='int8')
        self.pair_list = []
        #
        self._setup(qm, n_equiv)

    def _setup(self, qm, n_equiv):
        self._find_bonds_and_rings(qm)
        self._find_atom_types(n_equiv)
        self._find_neighbors()
        self._find_bonds_angles_dihedrals()

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

    def _find_atom_types(self, n_eq):
        atom_ids = [[] for _ in range(self.n_atoms)]

        for i in range(self.n_atoms):
            neighbors = nx.bfs_tree(self.graph, i, depth_limit=n_eq).nodes
            for n in neighbors:
                paths = nx.all_simple_paths(self.graph, i, n, cutoff=n_eq)
                for path in map(nx.utils.pairwise, paths):
                    types = [self.edge(*edge)['type'] for edge in path]
                    atom_ids[i].append("-".join(types))
                atom_ids[i].sort()

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
