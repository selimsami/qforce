import networkx as nx
import numpy as np
#
from ..elements import ATOM_SYM, ELE_MAXB
from ..forces import get_angle, get_dihed

class Topology(object):

    """"
    Contains all bonding etc. information of the system
    """

    def __init__(self, config, qm_out):
        self.n_equiv = config.n_equiv
        self.elements = qm_out.elements
        self.n_atoms = len(self.elements)
        self.coords = qm_out.coords
        self.b_order_matrix = qm_out.b_orders
        #
        self.n_types = 0
        self.n_terms = 0
        #
        self.neighbors = [[[] for j in range(self.n_atoms)] for i in range(3)]  # First 3 neighbors
        self.n_neighbors = []  # number of first neighbors for each atom
        self.list = []  # atom numbers of unique atoms grouped together
        self.types = [None for _ in self.elements]  # atom types of each atom
        self.unique_atomids = []  #
        self.atoms = np.zeros(self.n_atoms, dtype='int8')  # unique atom numbers of each atom
        self.all_rigid = config.all_rigid
        self.ba_couple_1_shared = config.ba_couple_1_shared
        #
        self._setup(qm_out)

    def _setup(self, qm_out):
        self._find_bonds_and_rings(qm_out)
        self._find_atom_types()
        self._find_neighbors()
        self._find_bonds_angles_dihedrals()

    def _find_bonds_and_rings(self, qm_out):
        """Setup networkx graph """
        self.graph = nx.Graph()
        for i_idx, i_elem in enumerate(self.elements):
            self.graph.add_node(i_idx, idx=i_idx, elem=i_elem, n_bonds=qm_out.n_bonds[i_idx],
                                q=qm_out.point_charges[i_idx], coords=self.coords[i_idx],
                                neighs=[], unique_neighs=[], nonrepeat_neighs=[], hybrid=None,
                                type=None, n_neighs=0, n_unique_neighs=0, n_nonrepeat_neighs=0)
            # add bonds
            for j_idx, j_elem in enumerate(self.elements):
                b_order = qm_out.b_orders[i_idx, j_idx]
                if b_order > 0.3:
                    id1, id2 = sorted([i_elem, j_elem])
                    b_order_half_rounded = np.round(b_order*2)/2
                    vec = self.coords[i_idx] - self.coords[j_idx]
                    dist = np.sqrt((vec**2).sum())
                    self.node(i_idx)['neighs'].append(j_idx)
                    self.node(i_idx)['n_neighs'] += 1
                    self.graph.add_edge(i_idx, j_idx, vector=vec, length=dist, order=b_order,
                                        type=f'{id1}({b_order_half_rounded}){id2}', n_rings=0)
            if qm_out.n_bonds[i_idx] > ELE_MAXB[i_elem]:
                print(f'WARNING: Atom {i_idx+1} ({ATOM_SYM[i_elem]}) has too many',
                      f' ({qm_out.n_bonds[i_idx]}) bonds?')
            elif qm_out.n_bonds[i_idx] == 0:
                print(f'WARNING: Atom {i_idx+1} ({ATOM_SYM[i_elem]}) has no bonds')

            if self.node(i_idx)['n_neighs'] == 1:
                self.node(i_idx)['hybrid'] = 'terminal'
            elif self.node(i_idx)['n_neighs'] == 2:
                theta = get_angle(self.coords[[self.node(i_idx)['neighs'][0], i_idx, self.node(i_idx)['neighs'][1]]])[0]

                if theta > 2.9671:  # if angle is larger than 170 degree, assume linear
                    self.node(i_idx)['hybrid'] = 'linear'
                else:
                    self.node(i_idx)['hybrid'] = 'bent'
            elif self.node(i_idx)['n_neighs'] == 3:
                phi = get_dihed(self.coords[[i_idx, self.node(i_idx)['neighs'][0], self.node(i_idx)['neighs'][1],
                                             self.node(i_idx)['neighs'][2]]])[0]
                if abs(phi) < 0.43625:
                    self.node(i_idx)['hybrid'] = 'planar'
                else:
                    self.node(i_idx)['hybrid'] = 'pyramidal'
            else:
                self.node(i_idx)['hybrid'] = 'tetrahedral'

        # add rings
        self.rings = nx.minimum_cycle_basis(self.graph)
        self.rings3 = [r for r in self.rings if len(r) == 3]

        for i in range(self.n_atoms):
            self.node(i)['n_ring'] = sum([i in ring for ring in self.rings])
        #
        for atoms in self.graph.edges:
            ring_members = [set(atoms).issubset(set(ring)) for ring in self.rings]
            self.edge(*atoms)['n_rings'] = sum(ring_members)
            self.edge(*atoms)['in_ring'] = any(ring_members)
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
            self.unique_atomids.append(n)
            self.n_types += 1

        types = {i: 1 for i in set(self.elements)}

        for eq in self.list:
            for i in eq:
                type = f"{ATOM_SYM[self.elements[i]]}{types[self.elements[i]]}"
                self.types[i] = type
                self.node(i)['type'] = type
            types[self.elements[eq[0]]] += 1
        self.types = np.array(self.types, dtype='str')

        for i in range(self.n_atoms):
            neigh_dict = {}
            for neigh in self.node(i)['neighs']:
                n_type = self.node(neigh)['type']
                if n_type in neigh_dict:
                    neigh_dict[n_type].append(neigh)
                else:
                    neigh_dict[n_type] = [neigh]

            for n_type, neighs in neigh_dict.items():
                if len(neighs) == 1:
                    self.node(i)['nonrepeat_neighs'].append(neighs[0])

            self.node(i)['neighs'] = sorted(self.node(i)['neighs'],  reverse=True,
                                                      key=lambda e: (self.node(e)['elem'], self.node(e)['n_neighs'], self.node(e)['idx']))
            self.node(i)['nonrepeat_neighs'] = sorted(self.node(i)['nonrepeat_neighs'],  reverse=True,
                                                      key=lambda e: (self.node(e)['elem'], self.node(e)['n_neighs'], self.node(e)['idx']))

            self.node(i)['unique_neighs'] = list(neigh_dict.values())
            self.node(i)['n_unique_neighs'] = len(neigh_dict)
            self.node(i)['n_nonrepeat_neighs'] = len(self.node(i)['nonrepeat_neighs'])

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
            self.n_neighbors.append(len(self.neighbors[0][i]))
        self.n_neighbors = np.array(self.n_neighbors)

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
