import numpy as np
import networkx as nx
from .elements import elements
from .forces import get_dist, get_angle, get_dihed

class Terms():
    
    def __init__(self,):
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

class Atoms():
    pass ## atoms.types , atoms.ids, atoms, elements, etc

class Angles(Terms):
    urey = Terms()

class Dihedrals(Terms):
    stiff = Terms()
    improper = Terms()
    flexible = Terms()

class Molecule():
    """
    Scope:
    ------
    self.list : atom numbers of unique atoms grouped together
    self.atoms : unique atom numbers of each atom
    self.types : atom types of each atom
    self.connect : First 3 neighbors of each atom    
    
    
    To do:
    -----
    - 180 degree angle = NO DIHEDRAL!
    - In future: Make a different type for in-ring with shorter scan
    - Flexible dihedrals should be treated differently:
        For type only A2 and A3 matters. Relevant when scanning is added.


    """
    def __init__(self, coords, atomids, inp):
        self.n_atoms = len(atomids)
        self.no = [[[i] for i in range(self.n_atoms)]]
        self.atomids = atomids
        self.coords = coords
        self.id = [atomids]
        self.atoms = np.zeros(self.n_atoms, dtype='int8')
        self.list = []
        self.conj = []
        self.double = []
        self.n_types = 0
        self.n_terms = 0
        self.neighbors = [[[] for j in range(self.n_atoms)] for i in range(3)]
        self.thole = []
        self.polar = []
        
        self.bonds = Terms()
        self.angles = Angles()
        self.dihedrals = Dihedrals() 

        self.find_bonds_and_rings()
        self.find_atom_types(inp.n_equiv)
        self.find_neighbors()
        self.find_parameter_types(inp.urey)
        self.prepare()
        self.polarize()


    def find_bonds_and_rings(self):
        bond_id, bond = [], []
        e = elements()
        self.graph = nx.Graph()
        for i, i_id in enumerate(self.id[0]):
            b, b_id = [], []
            self.graph.add_node(i, element=i_id)
            for j, j_id in enumerate(self.id[0]):
                order = 's'
                vector = self.coords[i] - self.coords[j]
                dist = np.sqrt((vector**2).sum())
                if dist > 0.4 and dist < e.cov[i_id] + e.cov[j_id] + 0.45:
                    b.append(j)
                    b_id.append(j_id)
                    cov_len = e.cov[self.id[0][i]]+e.cov[self.id[0][j]]
                    sorted_a = sorted([i,j])
                    if dist < cov_len - 0.04 and sorted_a not in self.conj:
                        self.conj.append(sorted_a)
                    if dist < cov_len - 0.15 and sorted_a not in self.double:
                        self.double.append(sorted_a)
                        

                    if dist < cov_len - 0.15:
                        order = 'd'
                    elif dist < cov_len - 0.04:
                        order = 'c'
                    ids = sorted([i_id, j_id])
                    self.graph.add_edge(i,j, b_type=f"{ids[0]}{order}{ids[1]}",
                                        b_vector = vector, b_length = dist)
            bond.append(sorted(b))
            bond_id.append(sorted(b_id))
            if len(bond[-1]) > e.maxb[i_id]:
                print("WARNING: Atom {} ({}) has too many ({}) neighbors?"
                      .format(i+1, e.sym[i_id], len(bond[-1])))
        self.no.append(bond)
        self.id.append(bond_id)

        self.rings = nx.minimum_cycle_basis(self.graph)
        self.rings3 = [r for r in self.rings if len(r) == 3]
        
    def find_atom_types(self, n_eq):

        atom_ids = [[] for n in range(self.n_atoms)]

        for i in range(self.n_atoms):
            neighbors = nx.bfs_tree(self.graph, i, depth_limit=n_eq).nodes
            for n in neighbors:
                paths = nx.all_simple_paths(self.graph, i, n, cutoff=n_eq)
                for path in map(nx.utils.pairwise, paths):
                    types = [self.graph.edges[edge]['b_type'] for edge in path]
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

        e = elements()
        types = {i : 1 for i in set(self.atomids)}
        self.types = [None] * len(self.atomids)
        for eq in self.list:
            for i in eq:
                self.types[i] = "{}{}".format(e.sym[self.atomids[i]],
                                              types[self.atomids[i]])
            types[self.atomids[eq[0]]] += 1
    
    def find_parameter_types(self, urey):
        bonds, angles, dihedrals = [], [], []
        bond_dict = {}
        
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
            dist = self.graph.edges[a1, a2]['b_length']

            b_type = tuple(sorted([self.types[a1], self.types[a2]]))
            if b_type not in bond_dict or abs(dist - bond_dict[b_type]) < 0.03:
                bond_dict[b_type] = dist
                t = 1
            else:
                t = 2
            self.graph.edges[a1, a2]['type'] = t
            self.bonds.add_term([a1, a2], dist, f"{b_type[0]}({t}){b_type[1]}")

        for a1, a2, a3 in angles:
            vec12, _ = get_dist(self.coords[a1], self.coords[a2])
            vec32, _ = get_dist(self.coords[a3], self.coords[a2])
            theta = get_angle(vec12, vec32)
            
            b21 = self.graph.edges[a2, a1]['type']
            b23 = self.graph.edges[a2, a3]['type']
            a_type = sorted([f"{self.types[a2]}({b21}){self.types[a1]}",
                             f"{self.types[a2]}({b23}){self.types[a3]}"])
            a_type = f"{a_type[0]}_{a_type[1]}"      
            self.angles.add_term([a1, a2, a3], theta, a_type)
            
            if urey:
                dist = get_dist(self.coords[a1], self.coords[a3])[1]
                self.angles.urey.add_term([a1, a3], dist, a_type)

        n_ring = [sum(a in r for r in self.rings) for a in range(self.n_atoms)]
        
        for a1, a2, a3, a4 in dihedrals:
            phi = get_dihed(self.coords[a1], self.coords[a2], 
                            self.coords[a3], self.coords[a4])[0]
             
            
            b12 = self.graph.edges[a1, a2]["type"]
            b23 = self.graph.edges[a2, a3]["type"]
            b43 = self.graph.edges[a4, a3]["type"]
            t23 = [self.types[a2], self.types[a3]]
            t12 = f"{self.types[a1]}({b12}){self.types[a2]}"            
            t43 = f"{self.types[a4]}({b43}){self.types[a3]}"    
            d_type = [t12, t43]
        
            if  t12 > t43:
                d_type.reverse()
                t23.reverse()
            
            d_type = f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"
            
            in_ring = any(set([a2, a3]).issubset(set(r)) for r in self.rings)
            in_ring3 = any(set([a2, a3]).issubset(set(r)) for r in self.rings3)
            multi_ring = sum([n_ring[a] > 2 for a in [a1,a2,a3,a4]]) > 2
            
            if in_ring3:
                continue
            elif [a2, a3] in self.double or ([a2, a3] in self.conj 
                                             and in_ring) or multi_ring:
                self.dihedrals.stiff.add_term([a1, a2, a3, a4], phi, d_type)   
#            elif in_ring: 
#                self.dihedrals.flexible.add_term([a1, a2, a3, a4], phi, d_type)
            else:
                a23 = [[a[1], a[2]] for a in self.dihedrals.flexible.atoms]
                if [a2, a3] not in a23:
                    self.dihedrals.flexible.add_term([a1, a2, a3, a4], phi, d_type)

        # find improper dihedrals
        for i in range(self.n_atoms):
            bonds = list(self.graph.neighbors(i))
            if len(bonds) != 3:
                continue
            atoms = [i,-1,-1,-1]
            n_bond = [len(list(self.graph.neighbors(b))) for b in bonds]
            atoms[3] = bonds[n_bond.index(min(n_bond))]
            for b in bonds:
                if b not in atoms:
                    atoms[atoms.index(-1)] = b

            phi = get_dihed(self.coords[atoms[0]], self.coords[atoms[1]],
                            self.coords[atoms[2]], self.coords[atoms[3]])[0]
            if abs(phi) > 0.035: #check planarity <2 degrees
                continue
            
            bonds = [sorted([b,i]) for b in bonds]
            
            # Only add improper dihedrals if there is no stiff dihedral
            # on the central improper atom and one of the neighbors
            if not any(b == a[1:3] for a in self.dihedrals.stiff.atoms 
                       for b in bonds):
                imp_type = f"ki_{self.types[i]}"
                self.dihedrals.improper.add_term(atoms, phi, imp_type)

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
        for term in [self.bonds, self.angles.urey, self.angles, 
                     self.dihedrals.stiff, self.dihedrals.improper]:
            term.term_ids = [i + self.n_terms for i in term.term_ids]
            self.n_terms += term.n_terms
                 
    def polarize(self):
        """
        TEST for relax_drude.
        """
        a_const = 2.6
        polar_dict = {1: 0.45330, 6: 1.30300, 7:0.98840, 8: 0.83690, 16: 2.47400}
#        polar_dict = { 1: 0.41383,  6: 1.45000,  7: 0.971573,
#                       8: 0.851973,  9: 0.444747, 16: 2.474448,
#                      17: 2.400281, 35: 3.492921, 53: 5.481056}  # 1: 0.413835

        for i in range(self.n_atoms):
            # add polarizability
            self.polar.append(polar_dict[self.atomids[i]])
            # add thole
            for a in self.neighbors[0][i]+self.neighbors[1][i]: #+self.neighbors[2][i]
                if i < a:
                    a1, a2 = polar_dict[self.atomids[i]], polar_dict[self.atomids[a]]
                    self.thole.append([i, a, a_const / (a1*a2)**(1./6.)])
