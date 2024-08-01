from abc import abstractmethod
import numpy as np
#
from .baseterms import TermABC, TermFactory
from ..forces import get_dist


class LocalFrameTermABC(TermABC):

    def __init__(self, frame_type, atomids, q, dipole, quadrupole, coords, center, a_type):
        self.atomids = np.array(atomids)
        self.center = atomids[center]
        self.frame_type = frame_type
        self.type = a_type
        self._name = f"{self.name}({a_type})"
        self.q = q
        self.equ = None
        self.fconst = None
        self.dipole, self.quadrupole = self.convert_multipoles_to_local_frame(coords, dipole, quadrupole)
        self.dipole_spher, self.quad_spher = self.convert_multipoles_to_local_spherical_frame()

    @classmethod
    def get_term(cls, frame_type, non_bonded, atomids, coords, center, a_type):
        q = non_bonded.q[atomids[0]]
        dipole = non_bonded.dipole[atomids[0]]
        quadrupole = non_bonded.quadrupole[atomids[0]]
        return cls(frame_type, atomids, q, dipole, quadrupole, coords, center, a_type)

    @abstractmethod
    def compute_rotation_matrix(self, coords):
        ...

    def _calc_forces(self, crd, force, fconst):
        return 0.

    @staticmethod
    def get_ex(v2, v_ez):
        val = (v2 - np.dot(v2, v_ez) * v_ez)
        norm = np.linalg.norm(val)
        return val/norm

    @staticmethod
    def get_ey(v_ez, v_ex):
        return np.cross(v_ez, v_ex)

    @staticmethod
    def make_quadrupole_matrix(quad):
        return np.array([[quad[0], quad[3], quad[4]],
                         [quad[3], quad[1], quad[5]],
                         [quad[4], quad[5], quad[2]]])

    def convert_multipoles_to_local_frame(self, coords, dipole, quadrupole):
        rotation_matrix = self.compute_rotation_matrix(coords)
        print('rot_mat\n', rotation_matrix.round(6))

        local_dipole = np.matmul(rotation_matrix, dipole)
        print('dip glob\n', np.round(dipole, 6))
        print('dip, local\n', local_dipole.round(6))
        print()

        matmul = np.matmul(self.make_quadrupole_matrix(quadrupole), rotation_matrix.T)
        local_quadrupole = np.matmul(rotation_matrix, matmul)
        print('quad, glob\n', self.make_quadrupole_matrix(quadrupole).round(6))
        print('quad, local\n', local_quadrupole.round(6))
        print()

        return local_dipole, local_quadrupole

    def convert_multipoles_to_cartesian_frame(self, coords):
        term_coords = coords[self.atomids]
        rotation_matrix = self.compute_rotation_matrix(term_coords)

        cart_dipole = np.matmul(self.dipole, rotation_matrix)
        matmul = np.matmul(self.quadrupole, rotation_matrix)
        cart_quadrupole = np.matmul(rotation_matrix.T, matmul)

        return cart_dipole, cart_quadrupole

    def convert_multipoles_to_local_spherical_frame(self):
        """
        Dipoles as: Q_10, Q_11c, Q_11s
        Quadrupoles as: Q_20, Q_21c, Q_21s, Q_22c, Q_22s
        See GDMA manual for the reference conversions
        """

        dipoles = np.array([self.dipole[2], self.dipole[0], self.dipole[1]])

        q_20 = self.quadrupole[2, 2]
        q_22c = 2 / 3**0.5 * (self.quadrupole[0, 0] + 0.5*q_20)
        q_22s = 2 / 3**0.5 * self.quadrupole[0, 1]
        q_21c = 2 / 3**0.5 * self.quadrupole[0, 2]
        q_21s = 2 / 3**0.5 * self.quadrupole[1, 2]
        quadrupoles = np.array([q_20, q_21c, q_21s, q_22c, q_22s])

        return dipoles, quadrupoles

    def convert_multipoles_to_local_cartesian_frame(self):

        dipoles = np.array([self.dipole_spher[1], self.dipole_spher[2], self.dipole_spher[0]])

        q_11 = -0.5 * self.quad_spher[0] + 3**0.5 / 2 * self.quad_spher[3]
        q_22 = -0.5 * self.quad_spher[0] - 3**0.5 / 2 * self.quad_spher[3]
        q_33 = self.quad_spher[0]
        q_12 = 3**0.5 / 2 * self.quad_spher[4]
        q_13 = 3**0.5 / 2 * self.quad_spher[1]
        q_23 = 3**0.5 / 2 * self.quad_spher[2]
        quadrupoles = np.array([q_11, q_22, q_33, q_12, q_13, q_23])

        return dipoles.round(10), quadrupoles.round(10)


class BisectorTerm(LocalFrameTermABC):
    name = 'BisectorLocalFrameTerm'

    def compute_rotation_matrix(self, coords):

        vec12, r12 = get_dist(coords[1], coords[0])
        vec13, r13 = get_dist(coords[2], coords[0])

        bisect = vec12*r13+vec13*r12
        ez = bisect/np.linalg.norm(bisect)
        ex = self.get_ex(vec13, ez)
        ey = self.get_ey(ez, ex)
        return np.array([ex, ey, ez])


class ZthenXTerm(LocalFrameTermABC):
    name = 'ZthenXTermLocalFrameTerm'

    def compute_rotation_matrix(self, coords):
        vec12, r12 = get_dist(coords[1], coords[0])
        vec13, r13 = get_dist(coords[2], coords[0])

        ez = vec12 / r12
        ex = self.get_ex(vec13, ez)
        ey = self.get_ey(ez, ex)
        return np.array([ex, ey, ez])


class ZThenBisectorTerm(LocalFrameTermABC):
    name = 'ZThenBisectorLocalFrameTerm'

    def compute_rotation_matrix(self, coords):
        vec12, r12 = get_dist(coords[1], coords[0])
        vec3c, r3c = get_dist(coords[2], coords[self.center])
        vec4c, r4c = get_dist(coords[3], coords[self.center])

        ez = vec12 / r12
        bisect = vec3c*r4c+vec4c*r3c
        ex = self.get_ex(bisect, ez)
        ey = self.get_ey(ez, ex)
        return np.array([ex, ey, ez])


class TrisectorTerm(LocalFrameTermABC):
    # This is for ammonia-like molecules only, and probably needs to be implemented with
    # lone pair virtual site for stability reasons

    name = 'TrisectorLocalFrameTerm'

    def compute_rotation_matrix(self, coords):
        vec12, r12 = get_dist(coords[1], coords[0])
        vec13, r13 = get_dist(coords[2], coords[0])
        vec14, r14 = get_dist(coords[3], coords[0])

        trisect = vec12*r13*r14 + vec13*r12*r14 + vec14*r12*r13
        ez = trisect/np.linalg.norm(trisect)
        ex = self.get_ex(vec12, ez)
        ey = self.get_ey(ez, ex)
        return np.array([ex, ey, ez])


class ZOnlyTerm(LocalFrameTermABC):
    name = 'ZOnlyTermLocalFrameTerm'

    def compute_rotation_matrix(self, coords):
        vec12, r12 = get_dist(coords[1], coords[0])

        if len(coords) == 3:
            vec3c, r3c = get_dist(coords[2], coords[self.center])
            dir2 = vec3c/r3c
        else:
            dir2 = np.array([vec12[1], vec12[2], vec12[0]])  # lazy but probably good enough?

        ez = vec12 / r12
        ex = self.get_ex(dir2, ez)
        ey = self.get_ey(ez, ex)

        return np.array([ex, ey, ez])


class LocalFrameTerms(TermFactory):
    name = 'LocalFrameTerms'

    _term_types = {
        'bisector': BisectorTerm,
        'z_then_x': ZthenXTerm,
        'z_only': ZOnlyTerm,
        'z_then_bisector': ZThenBisectorTerm,
        'trisector': TrisectorTerm,
    }

    _always_on = []
    _default_off = ['bisector', 'z_then_x', 'z_only', 'z_then_bisector', 'trisector']

    @classmethod
    def get_terms(cls, topo, non_bonded):
        terms = cls.get_terms_container()

        # helper functions to improve readability
        def add_term(name, atomids, *args):
            terms[name].append(cls._term_types[name].get_term(name, atomids, *args))

        if np.abs(non_bonded.quadrupole).sum() == 0 and np.abs(non_bonded.dipole).sum() == 0:
            return terms

        for i, node in topo.graph.nodes(data=True):
            print('atomid:', i, ', element:', node['elem'])

            if node['n_neighs'] == 1:  # terminal atoms
                neigh = topo.node(node['neighs'][0])

                if topo.n_atoms == 2:
                    print('diatomic, z-only!')
                    atomids = [i, node['neighs'][0]]
                    add_term('z_only', non_bonded, atomids, topo.coords[atomids], 0, node["type"])

                # 1st neighbor has 2-neighbor
                elif neigh['n_neighs'] == 2:
                    if neigh['hybrid'] == 'linear':
                        print('3-atom-linear edge, z-only!')
                        atomids = [i, node['neighs'][0]]
                        add_term('z_only', non_bonded, atomids, topo.coords[atomids], 1, node["type"])

                    elif neigh['hybrid'] == 'bent':
                        print('3-atom-bent, like H in water, z-then-x!')
                        other_neigh = [neigh for neigh in topo.node(node['neighs'][0])['neighs'] if neigh != i][0]
                        atomids = [i, node['neighs'][0], other_neigh]
                        add_term('z_then_x', non_bonded, atomids, topo.coords[atomids], 1, node["type"])

                # 1st neighbor has 3-neighbor
                elif neigh['n_neighs'] == 3:
                    other_neighs = [topo.node(neigh) for neigh in topo.node(node['neighs'][0])['neighs'] if neigh != i]
                    if neigh['hybrid'] == 'planar' and other_neighs[0]['type'] == other_neighs[1]['type']:  # Symmetric planar, like O in COH2
                        print('FOUND: terminal on planar 3-atom center, symmetric others - like O on COH2 or H on benzene - Z-only')
                        atomids = [i, node['neighs'][0], other_neighs[0]['idx']]
                        add_term('z_only', non_bonded, atomids, topo.coords[atomids], 1, node["type"])  # doesn't make sense! need diff XX and YY
                    elif neigh['hybrid'] == 'planar' and other_neighs[0]['type'] != other_neighs[1]['type']:  # Asymmetric planar, like H in COH2
                        print('FOUND: terminal on planar 3-atom center, asymmetric others - like H on COH2, z-then-x!')
                        atomids = [i, node['neighs'][0], other_neighs[0]['idx']]
                        add_term('z_then_x', non_bonded, atomids, topo.coords[atomids], 1, node["type"])

                    elif neigh['hybrid'] == 'pyramidal' and other_neighs[0]['type'] == other_neighs[1]['type']:
                        print('FOUND: Z-then-bisector, like H in Ammonia or H in dimethyl amine')
                        atomids = [i, node['neighs'][0], other_neighs[0]['idx'], other_neighs[1]['idx']]
                        add_term('z_then_bisector', non_bonded, atomids, topo.coords[atomids], 1, node["type"])

                    elif neigh['hybrid'] == 'pyramidal' and other_neighs[0]['type'] != other_neighs[1]['type']:
                        print('FOUND: Z-only, like H in methylamine')
                        atomids = [i, node['neighs'][0], other_neighs[0]['idx']]
                        add_term('z_only', non_bonded, atomids, topo.coords[atomids], 1, node["type"])

                # 1st neighbor has 4-neighbor
                elif neigh['n_neighs'] == 4:
                    if neigh['n_unique_neighs'] == 1:
                        print('FOUND: Terminal atom on a tetrahedral center with all same neighbors like H on methane. Z-only!')
                        other_neigh = [neigh for neigh in topo.node(node['neighs'][0])['neighs'] if neigh != i][0]
                        atomids = [i, node['neighs'][0], other_neigh]
                        add_term('z_only', non_bonded, atomids, topo.coords[atomids], 1, node["type"])
                    else:
                        ...  # add the rest

            elif node['n_neighs'] == 2:  # 2-neighbor atoms
                if node['hybrid'] == 'linear' and node['n_unique_neighs'] == 1:  # This one has no dipole!
                    print('3-atom symmetric linear center, z-only!')
                    atomids = [i, node['neighs'][0]]
                    add_term('z_only', non_bonded, atomids, topo.coords[atomids], 0, node["type"])
                elif node['hybrid'] == 'linear' and node['n_unique_neighs'] == 2:  # Has a dipole in local Z!
                    print('3-atom asymmetric linear center, z-only!')
                    atomids = [i, node['neighs'][0]]
                    add_term('z_only', non_bonded, atomids, topo.coords[atomids], 0, node["type"])
                elif node['hybrid'] == 'bent' and node['n_unique_neighs'] == 1:
                    print('3-atom symmetric center, like water oxygen, bisector!')
                    atomids = [i, node['neighs'][0], node['neighs'][1]]
                    add_term('bisector', non_bonded, atomids, topo.coords[atomids], 0, node["type"])
                elif node['hybrid'] == 'bent' and node['n_unique_neighs'] == 2:
                    print('3-atom asymmetric center, z-then-x!')
                    atomids = [i, node['neighs'][0], node['neighs'][1]]
                    add_term('z_then_x', non_bonded, atomids, topo.coords[atomids], 0, node["type"])

            elif node['n_neighs'] == 3:  # 3-neighbor atoms
                # Pyramidal
                if node['hybrid'] == 'pyramidal' and node['n_unique_neighs'] == 1:
                    atomids = [i, node['neighs'][0], node['neighs'][1], node['neighs'][2]]
                    add_term('trisector', non_bonded, atomids, topo.coords[atomids], 0, node["type"])
                    print('FOUND: ammonia N-like, trisector!')
                elif node['hybrid'] == 'pyramidal' and node['n_unique_neighs'] == 2:
                    print('FOUND: pyramidal, with 2 unique neighbors, like N on methylamine, z-then-bisector!')
                    lone = [unique for unique in node['unique_neighs'] if len(unique) == 1][0]
                    equal_pair = [unique for unique in node['unique_neighs'] if len(unique) == 2][0]
                    atomids = [i, lone, equal_pair[0], equal_pair[1]]
                    add_term('z_then_bisector', non_bonded, atomids, topo.coords[atomids], 0, node["type"])
                elif node['hybrid'] == 'pyramidal' and node['n_unique_neighs'] == 3:
                    print('FOUND: pyramidal, all neighbors different, z-then-x??')
                    atomids = [i, node['neighs'][0], node['neighs'][1]]
                    add_term('z_then_x', non_bonded, atomids, topo.coords[atomids], 0, node["type"])
                # Planar
                elif node['hybrid'] == 'planar' and node['n_unique_neighs'] == 1:
                    print('FOUND: planar 3-atom center, with all same neighbors, like N on NO3 , z-only??')
                elif node['hybrid'] == 'planar' and node['n_unique_neighs'] == 2:
                    print('FOUND: planar 3-atom center, with 2 unique neighbors, like C on COH2 or C in benzene, bisector!')
                    equal_pair = [unique for unique in node['unique_neighs'] if len(unique) == 2][0]
                    atomids = [i, equal_pair[0], equal_pair[1]]
                    add_term('bisector', non_bonded, atomids, topo.coords[atomids], 0, node["type"])
                elif node['hybrid'] == 'planar' and node['n_unique_neighs'] == 3:
                    print('FOUND: planar 3-atom center, with all different neighbors, like C on COHF , z-only??')

            elif node['n_neighs'] == 4:  # 4-neighbor atoms
                if node['n_unique_neighs'] == 2 and node['n_nonrepeat_neighs'] == 0:
                    print('FOUND: propane carbon alike, bisector!')
                else:
                    atomids = [i, node['neighs'][0], node['neighs'][1]]
                    print('FOUND: tetrahedral atom, like C on methane, z-only (no dip/quad)')
                    add_term('z_only', non_bonded, atomids, topo.coords[atomids], 0, node["type"])

        terms = LocalFrameTerms.average_equivalent_terms(terms)

        return terms

    @staticmethod
    def average_equivalent_terms(terms):
        term_dict = {}
        for term in terms:
            if term.type in term_dict:
                term_dict[term.type][0].append(term.dipole_spher)
                term_dict[term.type][1].append(term.quad_spher)
            else:
                term_dict[term.type] = [[term.dipole_spher], [term.quad_spher]]

        for key, val in term_dict.items():
            term_dict[key][0] = np.mean(val[0], axis=0)
            term_dict[key][1] = np.mean(val[1], axis=0)

        for term in terms:
            term.dipole_spher = term_dict[term.type][0].round(10)
            term.quad_spher = term_dict[term.type][1].round(10)

            term.dipole, term.quadrupole = term.convert_multipoles_to_local_cartesian_frame()
            term.quadrupole = term.make_quadrupole_matrix(term.quadrupole)
        return terms
