import numpy as np
from itertools import product
#
from .baseterms import TermBase
from ..forces import get_dist, get_angle
from ..forces import (calc_cross_bond_bond, calc_cross_bond_angle, calc_cross_bond_cos_angle, calc_cross_angle_angle,
                      calc_cross_cos_angle_angle, calc_cross_dihed_angle, calc_cross_dihed_bond,
                      calc_cross_cos_cube_dihed_angle, calc_cross_cos_cube_dihed_bond,
                      calc_cross_dihed_angle_angle, calc_cross_cos_cube_dihed_angle_angle)


class CrossBondBondTerm(TermBase):

    name = 'CrossBondBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_bond_bond(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l = self.atomids
        return [self.bondname(i, j), self.bondname(k, l)]

    def update_constants(self, dct):
        """update constants for the class"""
        b1, b2 = self.constants()
        b1 = dct.get(b1, None)
        if b1 is not None:
            self.equ[0] = b1
        b2 = dct.get(b2, None)
        if b2 is not None:
            self.equ[1] = b2

    @classmethod
    def _get_terms(cls, topo, non_bonded):

        cross_bond_bond_terms = cls.get_terms_container()

        z = 0
        for a1, a2 in topo.bonds:
            for a3, a4 in topo.bonds:

                if a1 >= a3 and a2 >= a4:
                    continue

                dist12 = get_dist(topo.coords[a1], topo.coords[a2])[1]
                dist34 = get_dist(topo.coords[a3], topo.coords[a4])[1]

                dists = np.array([dist12, dist34])

                n_shared = sum([a1 in [a3, a4], a2 in [a3, a4]])

                if n_shared == 0:
                    continue

                b21 = topo.edge(a2, a1)['vers']
                b_type1 = sorted([topo.types[a1], topo.types[a2]])
                b_type1 = f"{b_type1[0]}({b21}){b_type1[1]}"

                b43 = topo.edge(a4, a3)['vers']
                b_type2 = sorted([topo.types[a3], topo.types[a4]])
                b_type2 = f"{b_type2[0]}({b43}){b_type2[1]}"

                bb_type = sorted([b_type1, b_type2])
                bb_type = f"{bb_type[0]}-{bb_type[1]}-{n_shared}-{z}"
                cross_bond_bond_terms.append(cls([a1, a2, a3, a4], dists, bb_type))
                # z += 1

        return cross_bond_bond_terms

    def write_forcefield(self, software, writer):
        software.write_cross_bond_bond_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_bond_bond_header(writer)


class CrossBondAngleTerm(TermBase):
    name = 'CrossBondAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_bond_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m = self.atomids
        return [self.anglename(i, j, k), self.bondname(l, m)]

    def update_constants(self, dct):
        """update constants for the class"""
        a1, b1 = self.constants()
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1
        b1 = dct.get(b1, None)
        if b1 is not None:
            self.equ[1] = b1

    @classmethod
    def _get_terms(cls, topo, non_bonded):

        cross_bond_angle_terms = cls.get_terms_container()

        for a1, a2, a3 in topo.angles:
            theta = get_angle(topo.coords[[a1, a2, a3]])[0]
            #  No CrossBondAngle term  if linear angle (>170) or if in 3-member ring
            # if theta < 2.9671 and (not topo.edge(a2, a1)['in_ring3'] or
            #                        not topo.edge(a2, a3)['in_ring3']):

            for a4, a5 in topo.bonds:

                dist = get_dist(topo.coords[a4], topo.coords[a5])[1]

                equ = np.array([theta, dist])

                b21 = topo.edge(a2, a1)['vers']
                b23 = topo.edge(a2, a3)['vers']
                b45 = topo.edge(a4, a5)['vers']

                if sum([a4 in [a1, a2, a3], a5 in [a1, a2, a3]]) < 1:
                    continue

                n_shared = sum([a4 in [a1, a2, a3], a5 in [a1, a2, a3]])
                is_angle_centered = a2 in [a4, a5]

                # For formaldehyde, methane, ethene
                if n_shared == 1 and not topo.ba_couple_1_shared:
                    continue

                a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                                 f"{topo.types[a2]}({b23}){topo.types[a3]}"])
                a_type = f"{a_type[0]}_{a_type[1]}"
                b_type = sorted([topo.types[a4], topo.types[a5]])
                b_type = f"{b_type[0]}({b45}){b_type[1]}"

                ba_type = f'{a_type}-{b_type}-{n_shared}-{is_angle_centered}'

                cross_bond_angle_terms.append(cls([a1, a2, a3, a4, a5], equ, ba_type))

        return cross_bond_angle_terms

    def write_forcefield(self, software, writer):
        software.write_cross_bond_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_bond_angle_header(writer)

class CrossBondCosineAngleTerm(CrossBondAngleTerm):
    name = 'CrossBondCosineAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_bond_cos_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m = self.atomids
        return [self.anglename(i, j, k), self.bondname(l, m)]

    def update_constants(self, dct):
        """update constants for the class"""
        a1, b1 = self.constants()
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1
        b1 = dct.get(b1, None)
        if b1 is not None:
            self.equ[1] = b1

    def write_forcefield(self, software, writer):
        software.write_cross_bond_cos_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_bond_cos_angle_header(writer)


class CrossAngleAngleTerm(TermBase):
    name = 'CrossAngleAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_angle_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m, n = self.atomids
        return [self.anglename(i, j, k), self.anglename(l, m, n)]

    def update_constants(self, dct):
        """update constants for the class"""
        a1, a2 = self.constants()
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1
        a2 = dct.get(a2, None)
        if a2 is not None:
            self.equ[1] = a2

    @classmethod
    def _get_terms(cls, topo, non_bonded):

        cross_angle_angle_terms = cls.get_terms_container()

        for i, (a1, a2, a3) in enumerate(topo.angles):
            theta1 = get_angle(topo.coords[[a1, a2, a3]])[0]
            #  No CrossBondAngle term  if linear angle (>170) or if in 3-member ring
            # if theta < 2.9671 and (not topo.edge(a2, a1)['in_ring3'] or
            #                        not topo.edge(a2, a3)['in_ring3']):

            for j, (a4, a5, a6) in enumerate(topo.angles):

                # if a4 != a2 and a6 != a2:
                #     continue

                # if a4 != a2 and a6 != a2:
                #     continue

                if i <= j:
                    continue

                # if i >= j:
                #     continue

                # if a2 == a5:
                #     continue

                # # if a1 not in [a4, a5, a6] and a2 not in [a4, a5, a6] and a3 not in [a4, a5, a6]:
                # #     continue

                # if sum([a1 == a5, a3 == a5, a4 == a2, a6 == a2]) != 2:
                #     continue

                n_shared = sum([a1 in [a4, a5, a6], a2 in [a4, a5, a6], a3 in [a4, a5, a6]])
                n_matched = sum([a1 == a4 or a1 == a6, a2 == a5, a3 == a6 or a3 == a4])
                # For methane
                # if n_shared == 0:
                #     continue

                # For ethene
                if n_shared < 2:
                    continue
                # if n_matched == 0:
                #     continue

                theta2 = get_angle(topo.coords[[a4, a5, a6]])[0]
                equ = np.array([theta1, theta2])

                b1_21 = topo.edge(a2, a1)['vers']
                b1_23 = topo.edge(a2, a3)['vers']
                a1_type = sorted([f"{topo.types[a2]}({b1_21}){topo.types[a1]}",
                                  f"{topo.types[a2]}({b1_23}){topo.types[a3]}"])
                a1_type = f"{a1_type[0]}_{a1_type[1]}"

                b2_21 = topo.edge(a5, a4)['vers']
                b2_23 = topo.edge(a5, a6)['vers']
                a2_type = sorted([f"{topo.types[a5]}({b2_21}){topo.types[a4]}",
                                  f"{topo.types[a5]}({b2_23}){topo.types[a6]}"])
                a2_type = f"{a2_type[0]}_{a2_type[1]}"

                aa_type = sorted([a1_type, a2_type])
                aa_type = f"{aa_type[0]}-{aa_type[1]}-{n_shared}-{n_matched}"

                cross_angle_angle_terms.append(cls([a1, a2, a3, a4, a5, a6], equ, aa_type))

        return cross_angle_angle_terms

    def write_forcefield(self, software, writer):
        software.write_cross_angle_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_angle_angle_header(writer)


class CrossCosineAngleAngleTerm(CrossAngleAngleTerm):
    name = 'CrossCosineAngleAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_cos_angle_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m, n = self.atomids
        return [self.anglename(i, j, k), self.anglename(l, m, n)]

    def update_constants(self, dct):
        """update constants for the class"""
        a1, a2  = self.constants()
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1
        a2 = dct.get(a1, None)
        if a2 is not None:
            self.equ[1] = a2

    def write_forcefield(self, software, writer):
        software.write_cross_cos_angle_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_cos_angle_angle_header(writer)


class CrossDihedBondTerm(TermBase):
    name = 'CrossDihedBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_dihed_bond(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m, n = self.atomids
        return [self.torsionname(i, j, k, l), self.bondname(m, n)]

    def update_constants(self, dct):
        """update constants for the class"""
        # do not know if the torsion is updated or not???
        # currently its not
        t1, b1  = self.constants()
        # update only bond term
        b1 = dct.get(a1, None)
        if b1 is not None:
            self.equ[0] = b1

    @classmethod
    def _get_terms(cls, topo, non_bonded):

        cross_dihed_bond_terms = cls.get_terms_container()

        for a2, a3 in topo.bonds:
            central = topo.edge(a2, a3)
            a1s = [a1 for a1 in topo.neighbors[0][a2] if a1 != a3]
            a4s = [a4 for a4 in topo.neighbors[0][a3] if a4 != a2]

            if a1s == [] or a4s == []:
                continue

            a1s, a4s = np.array(a1s), np.array(a4s)
            atoms_comb = [list(d) for d in product(a1s, [a2], [a3],
                          a4s) if d[0] != d[-1]]

            # rigid
            for atoms in atoms_comb:
                a1, a2, a3, a4 = atoms
                b12 = topo.edge(a1, a2)["vers"]
                b23 = topo.edge(a2, a3)["vers"]
                b43 = topo.edge(a4, a3)["vers"]
                t23 = [topo.types[a2], topo.types[a3]]
                t12 = f"{topo.types[a1]}({b12}){topo.types[a2]}"
                t43 = f"{topo.types[a4]}({b43}){topo.types[a3]}"
                d_type = [t12, t43]

                if t12 > t43:
                    d_type.reverse()
                    t23.reverse()

                d_type = f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"

                for j, (a5, a6) in enumerate(topo.bonds):
                    dist = get_dist(topo.coords[a5], topo.coords[a6])[1]

                    b_type = sorted([topo.types[a5], topo.types[a6]])
                    b56 = topo.edge(a5, a6)['vers']
                    b_type = f"{b_type[0]}({b56}){b_type[1]}"

                    n_shared = sum([a5 in atoms, a6 in atoms])

                    connect = ''
                    if n_shared < 2:
                        continue
                    # elif n_shared == 1 and topo.types[a1] != topo.types[a4]:
                    #     if a5 in atoms:
                    #         connection = atoms.index(a5)
                    #     elif a6 in atoms:
                    #         connection = atoms.index(a6)
                    #
                    #     connect = f'-asym_{connection}'

                    db_type = f'{d_type}-{b_type}-{n_shared}{connect}'

                    cross_dihed_bond_terms.append(cls([a1, a2, a3, a4, a5, a6], [dist, 3, 0], f'{db_type}-3'))
                    cross_dihed_bond_terms.append(cls([a1, a2, a3, a4, a5, a6], [dist, 2, np.pi], f'{db_type}-2'))
                    cross_dihed_bond_terms.append(cls([a1, a2, a3, a4, a5, a6], [dist, 1, 0], f'{db_type}-1'))
        return cross_dihed_bond_terms

    def write_forcefield(self, software, writer):
        software.write_cross_dihed_bond_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_dihed_bond_header(writer)


class CrossCosCubeDihedBondTerm(CrossDihedBondTerm):
    name = 'CrossCosCubeDihedBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_cos_cube_dihed_bond(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m, n = self.atomids
        return [self.torsionname(i, j, k, l), self.bondname(m, n)]

    def update_constants(self, dct):
        """update constants for the class"""
        t1, b1  = self.constants()
        b1 = dct.get(b1, None)
        if b1 is not None:
            self.equ[0] = b1
        # torsion is also in this case not updated!

    def write_forcefield(self, software, writer):
        software.write_cross_cos_cube_dihed_bond_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_cos_cube_dihed_bond_header(writer)


class CrossDihedAngleTerm(TermBase):
    name = 'CrossDihedAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_dihed_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m, n, o = self.atomids
        return [self.torsionname(i, j, k, l), self.anglename(m, n, o)]

    def update_constants(self, dct):
        """update constants for the class"""
        t1, a1  = self.constants()
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1
        # torsion is also in this case not updated!

    @classmethod
    def _get_terms(cls, topo, non_bonded):

        cross_dihed_angle_terms = cls.get_terms_container()

        for a2, a3 in topo.bonds:
            central = topo.edge(a2, a3)
            a1s = [a1 for a1 in topo.neighbors[0][a2] if a1 != a3]
            a4s = [a4 for a4 in topo.neighbors[0][a3] if a4 != a2]

            if a1s == [] or a4s == []:
                continue

            a1s, a4s = np.array(a1s), np.array(a4s)
            atoms_comb = [list(d) for d in product(a1s, [a2], [a3],
                          a4s) if d[0] != d[-1]]

            for atoms in atoms_comb:
                a1, a2, a3, a4 = atoms
                b12 = topo.edge(a1, a2)["vers"]
                b23 = topo.edge(a2, a3)["vers"]
                b43 = topo.edge(a4, a3)["vers"]
                t23 = [topo.types[a2], topo.types[a3]]
                t12 = f"{topo.types[a1]}({b12}){topo.types[a2]}"
                t43 = f"{topo.types[a4]}({b43}){topo.types[a3]}"
                d_type = [t12, t43]

                d_type = f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"

                for a5, a6, a7 in topo.angles:
                    b21 = topo.edge(a6, a5)['vers']
                    b23 = topo.edge(a6, a7)['vers']
                    a_type = sorted([f"{topo.types[a6]}({b21}){topo.types[a5]}",
                                     f"{topo.types[a6]}({b23}){topo.types[a7]}"])
                    a_type = f"{a_type[0]}_{a_type[1]}"

                    theta = get_angle(topo.coords[[a5, a6, a7]])[0]
                    n_shared = sum([a5 in atoms, a6 in atoms, a7 in atoms])

                    connect = ''
                    if n_shared < 3:
                        continue
                    # elif n_shared == 2 and topo.types[a1] != topo.types[a4]:
                    #     if a6 in atoms:
                    #         connect += f'-center:{atoms.index(a6)}'

                    da_type = f'{d_type}-{a_type}-{n_shared}-{connect}'

                    cross_dihed_angle_terms.append(cls([a1, a2, a3, a4, a5, a6, a7], [theta, 3, 0], f'{da_type}-3'))
                    cross_dihed_angle_terms.append(cls([a1, a2, a3, a4, a5, a6, a7], [theta, 2, np.pi], f'{da_type}-2'))
                    cross_dihed_angle_terms.append(cls([a1, a2, a3, a4, a5, a6, a7], [theta, 1, 0], f'{da_type}-1'))

        return cross_dihed_angle_terms

    def write_forcefield(self, software, writer):
        software.write_cross_dihed_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_dihed_angle_header(writer)


class CrossCosCubeDihedAngleTerm(CrossDihedAngleTerm):
    name = 'CrossCosCubeDihedAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_cos_cube_dihed_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l, m, n, o = self.atomids
        return [self.torsionname(i, j, k, l), self.anglename(m, n, o)]

    def update_constants(self, dct):
        """update constants for the class"""
        t1, a1  = self.constants()
        # torsion is also in this case not updated!
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1

    def write_forcefield(self, software, writer):
        software.write_cross_cos_cube_dihed_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_cos_cube_dihed_angle_header(writer)


class CrossDihedAngleAngleTerm(TermBase):
    name = 'CrossDihedAngleAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_dihed_angle_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l = self.atomids
        return [self.anglename(i, j, k), self.anglename(j, k, l)]

    def update_constants(self, dct):
        """update constants for the class"""
        a1, a2  = self.constants()
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1
        a2 = dct.get(a2, None)
        if a2 is not None:
            self.equ[1] = a2

    @classmethod
    def _get_terms(cls, topo, non_bonded):

        cross_dihed_angle_angle_terms = cls.get_terms_container()

        for a2, a3 in topo.bonds:
            central = topo.edge(a2, a3)
            a1s = [a1 for a1 in topo.neighbors[0][a2] if a1 != a3]
            a4s = [a4 for a4 in topo.neighbors[0][a3] if a4 != a2]

            if a1s == [] or a4s == []:
                continue

            a1s, a4s = np.array(a1s), np.array(a4s)
            atoms_comb = [list(d) for d in product(a1s, [a2], [a3],
                          a4s) if d[0] != d[-1]]

            for atoms in atoms_comb:
                a1, a2, a3, a4 = atoms
                b12 = topo.edge(a1, a2)["vers"]
                b23 = topo.edge(a2, a3)["vers"]
                b43 = topo.edge(a4, a3)["vers"]
                t23 = [topo.types[a2], topo.types[a3]]
                t12 = f"{topo.types[a1]}({b12}){topo.types[a2]}"
                t43 = f"{topo.types[a4]}({b43}){topo.types[a3]}"
                d_type = [t12, t43]

                d_type = f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"
                theta1 = get_angle(topo.coords[[a1, a2, a3]])[0]
                theta2 = get_angle(topo.coords[[a2, a3, a4]])[0]

                cross_dihed_angle_angle_terms.append(cls([a1, a2, a3, a4], [theta1, theta2, 1, 0], d_type))

        return cross_dihed_angle_angle_terms

    def write_forcefield(self, software, writer):
        software.write_cross_dihed_angle_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_dihed_angle_angle_header(writer)


class CrossCosCubeDihedAngleAngleTerm(CrossDihedAngleAngleTerm):
    name = 'CrossCosCubeDihedAngleAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_cos_cube_dihed_angle_angle(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        i, j, k, l = self.atomids
        return [self.anglename(i, j, k), self.anglename(j, k, l)]

    def update_constants(self, dct):
        """update constants for the class"""
        a1, a2  = self.constants()
        a1 = dct.get(a1, None)
        if a1 is not None:
            self.equ[0] = a1
        a2 = dct.get(a2, None)
        if a2 is not None:
            self.equ[1] = a2

    def write_forcefield(self, software, writer):
        software.write_cross_cos_cube_dihed_angle_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cross_cos_cube_dihed_angle_angle_header(writer)


# class CrossDihedDihedTerm(TermBase):
#     name = 'CrossDihedDihedTerm'
#
#     def _calc_forces(self, crd, force, fconst):
#         return calc_cross_dihed_dihed(crd, self.atomids, self.equ, fconst, force)
#
#     @classmethod
#     def _get_terms(cls, topo, non_bonded):
#
#         cross_dihed_dihed_terms = cls.get_terms_container()
#
#         z = 0
#
#         for a2, a3 in topo.bonds:
#             central = topo.edge(a2, a3)
#             a1s = [a1 for a1 in topo.neighbors[0][a2] if a1 != a3]
#             a4s = [a4 for a4 in topo.neighbors[0][a3] if a4 != a2]
#
#             if a1s == [] or a4s == []:
#                 continue
#
#             a1s, a4s = np.array(a1s), np.array(a4s)
#             atoms_comb = [list(d) for d in product(a1s, [a2], [a3],
#                           a4s) if d[0] != d[-1]]
#
#             for atoms in atoms_comb:
#                 a1, a2, a3, a4 = atoms
#                 b12 = topo.edge(a1, a2)["vers"]
#                 b23 = topo.edge(a2, a3)["vers"]
#                 b43 = topo.edge(a4, a3)["vers"]
#                 t23 = [topo.types[a2], topo.types[a3]]
#                 t12 = f"{topo.types[a1]}({b12}){topo.types[a2]}"
#                 t43 = f"{topo.types[a4]}({b43}){topo.types[a3]}"
#                 d_type = [t12, t43]
#
#                 d_type = f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"
#
#                 for a5, a6, a7 in topo.angles:
#                     b21 = topo.edge(a6, a5)['vers']
#                     b23 = topo.edge(a6, a7)['vers']
#                     a_type = sorted([f"{topo.types[a6]}({b21}){topo.types[a5]}",
#                                      f"{topo.types[a6]}({b23}){topo.types[a7]}"])
#                     a_type = f"{a_type[0]}_{a_type[1]}"
#
#                     theta = get_angle(topo.coords[[a5, a6, a7]])[0]
#                     n_shared = sum([a5 in atoms, a6 in atoms, a7 in atoms])
#
#                     connect = ''
#                     if n_shared < 3:
#                         continue
#                     # elif n_shared == 2 and topo.types[a1] != topo.types[a4]:
#                     #     if a6 in atoms:
#                     #         connect += f'-center:{atoms.index(a6)}'
#
#                     da_type = f'{d_type}-{a_type}-{n_shared}-{connect}'
#
#                     cross_dihed_dihed_terms.append(cls([a1, a2, a3, a4, a5, a6, a7, a8], [3, 0, 3, 0], da_type))
#                     # cross_dihed_dihed_terms.append(cls([a1, a2, a3, a4, a5, a6, a7, a8], [1, 0, 1, 0], da_type))
#
#         return cross_dihed_dihed_terms
#
#     def write_forcefield(self, software, writer):
#         software.write_cross_dihed_dihed_term(self, writer)
#
#     def write_ff_header(self, software, writer):
#         return software.write_cross_dihed_dihed_header(writer)
