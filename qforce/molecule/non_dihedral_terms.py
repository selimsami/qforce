import numpy as np
#
from .baseterms import TermBase
#
from ..forces import get_dist, get_angle
from ..forces import (calc_bonds, calc_morse, calc_morse_mp, calc_morse_mp2,
                      calc_angles, calc_cross_bond_bond, calc_cross_bond_angle)
#
from .energies import bd_energy


class BondTerm(TermBase):
    name = 'BondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_bonds(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            b_order_half_rounded = np.round(bond['order']*2)/2
            type1, type2 = sorted([topo.types[a1], topo.types[a2]])
            bond['vers'] = f"{type1}({b_order_half_rounded}){type2}"
            # print(bond['vers'])
            bond_terms.append(cls([a1, a2], dist, bond['vers'], 1))

        return bond_terms


class MorseTerm(TermBase):
    name = 'MorseTerm'

    def _calc_forces(self, crd, force, fconst):
        # print(self.typename)
        info = self.typename.split('(')
        info = info[0:1] + info[1].split(')')
        info = (info[0].rstrip('0123456789'), int(float(info[1])), info[2].rstrip('0123456789'))
        return calc_morse(crd, self.atomids, self.equ, fconst, force, bd_energy[info])

    @classmethod
    def get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            b_order_half_rounded = np.round(bond['order'] * 2) / 2
            type1, type2 = sorted([topo.types[a1], topo.types[a2]])
            bond['vers'] = f"{type1}({b_order_half_rounded}){type2}"
            bond_terms.append(cls([a1, a2], dist, bond['vers'], 1))

        return bond_terms


class MorseMPTerm(TermBase):
    name = 'MorseMPTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_morse_mp(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            b_order_half_rounded = np.round(bond['order'] * 2) / 2
            type1, type2 = sorted([topo.types[a1], topo.types[a2]])
            bond['vers'] = f"{type1}({b_order_half_rounded}){type2}"
            bond_terms.append(cls([a1, a2], dist, bond['vers'], 2))

        return bond_terms


class MorseMP2Term(TermBase):
    name = 'MorseMP2Term'

    def _calc_forces(self, crd, force, fconst):
        return calc_morse_mp2(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            b_order_half_rounded = np.round(bond['order'] * 2) / 2
            type1, type2 = sorted([topo.types[a1], topo.types[a2]])
            bond['vers'] = f"{type1}({b_order_half_rounded}){type2}"
            bond_terms.append(cls([a1, a2], dist, bond['vers'], 2))

        return bond_terms


class AngleTerm(TermBase):
    name = 'AngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_angles(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):
        angle_terms = cls.get_terms_container()

        for a1, a2, a3 in topo.angles:

            if not topo.edge(a2, a1)['in_ring3'] or not topo.edge(a2, a3)['in_ring3']:
                theta = get_angle(topo.coords[[a1, a2, a3]])[0]
                if theta > 2.9671:  # if angle is larger than 170 degree, make it 180
                    theta = np.pi

                b21 = topo.edge(a2, a1)['vers']
                b23 = topo.edge(a2, a3)['vers']
                a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                                 f"{topo.types[a2]}({b23}){topo.types[a3]}"])
                a_type = f"{a_type[0]}_{a_type[1]}"
                angle_terms.append(cls([a1, a2, a3], theta, a_type, 1))

        return angle_terms


class UreyAngleTerm(TermBase):
    name = 'UreyAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_bonds(crd, self.atomids[::2], self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):
        urey_terms = cls.get_terms_container()

        for a1, a2, a3 in topo.angles:
            dist = get_dist(topo.coords[a1], topo.coords[a3])[1]
            theta = get_angle(topo.coords[[a1, a2, a3]])[0]
            #  No Urey term  if linear angle (>170) or if in 3-member ring
            if theta < 2.9671 and (not topo.edge(a2, a1)['in_ring3'] or
                                   not topo.edge(a2, a3)['in_ring3']):
                b21, b23 = topo.edge(a2, a1)['vers'], topo.edge(a2, a3)['vers']

                a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                                 f"{topo.types[a2]}({b23}){topo.types[a3]}"])
                a_type = f"{a_type[0]}_{a_type[1]}"

                urey_terms.append(cls([a1, a2, a3], dist, a_type, 1))

        return urey_terms


class CrossBondBondTerm(TermBase):
    name = 'CrossBondBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_bond_bond(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):
        cross_bond_bond_terms = cls.get_terms_container()

        for a1, a2, a3 in topo.angles:
            theta = get_angle(topo.coords[[a1, a2, a3]])[0]
            #  No CrossBondAngle term  if linear angle (>170) or if in 3-member ring
            if theta < 2.9671 and (not topo.edge(a2, a1)['in_ring3'] or
                                   not topo.edge(a2, a3)['in_ring3']):
                dist12 = get_dist(topo.coords[a1], topo.coords[a2])[1]
                dist32 = get_dist(topo.coords[a3], topo.coords[a2])[1]
                dists = np.array([dist12, dist32])

                b21 = topo.edge(a2, a1)['vers']
                b23 = topo.edge(a2, a3)['vers']
                a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                                 f"{topo.types[a2]}({b23}){topo.types[a3]}"])
                a_type = f"{a_type[0]}_{a_type[1]}"
                # Second argument in constructor of TermABC is the equilibrium data
                cross_bond_bond_terms.append(cls([a1, a2, a3], dists, a_type, 1))

        return cross_bond_bond_terms


class CrossBondAngleTerm(TermBase):
    name = 'CrossBondAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cross_bond_angle(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):

        cross_bond_angle_terms = cls.get_terms_container()

        for a1, a2, a3 in topo.angles:
            theta = get_angle(topo.coords[[a1, a2, a3]])[0]
            #  No CrossBondAngle term  if linear angle (>170) or if in 3-member ring
            if theta < 2.9671 and (not topo.edge(a2, a1)['in_ring3'] or
                                   not topo.edge(a2, a3)['in_ring3']):
                dist12 = get_dist(topo.coords[a1], topo.coords[a2])[1]
                dist32 = get_dist(topo.coords[a3], topo.coords[a2])[1]
                dist13 = get_dist(topo.coords[a1], topo.coords[a3])[1]
                dists = np.array([dist12, dist32, dist13])

                b21 = topo.edge(a2, a1)['vers']
                b23 = topo.edge(a2, a3)['vers']
                a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                                 f"{topo.types[a2]}({b23}){topo.types[a3]}"])
                a_type = f"{a_type[0]}_{a_type[1]}"
                cross_bond_angle_terms.append(cls([a1, a2, a3], dists, a_type, 1))

        return cross_bond_angle_terms
