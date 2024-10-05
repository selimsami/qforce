import numpy as np
#
from .baseterms import TermBase
from ..forces import get_dist, get_angle
from ..forces import calc_bonds, calc_morse_bonds, calc_angles, calc_cosine_angles


class HarmonicBondTerm(TermBase):
    name = 'HarmonicBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_bonds(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.bondname(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            bond_terms.append(cls([a1, a2], dist, bond['vers']))

        return bond_terms

    def write_forcefield(self, software, writer):
        software.write_harmonic_bond_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_harmonic_bond_header(writer)


class MorseBondTerm(TermBase):
    name = 'MorseBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_morse_bonds(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.bondname(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ[0] = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            bond_terms.append(cls([a1, a2], [dist, 2.2], bond['vers']))

        return bond_terms

    def write_forcefield(self, software, writer):
        software.write_morse_bond_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_morse_bond_header(writer)


class HarmonicAngleTerm(TermBase):
    name = 'HarmonicAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_angles(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.anglename(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
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
                angle_terms.append(cls([a1, a2, a3], theta, a_type))

        return angle_terms

    def write_forcefield(self, software, writer):
        software.write_harmonic_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_harmonic_angle_header(writer)


class CosineAngleTerm(HarmonicAngleTerm):
    name = 'CosineAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cosine_angles(crd, self.atomids, self.equ, fconst, force)

    def write_forcefield(self, software, writer):
        software.write_cosine_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cosine_angle_header(writer)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.anglename(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value


class UreyBradleyTerm(TermBase):
    name = 'UreyBradleyTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_bonds(crd, self.atomids[::2], self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.bondname(*self.atomids[::2])]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
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

                urey_terms.append(cls([a1, a2, a3], dist, a_type))

        return urey_terms

    def write_forcefield(self, software, writer):
        software.write_urey_bradley_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_urey_bradley_header(writer)
