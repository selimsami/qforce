from itertools import product, combinations
import numpy as np
#
from .baseterms import TermABC, TermFactory
from ..forces import get_dihed, get_angle
from ..forces import (calc_harmonic_diheds, calc_rb_diheds, calc_inversion, calc_cos_cube_diheds, calc_pitorsion_diheds,
                      calc_periodic_dihed)
from ..forces import lsq_rb_diheds


class DihedralBaseTerm(TermABC):
    @staticmethod
    def get_type(topo, a1, a2, a3, a4):
        # Use this only for rigid dihedral - to be moved
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

        phi = get_dihed(topo.coords[[a1, a2, a3, a4]])[0]
        phi = np.degrees(abs(phi))

        # To prevent very different angles being considered the same term
        if phi < 30:
            ang = 0
        elif 30 <= phi < 90:
            ang = 60
        elif 90 <= phi < 150:
            ang = 120
        elif 150 <= phi:
            ang = 180

        return f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"  #-{ang}

    @staticmethod
    def remove_linear_angles(coords, a1s, a2, a3, a4s):
        # Don't add a dihedral if its 3-atom planes have an angle > 170 degrees
        a1s = [a1 for a1 in a1s if get_angle(coords[[a1, a2, a3]])[0] < 2.9671]
        a4s = [a4 for a4 in a4s if get_angle(coords[[a4, a3, a2]])[0] < 2.9671]
        return a1s, a4s

    @staticmethod
    def check_angle(phi):
        if abs(phi) < 0.5236:  # if it's smaller than +- 30 degrees, make it 0
            phi = 0
        elif 2.6180 < abs(phi):  # if it's larger than 150 degrees, make it 180
            phi = np.pi
        return phi

    @classmethod
    def get_term(cls, topo, atomids, d_type):
        phi = get_dihed(topo.coords[atomids])[0]
        phi = DihedralBaseTerm.check_angle(phi)
        return cls(atomids, phi, d_type)


class PeriodicDihedralTerm(DihedralBaseTerm):
    name = 'PeriodicDihedralTerm'

    @classmethod
    def get_term(cls, topo, atomids, multiplicity, phi0, d_type):
        return cls(atomids, [multiplicity, phi0], d_type)

    def _calc_forces(self, crd, force, fconst):
        return calc_periodic_dihed(crd, self.atomids, self.equ, fconst, force)

    def write_forcefield(self, software, writer):
        software.write_periodic_dihedral_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_periodic_dihedral_header(writer)


class HarmonicDihedralTerm(DihedralBaseTerm):
    name = 'HarmonicDihedralTerm'

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        phi = DihedralBaseTerm.check_angle(phi)
        return cls(atomids, phi, d_type)

    def _calc_forces(self, crd, force, fconst):
        return calc_harmonic_diheds(crd, self.atomids, self.equ, fconst, force)
        # return calc_periodic_dihed(crd, self.atomids, self.equ, fconst, force)

    def write_forcefield(self, software, writer):
        software.write_harmonic_dihedral_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_harmonic_dihedral_header(writer)


class RBDihedralTerm(DihedralBaseTerm):
    name = 'RBDihedralTerm'
    idx_buffer = 6

    def _calc_forces(self, crd, force, fconst):
        """fconst is not used for this term"""
        return calc_rb_diheds(crd, self.atomids, self.equ, force)

    def do_fitting(self, crd, energies, forces):
        """compute fitting contributions"""
        en = lsq_rb_diheds(crd, self.atomids, forces[self.idx:self.idx+self.idx_buffer])
        for i, ele in enumerate(en):
            energies[self.idx + i] = ele

    def set_fitparameters(self, parameters):
        """set the parameters after fitting"""
        self.equ = np.array([val for val in parameters[self.idx:self.idx+self.idx_buffer]], dtype=float)

    @classmethod
    def get_term(cls, topo, atoms, d_type):
        return cls(atoms, np.zeros(6), d_type)

    def write_forcefield(self, software, writer):
        software.write_rb_dihed_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_rb_dihed_header(writer)


class InversionDihedralTerm(DihedralBaseTerm):
    name = 'InversionDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_inversion(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        return cls(atomids, phi, d_type)

    def write_forcefield(self, software, writer):
        software.write_inversion_dihed_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_inversion_dihed_header(writer)


class CosCubeDihedralTerm(DihedralBaseTerm):
    name = 'CosCubeDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cos_cube_diheds(crd, self.atomids, fconst, force)

    @classmethod
    def get_term(cls, topo, atomids, d_type):
        return cls(atomids, None, d_type)

    def write_forcefield(self, software, writer):
        software.write_cos_cube_dihedral_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cos_cube_dihedral_header(writer)


class PiTorsionDihedralTerm(DihedralBaseTerm):
    name = 'PiTorsionDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_pitorsion_diheds(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        phi = DihedralBaseTerm.check_angle(phi)
        return cls(atomids, phi, d_type)

    def write_forcefield(self, software, writer):
        software.write_pitorsion_dihedral_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_pitorsion_dihedral_header(writer)


class DihedralTerms(TermFactory):
    name = 'DihedralTerms'

    _term_types = {
        'rigid': PeriodicDihedralTerm,
        'improper': HarmonicDihedralTerm,

        # 'flexible': { # PeriodicDihedralTerm, # CosCubeDihedralTerm,# ,RBDihedralTerm
        #     'periodic': PeriodicDihedralTerm,
        #     'cos_cube': CosCubeDihedralTerm,
        # },

        'flexible': PeriodicDihedralTerm,  # CosCubeDihedralTerm,# ,RBDihedralTerm
        'cos_cube': CosCubeDihedralTerm,

        'inversion': InversionDihedralTerm,
        'pitorsion': PiTorsionDihedralTerm,
    }

    _always_on = []
    _default_off = []

    @classmethod
    def _get_terms(cls, topo, non_bonded, termtypes):
        terms = cls.get_terms_container(termtypes)

        # helper functions to improve readability
        def add_term(name, topo, atoms, *args):
            term = termtypes[name].get_term(topo, atoms, *args)
            if term is not None:
                terms[name].append(term)

        def get_dtype(topo, *args):
            return DihedralBaseTerm.get_type(topo, *args)

        # proper dihedrals
        for a2, a3 in topo.bonds:
            central = topo.edge(a2, a3)
            a1s = [a1 for a1 in topo.neighbors[0][a2] if a1 != a3]
            a4s = [a4 for a4 in topo.neighbors[0][a3] if a4 != a2]

            a1s, a4s = DihedralBaseTerm.remove_linear_angles(topo.coords, a1s, a2, a3, a4s)

            if a1s == [] or a4s == []:
                continue

            a1s, a4s = np.array(a1s), np.array(a4s)
            atoms_comb = [list(d) for d in product(a1s, [a2], [a3],
                          a4s) if d[0] != d[-1]]

            if (central['order'] >= 1.75 or central["in_ring3"]  # double bond or 3-member ring
                    or (central['in_ring'] and central['order'] >= 1.25)  # in ring and conjugated
                    or (all([topo.node(a)['n_ring'] > 1 for a in [a2, a3]]) and  # in many rings
                        any([topo.node(a)['n_ring'] > 1 for a in a1s]) and
                        any([topo.node(a)['n_ring'] > 1 for a in a4s]))
                    or (central['in_ring'] and check_if_in_a_fully_planar_ring(topo, a2, a3))
                    or topo.all_rigid):

                for atoms in atoms_comb:
                    d_type = get_dtype(topo, *atoms)
                    phi = get_dihed(topo.coords[atoms])[0]
                    # add_term('rigid', topo, atoms, phi, d_type)
                    add_term('rigid', topo, atoms, 2., np.pi, d_type+'_mult2')

            elif central['in_ring']:
                atoms_in_ring = [a for a in atoms_comb if any(set(a).issubset(set(r))
                                 for r in topo.rings)]

                for atoms in atoms_in_ring:
                    phi = get_dihed(topo.coords[atoms])[0]
                    d_type = get_dtype(topo, *atoms)

                    if abs(phi) < 0.43625:  # check planarity < 25 degrees
                        add_term('rigid', topo, atoms, phi, d_type)
                    else:
                        add_term('inversion', topo, atoms, phi, d_type)

            else:
                for atoms in atoms_comb:
                    d_type = get_dtype(topo, *atoms)
                    add_term('cos_cube', topo, atoms, d_type)
                    add_term('flexible', topo, atoms, 3, 0, d_type+'_mult3')
                    add_term('flexible', topo, atoms, 2, np.pi, d_type+'_mult2')
                    add_term('flexible', topo, atoms, 1, 0, d_type+'_mult1')

        # improper dihedrals
        for i in range(topo.n_atoms):
            bonds = list(topo.graph.neighbors(i))
            if len(bonds) != 3:
                continue
            if len(bonds) >= 3:
                triplets = (combinations(bonds, 3))
            else:
                continue

            for bonds in triplets:

                atoms = [i, -1, -1, -1]
                n_bond = [len(list(topo.graph.neighbors(b))) for b in bonds]
                non_ring = [a for a in bonds if not topo.edge(i, a)['in_ring']]

                if len(non_ring) == 1:
                    atoms[3] = non_ring[0]
                else:
                    atoms[3] = bonds[n_bond.index(min(n_bond))]

                for b in bonds:
                    if b not in atoms:
                        atoms[atoms.index(-1)] = b

                phi = get_dihed(topo.coords[atoms])[0]
                # Only add improper dihedrals if there is no stiff dihedral
                # on the central improper atom and one of the neighbors
                bonds = [sorted([b, i]) for b in bonds]
                # if any(b == list(term.atomids[1:3]) for term in terms['rigid'] for b in bonds):
                #     continue
                imp_type = f"ki_{topo.types[i]}"
                if abs(phi) < 0.43625:  # check planarity < 25 degrees
                    # add_term('improper', topo, atoms, phi, imp_type)
                    add_term('improper', topo,  [atoms[0], atoms[1], atoms[2], atoms[3]], phi, imp_type)
                    add_term('improper', topo, [atoms[0], atoms[1], atoms[3], atoms[2]], phi, imp_type)
                    add_term('improper', topo, [atoms[0], atoms[2], atoms[3], atoms[1]], phi, imp_type)
                else:
                    add_term('inversion', topo,  [atoms[0], atoms[1], atoms[2], atoms[3]], phi, imp_type)
                    add_term('inversion', topo, [atoms[0], atoms[1], atoms[3], atoms[2]], phi, imp_type)
                    add_term('inversion', topo, [atoms[0], atoms[2], atoms[1], atoms[3]], phi, imp_type)
                    add_term('inversion', topo, [atoms[0], atoms[2], atoms[3], atoms[1]], phi, imp_type)
                    add_term('inversion', topo, [atoms[0], atoms[3], atoms[1], atoms[2]], phi, imp_type)
                    add_term('inversion', topo, [atoms[0], atoms[3], atoms[2], atoms[1]], phi, imp_type)

        # add_term('pitorsion', topo, [1, 0, 3, 2, 4, 5], 0, 'test')
        # add_term('pitorsion', topo, [0, 2, 3, 1, 4, 5], 0, 'test')

        return terms


def check_if_in_a_fully_planar_ring(topo, a2, a3):
    rings = [r for r in topo.rings if set([a2, a3]).issubset(set(r))]
    for ring in rings:
        is_planar = []
        ring_graph = topo.graph.subgraph(ring)
        for edge in ring_graph.edges:
            a1 = [n for n in list(ring_graph.neighbors(edge[0])) if n not in edge][0]
            a4 = [n for n in list(ring_graph.neighbors(edge[1])) if n not in edge][0]
            dihed = [a1, edge[0], edge[1], a4]

            is_planar.append(get_dihed(topo.coords[dihed])[0] < 0.43625)  # < 25 degrees
        if all(is_planar):
            all_planar = True
            break
    else:
        all_planar = False
    return all_planar
