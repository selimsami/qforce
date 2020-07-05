from itertools import product
#
import numpy as np
#
from .baseterms import TermABC, TermFactory
#
from ..forces import get_dihed, get_angle
from ..forces import calc_imp_diheds, calc_rb_diheds


class DihedralBaseTerm(TermABC):

    @staticmethod
    def get_type(topo, a1, a2, a3, a4):
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
        return f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}"

    @staticmethod
    def remove_linear_angles(coords, a1s, a2, a3, a4s):
        # Don't add a dihedral if its 3-atom planes have an angle > 175 degrees
        a1s = [a1 for a1 in a1s if get_angle(coords[[a1, a2, a3]])[0] < 3.0543]
        a4s = [a4 for a4 in a4s if get_angle(coords[[a4, a3, a2]])[0] < 3.0543]
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


class RigidDihedralTerm(DihedralBaseTerm):

    name = 'RigidDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_imp_diheds(crd, self.atomids, self.equ, fconst, force)


class ImproperDihedralTerm(DihedralBaseTerm):

    name = 'ImproperDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_imp_diheds(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        phi = DihedralBaseTerm.check_angle(phi)
        return cls(atomids, phi, d_type)


class FlexibleDihedralTerm(DihedralBaseTerm):

    name = 'FlexibleDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_rb_diheds(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_term(cls, topo, a1s, a2, a3, a4s):
        def pick_end_atom(atom_list):
            n_neighbors = topo.n_neighbors[atom_list]
            choices = atom_list[n_neighbors > 1]
            if len(choices) == 0:
                choices = atom_list
            if len(choices) > 1:
                heaviest_elem = np.amax(topo.elements[choices])
                choices = choices[topo.elements[choices] == heaviest_elem]
            return choices[0]

        a1 = pick_end_atom(a1s)
        a4 = pick_end_atom(a4s)
        atoms = np.array((a1, a2, a3, a4))

        return cls(atoms, np.zeros(6), topo.edge(a2, a3)['vers'])


class ConstrDihedralTerm(DihedralBaseTerm):

    name = 'ConstrDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_rb_diheds(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        return cls(atomids, phi, d_type)


class DihedralTerms(TermFactory):

    name = 'DihedralTerms'

    _term_types = {
        'rigid': RigidDihedralTerm,
        'improper': ImproperDihedralTerm,
        'flexible': FlexibleDihedralTerm,
        'constr': ConstrDihedralTerm,
    }

    @classmethod
    def get_terms(cls, topo, non_bonded):
        terms = cls.get_terms_container()

        # helper functions to improve readability
        def add_term(name, topo, atoms, *args):
            terms[name].append(cls._term_types[name].get_term(topo, atoms, *args))

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

            if (central['order'] > 1.5 or central["in_ring3"]
                    or (central['in_ring'] and central['order'] > 1)
                    or all([topo.node(a)['n_ring'] > 2 for a in [a2, a3]])
                    or topo.all_rigid):
                # rigid
                for atoms in atoms_comb:
                    d_type = get_dtype(topo, *atoms)
                    add_term('rigid', topo, atoms, d_type)

            elif central['in_ring']:
                atoms_r = [a for a in atoms_comb if any(set(a).issubset(set(r))
                           for r in topo.rings)][0]
                phi = get_dihed(topo.coords[atoms_r])[0]
                if abs(phi) < 0.1745:  # check planarity <10 degrees
                    # rigid
                    for atoms in atoms_comb:
                        d_type = get_dtype(topo, *atoms)
                        add_term('rigid', topo, atoms, d_type)
                else:
                    d_type = get_dtype(topo, *atoms_r)
                    add_term('constr', topo, atoms_r, phi, d_type)
            else:
                add_term('flexible', topo, a1s, a2, a3, a4s)

        # improper dihedrals
        for i in range(topo.n_atoms):
            bonds = list(topo.graph.neighbors(i))
            if len(bonds) != 3:
                continue
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
            if any(b == list(term.atomids[1:3]) for term in terms['rigid'] for b in bonds):
                continue
            imp_type = f"ki_{topo.types[i]}"
            if abs(phi) < 0.1745:  # check planarity <10 degrees
                add_term('improper', topo, atoms, phi, imp_type)
            else:
                add_term('constr', topo, atoms, phi, imp_type)
        return terms
