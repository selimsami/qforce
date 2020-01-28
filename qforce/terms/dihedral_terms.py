from itertools import product
#
import numpy as np
#
from .baseterms import TermABC, TermFactory, MappingIterator
#
from ..forces import get_dihed
from ..forces import calc_imp_diheds
from ..elements import ATOMMASS

"""
*TODO*: FLEXIBLE DIHEDRALS ARE WRONG

"""


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

    @classmethod
    def get_term(cls, topo, atomids, d_type):
        phi = get_dihed(topo.coords[atomids])[0]
        return cls(atomids, phi, d_type)


class RigidDihedralTerm(DihedralBaseTerm):

    def _calc_forces(self, crd, force, fconst):
        calc_imp_diheds(crd, self.atomids, self.equ, fconst, self.idx, force)


class ImproperDihedralTerm(DihedralBaseTerm):

    def _calc_forces(self, crd, force, fconst):
        calc_imp_diheds(crd, self.atomids, self.equ, fconst, self.idx, force)

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        return cls(atomids, phi, d_type)


class FlexibleDihedralTerm(DihedralBaseTerm):

    def _calc_forces(self, crd, force, fconst):
        ...

    @classmethod
    def get_term(cls, topo, atoms_combin):
        """@Selim: what is atom_combin?

        TODO: write atoms_comb as a function of atomids!

        """
        heaviest = 0
        for a1, a2, a3, a4 in atoms_combin:
            mass = ATOMMASS[topo.atomids[a1]] + ATOMMASS[topo.atomids[a4]]
            if mass > heaviest:
                atoms = np.array((a1, a2, a3, a4))
                heaviest = mass
        return cls(atoms, 0, topo.edge(a2, a3)['vers'])


class ConstrDihedralTerm(DihedralBaseTerm):

    def _calc_forces(self, crd, force, fconst):
        ...

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        return cls(atomids, phi, d_type)

class DihedralTerms(TermFactory):

    _types = {
            'rigid': RigidDihedralTerm,
            'improper': ImproperDihedralTerm,
            'flexible': FlexibleDihedralTerm,
            'constr': ConstrDihedralTerm,
            }

    class DihedralTermsContainer(MappingIterator):
        pass

    @classmethod
    def get_terms(cls, topo):
        """

            Args:
                topo: Topology object, const
                    Stores all topology information

            Return:
                list of cls objects

        """
        terms = cls.DihedralTermsContainer({key: [] for key in cls._types.keys()})

        # helper functions to improve readability
        def add_term(name, topo, atoms, *args):
            terms[name].append(cls._types[name].get_term(topo, atoms, *args))

        def get_dtype(topo, *args):
            DihedralBaseTerm.get_type(topo, *args)
        # proper dihedrals
        for a2, a3 in topo.bonds:
            central = topo.edge(a2, a3)
            a1s = [a1 for a1 in topo.neighbors[0][a2] if a1 != a3]
            a4s = [a4 for a4 in topo.neighbors[0][a3] if a4 != a2]

            if a1s == [] or a4s == []:
                continue

            atoms_comb = [list(d) for d in product(a1s, [a2], [a3],
                          a4s) if d[0] != d[-1]]
            if (central['order'] > 1.5 or central["in_ring3"]
                    or (central['in_ring'] and central['order'] > 1)
                    or all([topo.node(a)['n_ring'] > 2 for a in [a2, a3]])):
                # rigid
                for atoms in atoms_comb:
                    d_type = get_dtype(topo, *atoms)
                    add_term('rigid', topo, atoms, d_type)

            elif central['in_ring']:
                atoms_r = [a for a in atoms_comb if any(set(a).issubset(set(r))
                           for r in topo.rings)][0]
                phi = get_dihed(topo.coords[atoms_r])[0]
                if abs(phi) < 0.07:
                    # rigid
                    d_type = get_dtype(topo, *atoms_r)
                    for atoms in atoms_comb:
                        add_term('rigid', topo, atoms, d_type)
                else:
                    d_type = get_dtype(topo, *atoms_r)
                    for atoms in atoms_comb:
                        add_term('constr', topo, atoms, phi, d_type)
            else:
                add_term('flexible', topo, atoms_comb)

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
            if abs(phi) < 0.07:  # check planarity <4 degrees
                add_term('improper', topo, atoms, phi, imp_type)
            else:
                add_term('constr', topo, atoms, phi, imp_type)
        return terms
