from .baseterms import TermBase
#
from ..forces import get_dist, get_angle
from ..forces import calc_bonds, calc_angles


class BondTerm(TermBase):

    def _calc_forces(self, crd, force, fconst):
        calc_bonds(crd, self.atomids, self.equ, fconst, self.idx, force)

    @classmethod
    def get_terms(cls, topo):
        """get terms"""

        bond_terms = []
        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            type1, type2 = sorted([topo.types[a1], topo.types[a2]])
            bond['vers'] = f"{type1}({bond['order']}){type2}"
            if bond['order'] > 1.5 or bond["in_ring"]:
                bond['breakable'] = False
            else:
                bond['breakable'] = True
            bond_terms.append(cls([a1, a2], dist, bond['vers']))
        return bond_terms


class AngleTerm(TermBase):

    def _calc_forces(self, crd, force, fconst):
        calc_angles(crd, self.atomids, self.equ, fconst, self.idx, force)

    @classmethod
    def get_terms(cls, topo):
        """

            Args:
                topo: Topology object, const
                    Stores all topology information

            Return:
                list of cls objects

        """
        angle_terms = []

        for a1, a2, a3 in topo.angles:
            vec12, _ = get_dist(topo.coords[a1], topo.coords[a2])
            vec32, _ = get_dist(topo.coords[a3], topo.coords[a2])
            theta = get_angle(vec12, vec32)

            b21 = topo.edge(a2, a1)['vers']
            b23 = topo.edge(a2, a3)['vers']
            a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                             f"{topo.types[a2]}({b23}){topo.types[a3]}"])
            a_type = f"{a_type[0]}_{a_type[1]}"
            angle_terms.append(cls([a1, a2, a3], theta, a_type))
        return angle_terms


class UrayAngleTerm(TermBase):

    def _calc_forces(self, crd, force, fconst):
        calc_bonds(crd, self.atomids, self.equ, fconst, self.idx, force)

    @classmethod
    def get_terms(cls, topo):
        """

            Args:
                topo: Topology object, const
                    Stores all topology information

            Return:
                list of cls objects

        """
        urey_terms = []
        for a1, a2, a3 in topo.angles:
            dist = get_dist(topo.coords[a1], topo.coords[a3])[1]
            #
            b21 = topo.edge(a2, a1)['vers']
            b23 = topo.edge(a2, a3)['vers']
            a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                             f"{topo.types[a2]}({b23}){topo.types[a3]}"])
            a_type = f"{a_type[0]}_{a_type[1]}"
            #
            urey_terms.append(cls([a1, a3], dist, a_type))
        return urey_terms
