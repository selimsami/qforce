from itertools import product
import numpy as np
#
from .baseterms import TermABC, TermFactory
from ..forces import get_dihed, get_angle
from ..forces import calc_imp_diheds, calc_rb_diheds, calc_inversion  # , calc_periodic_dihed


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

        return f"{d_type[0]}_{t23[0]}({b23}){t23[1]}_{d_type[1]}~{ang}"

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
    def get_term(cls, topo, atoms, d_type):
        return cls(atoms, np.zeros(6), d_type)


class InversionDihedralTerm(DihedralBaseTerm):

    name = 'InversionDihedralTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_inversion(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_term(cls, topo, atomids, phi, d_type):
        return cls(atomids, phi, d_type)


class DihedralTerms(TermFactory):

    name = 'DihedralTerms'

    _term_types = {
        'rigid': RigidDihedralTerm,
        'improper': ImproperDihedralTerm,
        'flexible': FlexibleDihedralTerm,
        'inversion': InversionDihedralTerm,
    }

    _always_on = []
    _default_off = []

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

            if (central['order'] >= 1.75 or central["in_ring3"]  # double bond or 3-member ring
                    or (central['in_ring'] and central['order'] >= 1.25)  # in ring and conjugated
                    or (all([topo.node(a)['n_ring'] > 1 for a in [a2, a3]]) and  # in many rings
                        any([topo.node(a)['n_ring'] > 1 for a in a1s]) and
                        any([topo.node(a)['n_ring'] > 1 for a in a4s]))
                    or (central['in_ring'] and check_if_in_a_fully_planar_ring(topo, a2, a3))
                    or topo.all_rigid):
                # rigid
                for atoms in atoms_comb:
                    d_type = get_dtype(topo, *atoms)
                    add_term('rigid', topo, atoms, d_type)

            elif central['in_ring']:
                atoms_in_ring = [a for a in atoms_comb if any(set(a).issubset(set(r))
                                 for r in topo.rings)]

                for atoms in atoms_in_ring:
                    phi = get_dihed(topo.coords[atoms])[0]
                    d_type = get_dtype(topo, *atoms)

                    if abs(phi) < 0.43625:  # check planarity < 25 degrees
                        add_term('rigid', topo, atoms, d_type)
                    else:
                        add_term('inversion', topo, atoms, phi, d_type)

            else:
                atoms = find_flexible_atoms(topo, a1s, a2, a3, a4s)
                for dih in atoms:
                    atypes = [topo.types[dih[0]], topo.types[dih[3]]]
                    if central['vers'].startswith(topo.types[a3]):
                        atypes = atypes[::-1]
                    d_type = f"{central['vers']}_{atypes[0]}-{atypes[1]}"
                    add_term('flexible', topo, dih, d_type)

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
            if abs(phi) < 0.43625:  # check planarity < 25 degrees
                add_term('improper', topo, atoms, phi, imp_type)
            else:
                add_term('inversion', topo, atoms, phi, imp_type)
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


def find_flexible_atoms(topo, a1s, a2, a3, a4s):
    def pick_end_atom(atom_list):
        n_neighbors = topo.n_neighbors[atom_list]
        choices = atom_list[n_neighbors > 1]
        if len(choices) == 0:
            choices = atom_list
        # if len(choices) > 1:
        #     heaviest_elem = np.amax(topo.elements[choices])
        #     choices = choices[topo.elements[choices] == heaviest_elem]
        return choices

    a1s = pick_end_atom(a1s)
    a4s = pick_end_atom(a4s)

    priority = [[] for _ in range(6)]

    for a1, a4 in product(a1s, a4s):
        phi = np.degrees(abs(get_dihed(topo.coords[[a1, a2, a3, a4]])[0]))
        if phi > 155:
            priority[0].append([a1, a4])
        elif phi < 25:
            priority[1].append([a1, a4])
        elif topo.edge(a1, a2)['in_ring'] and topo.edge(a3, a4)['in_ring']:
            priority[5].append([a1, a4])
        elif topo.edge(a1, a2)['in_ring'] or topo.edge(a3, a4)['in_ring']:
            priority[4].append([a1, a4])
        elif 55 < phi < 65 or 85 < phi < 95:
            priority[2].append([a1, a4])
        else:
            priority[3].append([a1, a4])

    ordered = [p for prio in priority for p in prio]
    a1_select, a4_select = ordered[0]
    atoms = [[a1_select, a2, a3, a4_select]]

    for a1, a4 in ordered[1:]:
        if ((len(topo.neighbors[0][a1]) > 1 or len(topo.neighbors[0][a2]) == 2) and
                (len(topo.neighbors[0][a4]) > 1 or len(topo.neighbors[0][a3]) == 2)):
            atoms.append([a1, a2, a3, a4])

    # add a second dihedral term if minimum is non-planar
    # if priority[0] == [] and priority[1] == []:
    #     phi0 = np.degrees(get_dihed(topo.coords[[a1_select, a2, a3, a4_select]])[0])
    #     for a1, a4 in product(a1s, a4s):
    #         phi = np.degrees(get_dihed(topo.coords[[a1, a2, a3, a4]])[0])
    #         if 100 < (phi0-phi) % 360 < 140 or 220 < (phi0-phi) % 360 < 260:
    #             atoms.append([a1, a2, a3, a4])
    #             break
    return atoms
