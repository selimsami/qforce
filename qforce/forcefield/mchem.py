import numpy as np
import xml.etree.cElementTree as ET
from ..elements import ATOM_SYM

from datetime import datetime
from ..misc import LOGO
from .forcefield_base import ForcefieldSettings


class MChem(ForcefieldSettings):

    always_on_terms = ['bond', 'angle']

    _optional_terms = {
            'cross_bond_bond': False,
            'cross_bond_angle': False,
            'cross_angle_angle': False,
            #
            'dihedral/rigid': True,
            'dihedral/improper': True,
            'dihedral/flexible': True,
            'dihedral/inversion': True,
            #
            'charge_flux/bond': True,
            'charge_flux/bond_prime': True,
            'charge_flux/angle': True,
            'charge_flux/angle_prime': True,
            'charge_flux/_bond_bond': False,
            'charge_flux/_bond_angle': False,
            'charge_flux/_angle_angle': False,
            #
            'local_frame': True,
    }

    _term_types = {
            'bond': ('morse', ['morse', 'harmonic']),
            'angle': ('cosine', ['cosine', 'harmonic']),
            'cross_bond_angle': ('false', ['bond_angle', 'bond_cos_angle', 'false']),
            'cross_angle_angle':  ('false', ['harmonic', 'cosine', 'false'])
    }

    def __init__(self, ff):
        self.ff = ff

    def write(self, directory, coords, box=[20., 20., 20.]):
        self.write_xml(directory, box)
        self.write_pdb(directory, coords, box)

    def write_pdb(self, directory, coords, box):
        with open(f'{directory}/gas.pdb', 'w') as file:
            file.write(f'CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1\n')
            file.write('MODEL     1\n')

            for i, (coord, name, symbol) in enumerate(zip(coords, self.ff.atom_names, self.ff.symbols), start=1):
                file.write(f'ATOM  {i:5d} {name:4s} {self.ff.residue:4s} {1:4d}    '
                           f'{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1:6.2f}{0:6.2f}          {symbol:2s}  \n')
            file.write('ENDMDL\n')

    def write_xml(self, directory, box):
        forcefield = ET.Element('Forcefield')

        self.write_atomtypes(forcefield)
        self.write_residues(forcefield)
        self.write_forces(forcefield)

        tree = ET.ElementTree(forcefield)
        ET.indent(tree)
        tree.write(f'{directory}/{self.ff.mol_name}_qforce.xml')

    def write_atomtypes(self, forcefield):
        atomtypes = ET.SubElement(forcefield, 'AtomTypes')
        for i in range(self.ff.n_atoms):
                    ET.SubElement(atomtypes, 'Type', {'name': str(i+1), 'class': str(i+1), 'element': self.ff.symbols[i],
                                                      'mass': str(self.ff.masses[i])})

    def write_residues(self, forcefield):
        residues = ET.SubElement(forcefield, 'Residues')
        residue = ET.SubElement(residues, 'Residue', {'name': self.ff.residue})
        for i, name in enumerate(self.ff.atom_names, start=1):
            ET.SubElement(residue, 'Atom', {'name': name, 'type': str(i)})
        for bond in self.ff.terms['bond']:
            ET.SubElement(residue, 'Bond', {'from': str(bond.atomids[0]+1), 'to': str(bond.atomids[1]+1)})

    def write_forces(self, forces):
        writer_dict = {}
        for term in self.ff.terms:
            # print(term.name, term.atomids+1, term.equ, term.fconst)

            if term.name not in writer_dict:
                writer = term.write_ff_header(self, forces)
                writer_dict[term.name] = writer

            term.write_forcefield(self, writer_dict[term.name])

    def write_charge_flux_bond_header(self, forces):
        force = ET.SubElement(forces, 'ChargeFluxBondForce')
        return force

    def write_charge_flux_bond_term(self, term, writer):
        # Charge flux between atom1 and atom2 for the bond distortion between atoms2 atom3
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ, 9))
        k = str(round(term.fconst, 5))
        ET.SubElement(writer, 'Flux', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'length': equ, 'j': k})

    def write_charge_flux_angle_header(self, forces):
        force = ET.SubElement(forces, 'ChargeFluxAngleForce')
        return force

    def write_charge_flux_angle_term(self, term, writer):
        # Charge flux between atom1 and atom3 for the angle distortion atoms2-atom3-atom4
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ, 9))
        k = str(round(term.fconst, 5))
        ET.SubElement(writer, 'Flux', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                       'angle': equ, 'j': k})

    def write_multipole_header(self, forces):
        force = ET.SubElement(forces, 'MultipoleForce')
        return force

    def write_multipole_term(self, term, writer):
        ids = np.zeros(4, dtype=int)
        ids[:len(term.atomids)] = term.atomids + 1
        center = term.atomids[term.center]+1

        ET.SubElement(writer, 'Multipole', {'class1': str(ids[0]), 'class2': str(ids[1]),
                                            'class3': str(ids[2]), 'class4': str(ids[3]), 'center': str(center),
                                            'frametype': term.frame_type, 'q00': str(term.q),
                                            'q10': str(term.dipole_spher[0]), 'q11c': str(term.dipole_spher[1]),
                                            'q11s': str(term.dipole_spher[2]), 'q20': str(term.quad_spher[0]),
                                            'q21c': str(term.quad_spher[1]), 'q21s': str(term.quad_spher[2]),
                                            'q22c': str(term.quad_spher[3]), 'q22s': str(term.quad_spher[4])
                                            })

    def write_harmonic_bond_header(self, forces):
        force = ET.SubElement(forces, 'HarmonicBondForce')
        return force

    def write_harmonic_bond_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ, 9))
        k = str(round(term.fconst, 5))
        ET.SubElement(writer, 'Bond', {'class1': ids[0], 'class2': ids[1], 'length': equ, 'k': k})

    def write_morse_bond_header(self, forces):
        force = ET.SubElement(forces, 'MorseBondForce')
        return force

    def write_morse_bond_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ[0], 9))
        k = str(round(term.fconst, 5))
        e_dis = str(self.ff.bond_dissociation_energies[term.atomids[0],  term.atomids[1]])
        ET.SubElement(writer, 'Bond', {'class1': ids[0], 'class2': ids[1], 'length': equ, 'k': k, 'e_dis': e_dis})

    def write_harmonic_angle_header(self, forces):
        force = ET.SubElement(forces, 'HarmonicAngleForce')
        return force

    def write_harmonic_angle_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ, 8))
        k = str(round(term.fconst, 6))
        ET.SubElement(writer, 'Angle', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'angle': equ, 'k': k})

    def write_cosine_angle_header(self, forces):
        force = ET.SubElement(forces, 'CosineAngleForce')
        return force

    def write_cosine_angle_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ, 8))
        k = str(round(term.fconst, 6))
        ET.SubElement(writer, 'Angle', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'angle': equ, 'k': k})

    def write_cross_bond_bond_header(self, forces):
        force = ET.SubElement(forces, 'StretchStretchHarmonicForce')
        return force

    def write_cross_bond_bond_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ1 = str(round(term.equ[0], 9))
        equ2 = str(round(term.equ[1], 9))
        k = str(round(term.fconst, 8))
        ET.SubElement(writer, 'StretchStretch', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                                 'length1': equ1, 'length2': equ2, 'k': k})

    def write_cross_bond_angle_header(self, forces):
        force = ET.SubElement(forces, 'StretchBendCouplingHarmonicForce')
        return force

    def write_cross_bond_angle_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ1 = str(round(term.equ[0], 8))
        equ2 = str(round(term.equ[1], 9))
        k = str(round(term.fconst, 7))
        ET.SubElement(writer, 'StretchBend', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                              'class5': ids[4], 'length': equ1, 'angle': equ2, 'k': k})

    def write_cross_bond_angle_header(self, forces):
        force = ET.SubElement(forces, 'StretchBendCouplingCosineForce')
        return force

    def write_cross_bond_angle_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ1 = str(round(term.equ[0], 8))
        equ2 = str(round(term.equ[1], 9))
        k = str(round(term.fconst, 7))
        ET.SubElement(writer, 'StretchBend', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                              'class5': ids[4], 'length': equ1, 'angle': equ2, 'k': k})

    def write_cross_bond_cos_angle_header(self, forces):
        force = ET.SubElement(forces, 'StretchBendCouplingCosineForce')
        return force

    def write_cross_bond_cos_angle_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ1 = str(round(term.equ[0], 8))
        equ2 = str(round(term.equ[1], 9))
        k = str(round(term.fconst, 7))
        ET.SubElement(writer, 'StretchBend', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                              'class5': ids[4], 'length': equ1, 'angle': equ2, 'k': k})

    def write_cross_angle_angle_header(self, forces):
        force = ET.SubElement(forces, 'BendBendHarmonicForce')
        return force

    def write_cross_angle_angle_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ1 = str(round(term.equ[0], 8))
        equ2 = str(round(term.equ[1], 8))
        k = str(round(term.fconst, 7))
        ET.SubElement(writer, 'BendBend', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                           'class5': ids[4], 'class6': ids[5], 'angle1': equ1, 'angle2': equ2, 'k': k})

    def write_cross_cos_angle_angle_header(self, forces):
        force = ET.SubElement(forces, 'BendBendCosineForce')
        return force

    def write_cross_cos_angle_angle_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ1 = str(round(term.equ[0], 8))
        equ2 = str(round(term.equ[1], 8))
        k = str(round(term.fconst, 7))
        ET.SubElement(writer, 'BendBend', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                           'class5': ids[4], 'class6': ids[5], 'angle1': equ1, 'angle2': equ2, 'k': k})

    def write_harmonic_dihedral_header(self, forces):
        force = ET.SubElement(forces, 'HarmonicDihedralForce')
        return force

    def write_harmonic_dihedral_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ, 8))
        k = str(round(term.fconst, 7))

        ET.SubElement(writer, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                          'angle': equ, 'k': k})

    def write_periodic_dihedral_header(self, forces):
        force = ET.SubElement(forces, 'PeriodicDihedralForce')
        return force

    def write_periodic_dihedral_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        k = str(round(term.fconst, 7))
        n = str(term.equ[0])
        phi0 = str(round(term.equ[1], 8))

        ET.SubElement(writer, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                          'n': n, 'shift': phi0, 'k': k})

    def write_inversion_dihedral_header(self, forces):
        force = ET.SubElement(forces, 'InversionDihedralForce')
        return force

    def write_inversion_dihedral_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ = str(round(term.equ, 8))
        k = str(round(term.fconst, 7))

        ET.SubElement(writer, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                          'angle': equ, 'k': k})


    def write_rb_dihedral_header(self, forces):
        force = ET.SubElement(forces, 'RBDihedralForce')
        return force

    def write_rb_dihedral_term(self, term, writer):
        ids = [str(i+1) for i in term.atomids]
        equ = [str(round(e, 8)) for e in term.equ]

        ET.SubElement(writer, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                          'c0': equ[0], 'c1': equ[1], 'c2': equ[2], 'c3': equ[3], 'c4': equ[4],
                                          'c5': equ[5]})


    # def write_cross_dihedral_bond(self, forces):
    #     force = ET.SubElement(forces, 'StretchTorsionForce')
    #
    #     for term in self.ff.terms['_cross_dihed_angle']:
    #         ids = [str(i+1) for i in term.atomids]
    #         equ = str(round(term.equ, 8))
    #         k = str(round(term.fconst, 7))
    #
    #         ET.SubElement(force, 'StretchTorsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2],
    #                                                 'class4': ids[3], 'angle1': equ, 'k': k})
    #
    # def write_cross_dihedral_angle(self, forces):
    #     force = ET.SubElement(forces, 'BendTorsionForce')
    #
    #     for term in self.ff.terms['_cross_dihed_angle']:
    #         ids = [str(i+1) for i in term.atomids]
    #         equ = str(round(term.equ, 8))
    #         # if self.ff.cos_angle:
    #         #     term.fconst /= np.sin(term.equ[0]) * np.sin(term.equ[1])
    #         k = str(round(term.fconst, 7))
    #
    #         ET.SubElement(force, 'BendTorsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2],
    #                                              'class4': ids[3], 'angle': equ, 'k': k})
