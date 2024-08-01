import numpy as np
import xml.etree.cElementTree as ET
from ..elements import ATOM_SYM

from datetime import datetime
from ..misc import LOGO
from .forcefield_base import ForcefieldSettings


class MChem(ForcefieldSettings):

    _always_on_terms = ['bond', 'angle']

    _optional_terms = {
            'urey': True,
            'cross_bond_bond': False,
            'cross_bond_angle': False,
            'cross_angle_angle': False,
            'dihedral/rigid': True,
            'dihedral/improper': True,
            'dihedral/flexible': True,
            'dihedral/inversion': True,
            'non_bonded': True,
            'charge_flux/bond': True,
            'charge_flux/bond_prime': True,
            'charge_flux/angle': True,
            'charge_flux/angle_prime': False,
            'charge_flux/_bond_bond': False,
            'charge_flux/_bond_angle': False,
            'charge_flux/_angle_angle': False,
            'local_frame': True,
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
        self.write_bonds(forces)
        self.write_angles(forces)

        for term in self.ff.terms:
            print(term.name, term.atomids+1, term.equ, term.fconst)

        if 'dihedral/improper' in self.ff.terms and len(self.ff.terms['dihedral/improper']) > 0:
            self.write_improper_dihedral(forces)
        if 'dihedral/rigid' in self.ff.terms and len(self.ff.terms['dihedral/rigid']) > 0:
            self.write_rigid_dihedral(forces)
        if 'dihedral/pitorsion' in self.ff.terms and len(self.ff.terms['dihedral/pitorsion']) > 0:
            self.write_pitorsion_dihedral(forces)
        if 'dihedral/inversion' in self.ff.terms and len(self.ff.terms['dihedral/inversion']) > 0:
            self.write_inversion_dihedral(forces)
        if 'cross_bond_bond' in self.ff.terms and len(self.ff.terms['cross_bond_bond']) > 0:
            self.write_cross_bond_bond(forces)
        if 'cross_bond_angle' in self.ff.terms and len(self.ff.terms['cross_bond_angle']) > 0:
            self.write_cross_bond_angle(forces)
            print('cross-bond-angle terms: ', len(self.ff.terms['cross_bond_angle']))
        if 'cross_angle_angle' in self.ff.terms and len(self.ff.terms['cross_angle_angle']) > 0:
            self.write_cross_angle_angle(forces)
            print('cross-angle-angle terms: ', len(self.ff.terms['cross_angle_angle']))
        if '_cross_dihed_angle' in self.ff.terms and len(self.ff.terms['_cross_dihed_angle']) > 0:
            self.write_cross_dihedral_angle(forces)

        if 'charge_flux' in self.ff.terms and len(self.ff.terms['charge_flux']) > 0:
            self.write_charge_flux(forces)
        # Non-bonded
        if 'local_frame' in self.ff.terms and len(self.ff.terms['local_frame']) > 0:
            self.write_multipoles(forces)

    def write_charge_flux(self, forces):
        # Charge flux between atom1 and atom2 for the bond distortion between atoms2 atom3
        force = ET.SubElement(forces, 'ChargeFluxBondForce')
        terms = [self.ff.terms[t] for t in ['charge_flux/bond', 'charge_flux/bond_prime'] if t in self.ff.terms]
        terms = [t for term in terms for t in term]
        for term in terms:
            ids = [str(i+1) for i in term.atomids]
            equ = str(round(term.equ, 9))
            k = str(round(term.fconst, 5))
            ET.SubElement(force, 'Flux', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'length': equ, 'j': k})

        # Charge flux between atom1 and atom3 for the angle distortion atoms2-atom3-atom4
        force = ET.SubElement(forces, 'ChargeFluxAngleForce')
        terms = [self.ff.terms[t] for t in ['charge_flux/angle', 'charge_flux/angle_prime'] if t in self.ff.terms]
        terms = [t for term in terms for t in term]
        for term in terms:
            ids = [str(i+1) for i in term.atomids]
            equ = str(round(term.equ, 9))
            k = str(round(term.fconst, 5))
            ET.SubElement(force, 'Flux', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                          'angle': equ, 'j': k})

    def write_multipoles(self, forces):
        force = ET.SubElement(forces, 'MultipoleForce')
        for i, term in enumerate(self.ff.terms['local_frame'], start=1):
            ids = np.zeros(4, dtype=int)
            ids[:len(term.atomids)] = term.atomids + 1
            center = term.atomids[term.center]+1

            ET.SubElement(force, 'Multipole', {'type': str(i), 'class1': str(ids[0]), 'class2': str(ids[1]),
                                               'class3': str(ids[2]), 'class4': str(ids[3]), 'center': str(center),
                                               'frametype': term.frame_type, 'q00': str(term.q),
                                               'q10': str(term.dipole_spher[0]), 'q11c': str(term.dipole_spher[1]),
                                               'q11s': str(term.dipole_spher[2]), 'q20': str(term.quad_spher[0]),
                                               'q21c': str(term.quad_spher[1]), 'q21s': str(term.quad_spher[2]),
                                               'q22c': str(term.quad_spher[3]), 'q22s': str(term.quad_spher[4])
                                               })

    def write_bonds(self, forces):
        if not self.ff.morse:
            force = ET.SubElement(forces, 'HarmonicBondForce')
            for term in self.ff.terms['bond']:
                ids = [str(i+1) for i in term.atomids]

                ET.SubElement(force, 'Bond', {'class1': ids[0], 'class2': ids[1], 'length': equ, 'k': k})

        else:
            force = ET.SubElement(forces, 'MorseBondForce')
            for term in self.ff.terms['bond']:
                ids = [str(i+1) for i in term.atomids]
                equ = str(round(term.equ, 9))
                k = str(round(term.fconst, 5))
                e_dis = str(self.ff.bond_dissociation_energies[term.atomids[0],  term.atomids[1]])
                ET.SubElement(force, 'Bond', {'class1': ids[0], 'class2': ids[1], 'length': equ, 'k': k,
                                              'e_dis': e_dis})

    def write_angles(self, forces):
        if not self.ff.cos_angle:
            force = ET.SubElement(forces, 'HarmonicAngleForce')
            for term in self.ff.terms['angle']:
                ids = [str(i+1) for i in term.atomids]
                equ = str(round(term.equ, 8))
                k = str(round(term.fconst, 6))
                ET.SubElement(force, 'Angle', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'angle': equ,
                                               'k': k})
        else:
            force = ET.SubElement(forces, 'CosineAngleForce')
            for term in self.ff.terms['angle']:
                ids = [str(i+1) for i in term.atomids]
                equ = str(round(np.degrees(term.equ), 8))
                k = str(round(term.fconst/np.sin(term.equ)**2, 6))
                ET.SubElement(force, 'Angle', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'angle': equ,
                                               'k': k})

    def write_cross_bond_bond(self, forces):
        force = ET.SubElement(forces, 'StretchStretchHarmonicForce')

        for term in self.ff.terms['cross_bond_bond']:
            ids = [str(i+1) for i in term.atomids]
            equ1 = str(round(term.equ[0], 9))
            equ2 = str(round(term.equ[1], 9))
            k = - term.fconst
            k = str(round(k, 8))
            ET.SubElement(force, 'StretchStretch', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2],
                                                    'class4': ids[3], 'length1': equ1, 'length2': equ2, 'k': k})

    def write_cross_bond_angle(self, forces):
        if self.ff.cos_angle:
            force = ET.SubElement(forces, 'StretchBendCouplingCosineForce')
        else:
            force = ET.SubElement(forces, 'StretchBendCouplingHarmonicForce')

        for term in self.ff.terms['cross_bond_angle']:
            ids = [str(i+1) for i in term.atomids]
            equ1 = str(round(term.equ[0], 8))
            equ2 = str(round(term.equ[1], 9))
            if self.ff.cos_angle:
                term.fconst /= -np.sin(term.equ[0])
            k = -term.fconst * 10
            k = str(round(k, 7))

            ET.SubElement(force, 'StretchBend', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                                 'class5': ids[4], 'length': equ1, 'angle': equ2, 'k': k})

    def write_cross_angle_angle(self, forces):
        if self.ff.cos_angle:
            force = ET.SubElement(forces, 'BendBendCosineForce')
        else:
            force = ET.SubElement(forces, 'BendBendHarmonicForce')

        for term in self.ff.terms['cross_angle_angle']:
            ids = [str(i+1) for i in term.atomids]
            equ1 = str(round(term.equ[0], 8))
            equ2 = str(round(term.equ[1], 8))
            if self.ff.cos_angle:
                term.fconst /= np.sin(term.equ[0]) * np.sin(term.equ[1])
            k = str(round(-term.fconst, 7))

            ET.SubElement(force, 'BendBend', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                              'class5': ids[4], 'class6': ids[5], 'angle1': equ1, 'angle2': equ2,
                                              'k': k})

    def write_cross_dihedral_bond(self, forces):
        force = ET.SubElement(forces, 'StretchTorsionForce')

        for term in self.ff.terms['_cross_dihed_angle']:
            ids = [str(i+1) for i in term.atomids]
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))

            ET.SubElement(force, 'StretchTorsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2],
                                                    'class4': ids[3], 'angle1': equ, 'k': k})

    def write_cross_dihedral_angle(self, forces):
        force = ET.SubElement(forces, 'BendTorsionForce')

        for term in self.ff.terms['_cross_dihed_angle']:
            ids = [str(i+1) for i in term.atomids]
            equ = str(round(term.equ, 8))
            # if self.ff.cos_angle:
            #     term.fconst /= np.sin(term.equ[0]) * np.sin(term.equ[1])
            k = str(round(term.fconst, 7))

            ET.SubElement(force, 'BendTorsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2],
                                                 'class4': ids[3], 'angle': equ, 'k': k})

    def write_improper_dihedral(self, forces):
        force = ET.SubElement(forces, 'HarmonicDihedralForce')

        for term in self.ff.terms['dihedral/improper']:
            ids = [str(i+1) for i in term.atomids]
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))

            ET.SubElement(force, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                             'angle': equ, 'k': k})

    def write_inversion_dihedral(self, forces):
        force = ET.SubElement(forces, 'InversionDihedralForce')

        for term in self.ff.terms['dihedral/inversion']:
            ids = [str(i+1) for i in term.atomids]
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))

            ET.SubElement(force, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                             'angle': equ, 'k': k})

    def write_rigid_dihedral(self, forces):
        if self.ff.cosine_dihed_period == 2:
            imp_dih_eq = '0.25*k*(1+cos(2*theta - 3.1415926535897932384626433832795))'
        elif self.ff.cosine_dihed_period == 3:
            imp_dih_eq = '1/9*k*(1+cos(3*theta))'
        elif self.ff.cosine_dihed_period == 0:
            imp_dih_eq = '0.5*k*(theta-theta0)^2'
        else:
            raise Exception('Dihedral periodicity not implemented')

        force = ET.SubElement(forces, 'RigidDihedralForce')

        for term in self.ff.terms['dihedral/rigid']:
            ids = [str(i+1) for i in term.atomids]
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))

            ET.SubElement(force, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                             'angle': equ, 'k': k})

    def write_flexible_torsion(self, forces):
        force = ET.SubElement(forces, 'RBDihedralForce')

        for term in self.ff.terms['dihedral/flexible']:
            ids = [str(i+1) for i in term.atomids]
            equ = [str(round(e, 8)) for e in term.equ]

            ET.SubElement(force, 'Torsion', {'class1': ids[0], 'class2': ids[1], 'class3': ids[2], 'class4': ids[3],
                                             'c0': equ[0], 'c1': equ[1], 'c2': equ[2], 'c3': equ[3], 'c4': equ[4],
                                             'c5': equ[5]})
