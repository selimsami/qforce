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
            'dihedral/pitorsion': True,
            'non_bonded': True,
            'charge_flux/bond': False,
            'charge_flux/bond_prime': False,
            'charge_flux/angle': False,
            'charge_flux/angle_prime': False,
            'charge_flux/bond_bond': False,
            'charge_flux/bond_angle': False,
            'charge_flux/angle_angle': False,
            'local_frame/bisector': False,
            'local_frame/z_then_x': False,
            'local_frame/z_only': True,
            'local_frame/z_then_bisector': True,
            'local_frame/trisector': True,
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

    def write_bonds(self, forces):
        if not self.ff.morse:
            bond_force = ET.SubElement(forces, 'HarmonicBondForce')
            for bond in self.ff.terms['bond']:
                ids = bond.atomids
                equ = str(round(bond.equ * 0.1, 9))
                k = str(round(bond.fconst * 100, 3))
                ET.SubElement(bond_force, 'Bond', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                   'length': equ, 'k': k})

        else:
            bond_force = ET.SubElement(forces, 'MorseBondForce')
            for bond in self.ff.terms['bond']:
                ids = bond.atomids
                equ = str(round(bond.equ * 0.1, 9))
                k = str(round(bond.fconst * 100, 3))
                e_dis = self.ff.bond_dissociation_energies[ids[0], ids[1]]
                ET.SubElement(bond_force, 'Bond', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                   'length': equ, 'k': k, 'e_dis': str(e_dis)})

    def write_angles(self, forces):
        if not self.ff.cos_angle:
            angle_force = ET.SubElement(forces, 'HarmonicAngleForce')
            for angle in self.ff.terms['angle']:
                ids = angle.atomids
                equ = str(round(angle.equ, 8))
                k = str(round(angle.fconst, 6))
                ET.SubElement(angle_force, 'Angle', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                     'class3': str(ids[2]+1), 'angle': equ, 'k': k})
        else:
            angle_force = ET.SubElement(forces, 'CosineAngleForce')
            for angle in self.ff.terms['angle']:
                ids = angle.atomids
                equ = str(round(np.degrees(angle.equ), 8))
                k = str(round(angle.fconst/np.sin(angle.equ)**2, 6))
                ET.SubElement(angle_force, 'Angle', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                     'class3': str(ids[2]+1), 'angle': equ, 'k': k})

    def write_cross_bond_bond(self, forces):
        bb_force = ET.SubElement(forces, 'StretchStretchHarmonicForce')

        for cross_bb in self.ff.terms['cross_bond_bond']:
            ids = cross_bb.atomids
            equ1 = str(round(cross_bb.equ[0] * 0.1, 9))
            equ2 = str(round(cross_bb.equ[1] * 0.1, 9))

            k = - cross_bb.fconst * 100
            k = str(round(k, 6))

            ET.SubElement(bb_force, 'StretchStretch', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                       'class3': str(ids[2]+1), 'class4': str(ids[3]+1),
                                                       'length1': equ1, 'length2': equ2, 'k': k})

    def write_cross_bond_angle(self, forces):
        if self.ff.cos_angle:
            ba_force = ET.SubElement(forces, 'StretchBendCouplingCosineForce')
        else:
            ba_force = ET.SubElement(forces, 'StretchBendCouplingHarmonicForce')

        for cross_ba in self.ff.terms['cross_bond_angle']:
            ids = cross_ba.atomids
            equ1 = str(round(cross_ba.equ[0], 8))
            equ2 = str(round(cross_ba.equ[1] * 0.1, 9))
            if self.ff.cos_angle:
                cross_ba.fconst /= -np.sin(cross_ba.equ[0])
            k = -cross_ba.fconst * 10
            k = str(round(k, 7))

            ET.SubElement(ba_force, 'StretchBend', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                    'class3': str(ids[2]+1), 'class4': str(ids[3]+1),
                                                    'class5': str(ids[4]+1), 'length': equ1, 'angle': equ2, 'k': k})

    def write_cross_angle_angle(self, forces):
        if self.ff.cos_angle:
            ba_force = ET.SubElement(forces, 'BendBendCosineForce')
        else:
            ba_force = ET.SubElement(forces, 'BendBendHarmonicForce')

        for cross_aa in self.ff.terms['cross_angle_angle']:
            ids = cross_aa.atomids
            equ1 = str(round(cross_aa.equ[0], 8))
            equ2 = str(round(cross_aa.equ[1], 8))
            if self.ff.cos_angle:
                cross_aa.fconst /= np.sin(cross_aa.equ[0]) * np.sin(cross_aa.equ[1])
            k = str(round(-cross_aa.fconst, 7))

            ET.SubElement(ba_force, 'BendBend', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                 'class3': str(ids[2]+1), 'class4': str(ids[3]+1),
                                                 'class5': str(ids[4]+1), 'class6': str(ids[5]+1),
                                                 'angle1': equ1, 'angle2': equ2, 'k': k})

    def write_cross_dihedral_bond(self, forces):
        db_force = ET.SubElement(forces, 'StretchTorsionForce')

        for cross_da in self.ff.terms['_cross_dihed_angle']:
            ids = cross_da.atomids
            equ = str(round(cross_da.equ, 8))
            k = str(round(cross_da.fconst, 7))

            ET.SubElement(db_force, 'StretchTorsion', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                             'class3': str(ids[2]+1), 'class4': str(ids[3]+1),
                                             'param1': equ, 'param2': k})

    def write_cross_dihedral_angle(self, forces, n_term):
        da_force = ET.SubElement(forces, 'BendTorsionForce')

        for cross_da in self.ff.terms['_cross_dihed_angle']:
            ids = cross_da.atomids
            equ = str(round(cross_da.equ, 8))
            # if self.ff.cos_angle:
            #     cross_daa.fconst /= np.sin(cross_daa.equ[0]) * np.sin(cross_daa.equ[1])
            k = str(round(cross_da.fconst, 7))

            ET.SubElement(da_force, 'BendTorsion', {'class1': str(ids[0]+1), 'class2': str(ids[1]+1),
                                                    'class3': str(ids[2]+1), 'class4': str(ids[3]+1),
                                                    'angle': equ, 'k': k})

    def write_inversion_dihedral(self, forces, n_term):
        inv_dih_eq = 'k*(cos(theta)-cos(theta0))^2'
        inv_dih_force = ET.SubElement(forces, 'Force', {'energy': inv_dih_eq, 'name': 'InversionDihedral', 'usesPeriodic': '0',
                                                        'type': 'CustomTorsionForce', 'forceGroup': str(n_term), 'version': '3',
                                                        })

        per_torsion_params = ET.SubElement(inv_dih_force, 'PerTorsionParameters')
        ET.SubElement(per_torsion_params, 'Parameter', {'name': 'theta0'})
        ET.SubElement(per_torsion_params, 'Parameter', {'name': 'k'})

        ET.SubElement(inv_dih_force, 'GlobalParameters')
        ET.SubElement(inv_dih_force, 'EnergyParameterDerivatives')

        inv = ET.SubElement(inv_dih_force, 'Torsions')
        for term in self.ff.terms['dihedral/inversion']:
            ids = term.atomids
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))

            ET.SubElement(inv, 'Torsion', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                           'param1': equ, 'param2': k})

    def write_pitorsion_dihedral(self, forces, n_term):
        # eq = """k*sin(phi-phi0)^2;
        #              phi = pointdihedral(x3+c1x, y3+c1y, z3+c1z, x3, y3, z3, x4, y4, z4, x4+c2x, y4+c2y, z4+c2z);
        #              c1x = (d14y*d24z-d14z*d24y); c1y = (d14z*d24x-d14x*d24z); c1z = (d14x*d24y-d14y*d24x);
        #              c2x = (d53y*d63z-d53z*d63y); c2y = (d53z*d63x-d53x*d63z); c2z = (d53x*d63y-d53y*d63x);
        #              d14x = x1-x4; d14y = y1-y4; d14z = z1-z4;
        #              d24x = x2-x4; d24y = y2-y4; d24z = z2-z4;
        #              d53x = x5-x3; d53y = y5-y3; d53z = z5-z3;
        #              d63x = x6-x3; d63y = y6-y3; d63z = z6-z3"""

        # eq = """k*sin(phi-phi0)^2;
        #              phi = pointdihedral(x1+c1x, y1+c1y, z1+c1z, x1, y1, z1, x4, y4, z4, x4+c2x, y4+c2y, z4+c2z);
        #              c1x = (d12y*d13z-d12z*d13y); c1y = (d12z*d13x-d12x*d13z); c1z = (d12x*d13y-d12y*d13x);
        #              c2x = (d45y*d46z-d45z*d46y); c2y = (d45z*d46x-d45x*d46z); c2z = (d45x*d46y-d45y*d46x);
        #              d12x = x2-x1; d12y = y2-y1; d12z = z2-z1;
        #              d13x = x3-x1; d13y = y3-y1; d13z = z3-z1;
        #              d45x = x5-x4; d45y = y5-y4; d45z = z5-z4;
        #              d46x = x6-x4; d46y = y6-y4; d46z = z6-z4"""

        eq = """k*sin(phi-phi0)^2;
                     phi = acos(dot);
                     dot = min(dot, 1);
                     dot = max(dot, -1);
                     dot = n1x*n2x+n1y*n2y+n1z*n2z;
                     n1x = c1x/norm1; n1y = c1y/norm1; n1z = c1z/norm1;
                     n2x = c2x/norm2; n2y = c2y/norm2; n2z = c2z/norm2;
                     norm1 = sqrt(c1x*c1x + c1y*c1y + c1z*c1z);
                     norm2 = sqrt(c2x*c2x + c2y*c2y + c2z*c2z);
                     c1x = (d12y*d13z-d12z*d13y); c1y = (d12z*d13x-d12x*d13z); c1z = (d12x*d13y-d12y*d13x); 
                     c2x = (d45y*d46z-d45z*d46y); c2y = (d45z*d46x-d45x*d46z); c2z = (d45x*d46y-d45y*d46x); 
                     d12x = x2-x1; d12y = y2-y1; d12z = z2-z1; 
                     d13x = x3-x1; d13y = y3-y1; d13z = z3-z1; 
                     d45x = x5-x4; d45y = y5-y4; d45z = z5-z4; 
                     d46x = x6-x4; d46y = y6-y4; d46z = z6-z4"""

        force = ET.SubElement(forces, 'Force', {'energy': eq, 'name': 'PiTorsionDihedral', 'usesPeriodic': '0',
                                      'type': 'CustomCompoundBondForce',  'forceGroup': str(n_term),
                                      'particles': '6', 'version': '3'})

        per_bond_params = ET.SubElement(force, 'PerBondParameters')
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'phi0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'k'})

        ET.SubElement(force, 'GlobalParameters')
        ET.SubElement(force, 'EnergyParameterDerivatives')
        ET.SubElement(force, 'Functions')

        ba = ET.SubElement(force, 'Bonds')
        for term in self.ff.terms['dihedral/pitorsion']:
            ids = term.atomids
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))
            # ET.SubElement(ba, 'Bond', {'p1': str(ids[2]), 'p2': str(ids[1]), 'p3': str(ids[0]), 'p4': str(ids[3]),
            #                            'p5': str(ids[4]), 'p6': str(ids[5]), 'param1': equ, 'param2': k})
            ET.SubElement(ba, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'p5': str(ids[4]), 'p6': str(ids[5]), 'param1': equ, 'param2': k})

    def write_improper_dihedral(self, forces, n_term):
        imp_dih_eq = '0.5*k*(theta-theta0)^2'
        # imp_dih_eq = '-0.25*k*(cos(2*theta)-cos(2*theta0))'
        imp_dih_force = ET.SubElement(forces, 'Force', {'energy': imp_dih_eq, 'name': 'ImproperDihedral', 'usesPeriodic': '0',
                                      'type': 'CustomTorsionForce', 'forceGroup': str(n_term), 'version': '3'})

        per_torsion_params = ET.SubElement(imp_dih_force, 'PerTorsionParameters')
        ET.SubElement(per_torsion_params, 'Parameter', {'name': 'theta0'})
        ET.SubElement(per_torsion_params, 'Parameter', {'name': 'k'})

        ET.SubElement(imp_dih_force, 'GlobalParameters')
        ET.SubElement(imp_dih_force, 'EnergyParameterDerivatives')

        imp = ET.SubElement(imp_dih_force, 'Torsions')
        for term in self.ff.terms['dihedral/improper']:
            ids = term.atomids
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))

            ET.SubElement(imp, 'Torsion', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                           'param1': equ, 'param2': k})

    def write_rigid_dihedral(self, forces, n_term):
        if self.ff.cosine_dihed_period == 2:
            imp_dih_eq = '0.25*k*(1+cos(2*theta - 3.1415926535897932384626433832795))'
        elif self.ff.cosine_dihed_period == 3:
            imp_dih_eq = '1/9*k*(1+cos(3*theta))'
        elif self.ff.cosine_dihed_period == 0:
            imp_dih_eq = '0.5*k*(theta-theta0)^2'
        else:
            raise Exception('Dihedral periodicity not implemented')

        imp_dih_force = ET.SubElement(forces, 'Force', {'energy': imp_dih_eq, 'name': 'RigidDihedral', 'usesPeriodic': '0',
                                      'type': 'CustomTorsionForce', 'forceGroup': str(n_term), 'version': '3'})

        per_torsion_params = ET.SubElement(imp_dih_force, 'PerTorsionParameters')
        ET.SubElement(per_torsion_params, 'Parameter', {'name': 'theta0'})
        ET.SubElement(per_torsion_params, 'Parameter', {'name': 'k'})

        ET.SubElement(imp_dih_force, 'GlobalParameters')
        ET.SubElement(imp_dih_force, 'EnergyParameterDerivatives')

        rigid = ET.SubElement(imp_dih_force, 'Torsions')
        for term in self.ff.terms['dihedral/rigid']:
            ids = term.atomids
            equ = str(round(term.equ, 8))
            k = str(round(term.fconst, 7))

            ET.SubElement(rigid, 'Torsion', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                             'param1': equ, 'param2': k})
