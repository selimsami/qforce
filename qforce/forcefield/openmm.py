import numpy as np
import xml.etree.cElementTree as ET

from .forcefield_base import ForcefieldSettings
from ..molecule.non_dihedral_terms import MorseBondTerm, CosineAngleTerm


class OpenMM(ForcefieldSettings):

    _always_on_terms = ['bond', 'angle']

    _optional_terms = {
            'urey': True,
            'cross_bond_bond': False, 
            'cross_bond_angle': False, 
            'cross_angle_angle': False, 
            '_cross_dihed_angle': False, 
            '_cross_dihed_bond': False, 
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
            'charge_flux/_bond_bond': False,
            'charge_flux/_bond_angle': False,
            'charge_flux/_angle_angle': False,
            'local_frame': True,
    }

    _term_types = {
            'bond': ('morse', ['morse', 'harmonic']),
            'angle': ('cosine', ['cosine', 'harmonic']),
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
        system = ET.Element('System', {'openmmVersion': '8.1', 'type': 'System', 'version': '1'})
        box_vec = ET.SubElement(system, 'PeriodicBoxVectors')
        ET.SubElement(box_vec, 'A', {'x': str(box[0]), 'y': '0', 'z': '0'})
        ET.SubElement(box_vec, 'B', {'x': '0', 'y': str(box[1]), 'z': '0'})
        ET.SubElement(box_vec, 'C', {'x': '0', 'y': '0', 'z': str(box[2])})

        particles = ET.SubElement(system, 'Particles')
        for mass in self.ff.masses:
            ET.SubElement(particles, 'Particle', {'mass': str(mass)})
        ET.SubElement(system, 'Constraints')

        forces = ET.SubElement(system, 'Forces')

        self.write_forces(forces)

        tree = ET.ElementTree(system)
        ET.indent(tree)
        tree.write(f'{directory}/{self.ff.mol_name}_qforce.xml')

    def write_forces(self, forces):
        n_terms = 2

        self.write_bonds(forces)
        self.write_angles(forces)

        for term in self.ff.terms:
            print(term.name, term.atomids+1, term.equ, term.fconst)

        if 'dihedral/improper' in self.ff.terms and len(self.ff.terms['dihedral/improper']) > 0:
            self.write_improper_dihedral(forces, n_terms)
            n_terms += 1
        if 'dihedral/rigid' in self.ff.terms and len(self.ff.terms['dihedral/rigid']) > 0:
            self.write_rigid_dihedral(forces, n_terms)
            n_terms += 1
        if 'dihedral/pitorsion' in self.ff.terms and len(self.ff.terms['dihedral/pitorsion']) > 0:
            self.write_pitorsion_dihedral(forces, n_terms)
            n_terms += 1
        if 'dihedral/inversion' in self.ff.terms and len(self.ff.terms['dihedral/inversion']) > 0:
            self.write_inversion_dihedral(forces, n_terms)
            n_terms += 1
        if 'cross_bond_bond' in self.ff.terms and len(self.ff.terms['cross_bond_bond']) > 0:
            self.write_cross_bond_bond(forces, n_terms)
            n_terms += 1
        if 'cross_bond_angle' in self.ff.terms and len(self.ff.terms['cross_bond_angle']) > 0:
            self.write_cross_bond_angle(forces, n_terms)
            print('cross-bond-angle terms: ', len(self.ff.terms['cross_bond_angle']))
            n_terms += 1
        if 'cross_angle_angle' in self.ff.terms and len(self.ff.terms['cross_angle_angle']) > 0:
            self.write_cross_angle_angle(forces, n_terms)
            print('cross-angle-angle terms: ', len(self.ff.terms['cross_angle_angle']))
            n_terms += 1
        if '_cross_dihed_angle' in self.ff.terms and len(self.ff.terms['_cross_dihed_angle']) > 0:
            self.write_cross_dihedral_angle(forces, n_terms)
            n_terms += 1

        # Non-bonded
        if 'local_frame' in self.ff.terms and len(self.ff.terms['local_frame']) > 0:
            self.write_multipoles(forces, n_terms)

    def write_multipoles(self, forces, n_terms):
        axis_types = {'z_then_x': '0', 'bisector': '1', 'z_then_bisector': '2', 'trisector': '3', 'z_only': '4'}

        force = ET.SubElement(forces, 'Force', {'type': 'AmoebaMultipoleForce', 'name': 'AmoebaMultipoleForce',
                                                'forceGroup': str(n_terms), 'version': '4', 'aEwald': '0',
                                                'cutoffDistance': '1', 'ewaldErrorTolerance': '.0001',
                                                'mutualInducedMaxIterations': '60', 'nonbondedMethod': '0',
                                                'mutualInducedTargetEpsilon': '1e-05', 'polarizationType': '0'})

        ET.SubElement(force, 'MultipoleParticleGridDimension', {'d0': '0', 'd1': '0', 'd2': '0'})
        ET.SubElement(force, 'ExtrapolationCoefficients', {'c0': '-.154', 'c1': '.017', 'c2': '.658', 'c3': '.474'})
        mults = ET.SubElement(force, 'MultipoleParticles')

        for i, term in enumerate(self.ff.terms['local_frame'], start=1):
            # ids = term.atomids[1:].copy()
            # if term.frame_type == 'trisector':
            #     ids[0], ids[1], ids[2] = -ids[0], -ids[1], -ids[2]
            # elif term.frame_type == 'z_then_bisector':
            #     ids[1], ids[2] = -ids[1], -ids[2]
            # elif term.frame_type == 'bisector':
            #     ids[1] = -ids[1]
            #     ids.append(ids)
            # new_ids = np.array(['' for _ in range(3)])
            # for j, id in enumerate(ids):
            #     new_ids[j] = str(id)
            # ids = new_ids

            ids = - np.ones(3, dtype=int)
            ids[:len(term.atomids)-1] = term.atomids[1:]

            dips = term.dipole.copy() / 10
            quads = term.quadrupole.copy() / 100

            part = ET.SubElement(mults, 'Particle', {'axisType': axis_types[term.frame_type],
                                                     'multipoleAtomZ': str(ids[0]), 'multipoleAtomX': str(ids[1]),
                                                     'multipoleAtomY': str(ids[2]), 'charge': str(term.q),
                                                     'damp': '0', 'polarity': '0', 'thole': '0'})

            ET.SubElement(part, 'Dipole', {'d0': str(dips[0]), 'd1': str(dips[1]), 'd2': str(dips[2])})

            ET.SubElement(part, 'Quadrupole', {'q0': str(quads[0, 0]), 'q1': str(quads[0, 1]), 'q2': str(quads[0, 2]),
                                               'q3': str(quads[1, 0]), 'q4': str(quads[1, 1]), 'q5': str(quads[1, 2]),
                                               'q6': str(quads[2, 0]), 'q7': str(quads[2, 1]), 'q8': str(quads[2, 2])})

            cov12 = ET.SubElement(part, 'Covalent12')
            for neigh in self.ff.topo.neighbors[0][term.atomids[0]]:
                ET.SubElement(cov12, 'Cv', {'v': str(neigh)})
            cov13 = ET.SubElement(part, 'Covalent13')
            for neigh in self.ff.topo.neighbors[1][term.atomids[0]]:
                ET.SubElement(cov13, 'Cv', {'v': str(neigh)})
            cov14 = ET.SubElement(part, 'Covalent14')
            for neigh in self.ff.topo.neighbors[2][term.atomids[0]]:
                ET.SubElement(cov14, 'Cv', {'v': str(neigh)})
            ET.SubElement(part, 'Covalent15')
            ET.SubElement(part, 'PolarizationCovalent11')
            ET.SubElement(part, 'PolarizationCovalent12')
            ET.SubElement(part, 'PolarizationCovalent13')
            ET.SubElement(part, 'PolarizationCovalent14')

    def write_bonds(self, forces):
        write_morse = False
        if isinstance(list(self.ff.terms['bond'])[0], MorseBondTerm):
            write_morse = True

        if write_morse:
            morse_bond_eq = 'D*(1-exp(-b*(r-r0)))^2; b = sqrt(k/(2*D))'
            bond_force = ET.SubElement(forces, 'Force', {'energy': morse_bond_eq, 'name': 'Bond', 'usesPeriodic': '0',
                                                         'type': 'CustomBondForce', 'forceGroup': '0', 'version': '3'})

            per_bond_params = ET.SubElement(bond_force, 'PerBondParameters')
            ET.SubElement(per_bond_params, 'Parameter', {'name': 'r0'})
            ET.SubElement(per_bond_params, 'Parameter', {'name': 'k'})
            ET.SubElement(per_bond_params, 'Parameter', {'name': 'D'})

            ET.SubElement(bond_force, 'GlobalParameters')
            ET.SubElement(bond_force, 'EnergyParameterDerivatives')

            bonds = ET.SubElement(bond_force, 'Bonds')
            for bond in self.ff.terms['bond']:
                ids = bond.atomids
                equ = str(round(bond.equ[0] * 0.1, 9))
                k = str(round(bond.fconst * 100, 3))
                e_dis = self.ff.bond_dissociation_energies[ids[0], ids[1]]

                ET.SubElement(bonds, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]),
                                              'param1': equ, 'param2': k, 'param3': str(e_dis)})
        else:
            bond_force = ET.SubElement(forces, 'Force', {'type': 'HarmonicBondForce', 'name': 'Bond',
                                                         'forceGroup': '0', 'usesPeriodic': '0', 'version': '2'})
            bonds = ET.SubElement(bond_force, 'Bonds')
            for bond in self.ff.terms['bond']:
                ids = bond.atomids
                equ = str(round(bond.equ * 0.1, 9))
                k = str(round(bond.fconst * 100, 3))
                ET.SubElement(bonds, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'd': equ, 'k': k})

    def write_angles(self, forces):
        write_cosine = False
        if isinstance(list(self.ff.terms['angle'])[0], CosineAngleTerm):
            write_cosine = True

        if write_cosine:
            cos_angle_eq = '0.5*k*(cos(theta)-cos(theta0))^2'
            angle_force = ET.SubElement(forces, 'Force', {'energy': cos_angle_eq, 'name': 'Angle', 'usesPeriodic': '0',
                                                          'type': 'CustomAngleForce', 'forceGroup': '1', 'version': '3'})

            per_angle_params = ET.SubElement(angle_force, 'PerAngleParameters')
            ET.SubElement(per_angle_params, 'Parameter', {'name': 'theta0'})
            ET.SubElement(per_angle_params, 'Parameter', {'name': 'k'})

            ET.SubElement(angle_force, 'GlobalParameters')
            ET.SubElement(angle_force, 'EnergyParameterDerivatives')

            angles = ET.SubElement(angle_force, 'Angles')
            for angle in self.ff.terms['angle']:
                ids = angle.atomids
                equ = str(round(angle.equ, 8))
                k = str(round(angle.fconst, 6))
                ET.SubElement(angles, 'Angle', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]),
                                                'param1': equ, 'param2': k})

        else:
            angle_force = ET.SubElement(forces, 'Force', {'type': 'HarmonicAngleForce', 'name': 'Angle',
                                                          'forceGroup': '1', 'usesPeriodic': '0', 'version': '2'})
            angles = ET.SubElement(angle_force, 'Angles')
            for angle in self.ff.terms['angle']:
                ids = angle.atomids
                equ = str(round(angle.equ, 8))
                k = str(round(angle.fconst, 6))
                ET.SubElement(angles, 'Angle', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]),
                                                'a': equ, 'k': k})

    def write_cross_bond_bond(self, forces, n_term):
        bb_cross_eq = 'max(k*(distance(p1,p2)-r1_0)*(distance(p3,p4)-r2_0), -20)'
        bb_force = ET.SubElement(forces, 'Force', {'energy': bb_cross_eq, 'name': 'BondBond', 'usesPeriodic': '0',
                                                   'type': 'CustomCompoundBondForce', 'forceGroup': str(n_term),
                                                   'particles': '4', 'version': '3'})

        per_bond_params = ET.SubElement(bb_force, 'PerBondParameters')
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'r1_0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'r2_0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'k'})

        ET.SubElement(bb_force, 'GlobalParameters')
        ET.SubElement(bb_force, 'EnergyParameterDerivatives')
        ET.SubElement(bb_force, 'Functions')

        bb = ET.SubElement(bb_force, 'Bonds')
        for cross_bb in self.ff.terms['cross_bond_bond']:
            ids = cross_bb.atomids
            equ1 = str(round(cross_bb.equ[0] * 0.1, 9))
            equ2 = str(round(cross_bb.equ[1] * 0.1, 9))

            k = cross_bb.fconst * 100
            k = str(round(k, 6))

            ET.SubElement(bb, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'param1': equ1, 'param2': equ2, 'param3': k})

    def write_cross_bond_angle(self, forces, n_term):
        if self.ff.cos_angle:
            ba_cross_eq = 'max(k*(cos(angle(p1,p2,p3))-cos(theta0))*(distance(p4,p5)-r0), -40)'
        else:
            ba_cross_eq = 'k*(angle(p1,p2,p3)-theta0)*(distance(p4,p5)-r0)'
        ba_force = ET.SubElement(forces, 'Force', {'energy': ba_cross_eq, 'name': 'BondAngle', 'usesPeriodic': '0',
                                                   'type': 'CustomCompoundBondForce', 'forceGroup': str(n_term),
                                                   'particles': '5', 'version': '3'})

        per_bond_params = ET.SubElement(ba_force, 'PerBondParameters')
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'theta0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'r0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'k'})

        ET.SubElement(ba_force, 'GlobalParameters')
        ET.SubElement(ba_force, 'EnergyParameterDerivatives')
        ET.SubElement(ba_force, 'Functions')

        ba = ET.SubElement(ba_force, 'Bonds')
        for cross_ba in self.ff.terms['cross_bond_angle']:
            ids = cross_ba.atomids
            equ1 = str(round(cross_ba.equ[0], 8))
            equ2 = str(round(cross_ba.equ[1] * 0.1, 9))
            # if self.ff.cos_angle:
            #     cross_ba.fconst /= -np.sin(cross_ba.equ[0])

            k = cross_ba.fconst * 10

            k = str(round(k, 7))

            ET.SubElement(ba, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'p5': str(ids[4]), 'param1': equ1, 'param2': equ2, 'param3': k})

    def write_cross_angle_angle(self, forces, n_term):
        if self.ff.cos_angle:
            aa_cross_eq = 'k*(cos(angle(p1,p2,p3))-cos(theta1_0))*(cos(angle(p4,p5,p6))-cos(theta2_0))'
        else:
            aa_cross_eq = 'k*(angle(p1,p2,p3)-theta1_0)*(angle(p4,p5,p6)-theta2_0)'
        ba_force = ET.SubElement(forces, 'Force', {'energy': aa_cross_eq, 'name': 'AngleAngle', 'usesPeriodic': '0',
                                                   'type': 'CustomCompoundBondForce', 'forceGroup': str(n_term),
                                                   'particles': '6', 'version': '3'})

        per_bond_params = ET.SubElement(ba_force, 'PerBondParameters')
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'theta1_0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'theta2_0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'k'})

        ET.SubElement(ba_force, 'GlobalParameters')
        ET.SubElement(ba_force, 'EnergyParameterDerivatives')
        ET.SubElement(ba_force, 'Functions')

        ba = ET.SubElement(ba_force, 'Bonds')
        for cross_aa in self.ff.terms['cross_angle_angle']:
            ids = cross_aa.atomids
            equ1 = str(round(cross_aa.equ[0], 8))
            equ2 = str(round(cross_aa.equ[1], 8))
            # if self.ff.cos_angle:
            #     cross_aa.fconst /= np.sin(cross_aa.equ[0]) * np.sin(cross_aa.equ[1])
            k = str(round(cross_aa.fconst, 7))

            ET.SubElement(ba, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'p5': str(ids[4]), 'p6': str(ids[5]), 'param1': equ1, 'param2': equ2, 'param3': k})

    def write_cross_dihedral_bond(self, forces, n_term):

        # if self.ff.cos_angle:
        #     aa_cross_eq = 'k*(1-cos(dihedral(p1,p2,p3,p4)))*(cos(angle(p1,p2,p3))-cos(theta1_0))*(cos(angle(p2,p3,p4))-cos(theta2_0))'
        # else:
        da_cross_eq = 'k*(1-cos(dihedral(p1,p2,p3,p4)))*(distance(p2,p3)-r0)'
        da_force = ET.SubElement(forces, 'Force', {'energy': da_cross_eq, 'name': 'DihedralAngle', 'usesPeriodic': '0',
                                                   'type': 'CustomCompoundBondForce', 'forceGroup': str(n_term),
                                                   'particles': '6', 'version': '3'})

        per_bond_params = ET.SubElement(da_force, 'PerBondParameters')
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'theta0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'k'})

        ET.SubElement(da_force, 'GlobalParameters')
        ET.SubElement(da_force, 'EnergyParameterDerivatives')
        ET.SubElement(da_force, 'Functions')

        ba = ET.SubElement(da_force, 'Bonds')
        for cross_da in self.ff.terms['_cross_dihed_angle']:
            ids = cross_da.atomids
            equ = str(round(cross_da.equ, 8) * 0.1)
            k = str(round(cross_da.fconst, 7) * 10)

            ET.SubElement(ba, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'param1': equ, 'param2': k})

    def write_cross_dihedral_angle(self, forces, n_term):

        # if self.ff.cos_angle:
        #     aa_cross_eq = 'k*(1-cos(dihedral(p1,p2,p3,p4)))*(cos(angle(p1,p2,p3))-cos(theta1_0))*(cos(angle(p2,p3,p4))-cos(theta2_0))'
        # else:
        da_cross_eq = 'k*(1-cos(dihedral(p1,p2,p3,p4)))*(angle(p1,p2,p3)-theta0)'
        da_force = ET.SubElement(forces, 'Force', {'energy': da_cross_eq, 'name': 'DihedralAngle', 'usesPeriodic': '0',
                                                   'type': 'CustomCompoundBondForce', 'forceGroup': str(n_term),
                                                   'particles': '4', 'version': '3'})

        per_bond_params = ET.SubElement(da_force, 'PerBondParameters')
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'theta0'})
        ET.SubElement(per_bond_params, 'Parameter', {'name': 'k'})

        ET.SubElement(da_force, 'GlobalParameters')
        ET.SubElement(da_force, 'EnergyParameterDerivatives')
        ET.SubElement(da_force, 'Functions')

        ba = ET.SubElement(da_force, 'Bonds')
        for cross_da in self.ff.terms['_cross_dihed_angle']:
            ids = cross_da.atomids
            equ = str(round(cross_da.equ, 8))
            # if self.ff.cos_angle:
            #     cross_daa.fconst /= np.sin(cross_daa.equ[0]) * np.sin(cross_daa.equ[1])
            k = str(round(cross_da.fconst, 7))

            ET.SubElement(ba, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'param1': equ, 'param2': k})


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
            imp_dih_eq = 'k*(1+cos(3*theta))'
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
