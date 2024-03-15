import numpy as np
import xml.etree.cElementTree as ET


class OpenMM:
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

        if 'cross_bond_bond' in self.ff.terms and len(self.ff.terms['cross_bond_bond']) > 0:
            self.write_cross_bond_bond(forces, n_terms)
            n_terms += 1
        if 'cross_bond_angle' in self.ff.terms and len(self.ff.terms['cross_bond_angle']) > 0:
            self.write_cross_bond_angle(forces, n_terms)
            n_terms += 1
        if 'cross_angle_angle' in self.ff.terms and len(self.ff.terms['cross_angle_angle']) > 0:
            self.write_cross_angle_angle(forces, n_terms)
            n_terms += 1

    def determine_bond_dissociation_energy(self, ids):
        elements = self.ff.elements[ids]
        symbols = self.ff.symbols[ids]

        if self.ff.topo.edge(*ids)['type'] == '1(1.0)8':
            o_idx = np.where(elements == 8)[0][0]
            neighs = self.ff.elements[self.ff.topo.neighbors[0][o_idx]]
            n_hydrogen = len(neighs == 1)
            if len(neighs) == 2 and n_hydrogen == 2:  # Water
                e_dis = 498.7
            else:
                e_dis = 440.0

        elif self.ff.topo.edge(*ids)['type'] == '1(1.0)7':
            n_idx = np.where(elements == 7)[0][0]
            neighs = self.ff.elements[self.ff.topo.neighbors[0][n_idx]]
            n_hydrogen = len(neighs == 1)
            if len(neighs) == 3 and n_hydrogen == 3:  # Ammonia
                e_dis = 435.0
            else:
                e_dis = 391.0

        else:
            raise Exception('Morse potential chosen, but dissociation energy not known for this atom pair'
                            f' ({symbols[0]}, {symbols[1]}).')
        return e_dis

    def write_bonds(self, forces):
        if not self.ff.morse:
            bond_force = ET.SubElement(forces, 'Force', {'type': 'HarmonicBondForce', 'name': 'Bond',
                                                         'forceGroup': '0', 'usesPeriodic': '0', 'version': '2'})
            bonds = ET.SubElement(bond_force, 'Bonds')
            for bond in self.ff.terms['bond']:
                ids = bond.atomids
                equ = str(round(bond.equ * 0.1, 9))
                k = str(round(bond.fconst * 100, 3))
                ET.SubElement(bonds, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'd': equ, 'k': k})
        else:
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
                equ = str(round(bond.equ * 0.1, 9))
                k = str(round(bond.fconst * 100, 3))
                e_dis = self.determine_bond_dissociation_energy(ids)
                ET.SubElement(bonds, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]),
                                              'param1': equ, 'param2': k, 'param3': str(e_dis)})

    def write_angles(self, forces):
        if not self.ff.cos_angle:
            angle_force = ET.SubElement(forces, 'Force', {'type': 'HarmonicAngleForce', 'name': 'Angle',
                                                          'forceGroup': '1', 'usesPeriodic': '0', 'version': '2'})
            angles = ET.SubElement(angle_force, 'Angles')
            for angle in self.ff.terms['angle']:
                ids = angle.atomids
                equ = str(round(angle.equ, 8))
                k = str(round(angle.fconst, 6))
                ET.SubElement(angles, 'Angle', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]),
                                                'a': equ, 'k': k})
        else:
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
                k = str(round(angle.fconst/np.sin(angle.equ)**2, 6))
                ET.SubElement(angles, 'Angle', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]),
                                                'param1': equ, 'param2': k})

    def write_cross_bond_bond(self, forces, n_term):
        bb_cross_eq = 'k*(distance(p1,p2)-r1_0)*(distance(p3,p4)-r2_0)'
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
            k = str(round(-cross_bb.fconst * 100, 6))
            ET.SubElement(bb, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'param1': equ1, 'param2': equ2, 'param3': k})

    def write_cross_bond_angle(self, forces, n_term):
        ba_cross_eq = 'k*(cos(angle(p1,p2,p3))-cos(theta0))*(distance(p4,p5)-r0)'
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
            k = str(round(-cross_ba.fconst * 10, 7))

            ET.SubElement(ba, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'p5': str(ids[4]), 'param1': equ1, 'param2': equ2, 'param3': k})

    def write_cross_angle_angle(self, forces, n_term):
        aa_cross_eq = 'k*(cos(angle(p1,p2,p3))-cos(theta1_0))*(cos(angle(p4,p5,p6))-cos(theta2_0))'
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
            k = str(round(-cross_aa.fconst * 10, 7))

            ET.SubElement(ba, 'Bond', {'p1': str(ids[0]), 'p2': str(ids[1]), 'p3': str(ids[2]), 'p4': str(ids[3]),
                                       'p5': str(ids[4]), 'p6': str(ids[5]), 'param1': equ1, 'param2': equ2, 'param3': k})
