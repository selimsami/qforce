import sys
import numpy as np
from ase.units import Hartree, mol, kJ, Bohr, Debye
#
from .qm_base import WriteABC, ReadABC, QMInterface, Calculator
from ..elements import ATOM_SYM


class QChem(QMInterface):

    _user_input = """

    charge_method = cm5 :: str :: [cm5, resp]

    # QM method to be used
    method = PBE :: str

    # Dispersion (enter "no"/"false" to turn off)
    dispersion = d3_bj :: str, optional :: [d3, d3_bj, d3_bjm, d3_zero, d3_op, empirical_grimme, false, no]

    # QM basis set to be used (enter "no"/"false" to turn off)
    basis = 6-31+G(D) :: str, optional

    # Number of maximum SCF cycles
    max_scf_cycles = 100 :: int

    # Number of maximum optimization cycles
    max_opt_cycles = 100 :: int

    # DFT Quadrature grid size
    xc_grid = 3 :: int ::

    # SCF convergence criteria for Hessian/Opt calculations
    hessian_scf_convergence = 8 :: int
    
    # SCF convergence criteria for Energy/Force calculations
    energy_scf_convergence = 6 :: int

    # Basis set linear dependence threshold
    basis_lin_dep_thresh = :: int, optional

    # Threshold for two electron integrals for Hessian/Opt calculations
    hessian_thresh = 11 :: int

    # Threshold for two electron integrals for Hessian/Opt calculations
    energy_thresh = 9 :: int

    # Number of CIS roots to ask
    cis_n_roots = :: int, optional

    # CIS singlets turned on or off
    cis_singlets = :: bool, optional

    # CIS triplets turned on or off
    cis_triplets = :: bool, optional

    # Sets CIS state for excited state optimizations and vibrational analysis
    cis_state_deriv = :: int, optional

    # Include implicit solvent for the complete parametrization
    solvent_method = :: str, optional

    """

    _method = ['method', 'dispersion', 'basis', 'cis_n_roots', 'cis_singlets', 'cis_triplets',
               'cis_state_deriv', 'solvent_method']
    name = 'qchem'
    fileending = 'in'
    has_torsiondrive = False

    def __init__(self, config):
        super().__init__(config, ReadQChem(config), WriteQChem(config))


class QChemCalculator(Calculator):

    name = 'qchem'
    _user_input = """
    qchemexe = qchem
    """

    def __init__(self, qchemexe):
        self.qchemexe = qchemexe

    @classmethod
    def from_config(cls, config):
        return cls(config['qchemexe'])

    def _commands(self, filename, basename, ncores):
        return [f'{self.qchemexe} -nt {ncores} {filename} > {basename}.log']


class ReadQChem(ReadABC):

    hessian_files = {'out_file': ['${base}.out', '${base}.log'],
                     'fchk_file': ['${base}.fchk', '${base}.fck', '${base}.in.fchk', '${base}.inp.fchk']}
    opt_files = {'out_file': ['${base}.out', '${base}.log']}
    sp_files = {'out_file': ['${base}.out', '${base}.log']}
    sp_ec_files = {'out_file': ['${base}.out', '${base}.log']}
    gradient_files = {'out_file': ['${base}.out', '${base}.log']}
    charge_files = {'out_file': ['${base}.out', '${base}.log']}
    scan_files = {'file_name': ['${base}.out', '${base}.log']}
    scan_torsiondrive_files = {'xyz': ['scan.xyz']}

    def hessian(self, config, out_file, fchk_file):
        b_orders, point_charges, dip_ders = [], [], []
        n_atoms, charge, multiplicity, elements, coords, hessian = self._read_fchk_file(fchk_file)

        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if "Charge Model 5" in line and self.config.charge_method == "cm5":
                    point_charges = self._read_cm5_charges(file)
                elif "Merz-Kollman RESP Net Atomic" in line and self.config.charge_method == "resp":
                    point_charges = self._read_resp_charges(file)
                if "N A T U R A L   B O N D   O R B I T A L" in line:
                    b_orders = self._read_bond_order_from_nbo_analysis(file, n_atoms)
                elif ' Dipole Derivatives:' in line:
                    file.readline()
                    file.readline()
                    line = file.readline()
                    while line.strip():
                        dip_ders.append([float(val) for val in line.split()])
                        line = file.readline()
                elif 'Total energy =' in line:
                    energy = float(line.split()[3])
        energy *= Hartree * mol / kJ

        return (n_atoms, charge, multiplicity, elements, coords, energy,
                hessian, b_orders, point_charges, dip_ders)

    def scan(self, config, file_name):
        n_atoms, angles, energies, coords, point_charges = None, [], [], [], {}
        with open(file_name, "r", encoding='utf-8') as file:
            angles, energies, coords, dipoles = [], [], [], []
            found_n_atoms = False

            for line in file:
                if not found_n_atoms and " NAtoms, " in line:
                    line = file.readline()
                    n_atoms = int(line.split()[0])
                    found_n_atoms = True

                elif "OPTIMIZATION CONVERGED" in line:
                    coord = []
                    for _ in range(4):
                        file.readline()
                    for n in range(n_atoms):
                        line = file.readline().split()
                        coord.append([float(c_xyz) for c_xyz in line[2:]])

                elif "Final energy is" in line:
                    energy = float(line.split()[3])

                elif 'Dipole Moment (Debye)' in line:
                    line = next(file)
                    dipole = [float(val) for val in line.split()[1::2]]

                elif "PES scan, value:" in line:
                    angles.append(float(line.split()[3]))
                    energies.append(energy)
                    coords.append(coord)
                    dipoles.append(dipole)

                elif "Charge Model 5" in line:
                    point_charges['cm5'] = self._read_cm5_charges(file)

                elif "Merz-Kollman RESP Net Atomic" in line:
                    point_charges['resp'] = self._read_resp_charges(file)

        energies = np.array(energies) * Hartree * mol / kJ
        return n_atoms, coords, angles, energies, np.array(dipoles)*Debye, point_charges

    def charges(self, config, out_file):
        """read charge from file"""
        point_charges = {}
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if "Charge Model 5" in line:
                    point_charges['cm5'] = self._read_cm5_charges(file)

                elif "Merz-Kollman RESP Net Atomic" in line:
                    point_charges['resp'] = self._read_resp_charges(file)

        if len(point_charges) == 0:
            raise ValueError("Charge not found")
        return point_charges

    def opt(self, config, out_file):

        with open(out_file, "r", encoding='utf-8') as file:
            found_n_atoms = False

            for line in file:
                if not found_n_atoms and " NAtoms, " in line:
                    line = file.readline()
                    n_atoms = int(line.split()[0])
                    found_n_atoms = True
                elif "OPTIMIZATION CONVERGED" in line:
                    coord = []
                    for _ in range(4):
                        file.readline()
                    for n in range(n_atoms):
                        line = file.readline().split()
                        coord.append([float(c_xyz) for c_xyz in line[2:]])
                    return [coord]
        raise ValueError(f"Could not parse file '{out_file}'")

    def sp(self, config, out_file):
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if 'Total energy =' in line:
                    return float(line.split()[-1]) * Hartree * mol / kJ
        raise ValueError("Could not find energy in file!")

    def sp_ec(self, config, out_file):
        energy = None
        atomids = None
        coords = None
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if 'Standard Nuclear Orientation' in line:
                    next(file)
                    next(file)
                    coords = []
                    atomids = []
                    line = next(file)
                    while '-----------' not in line:
                        _, ids, x, y, z = line.split()
                        atomids.append(ids)
                        coords.append([float(x), float(y), float(z)])
                        line = next(file)
                if 'Total energy =' in line:
                    energy = float(line.split()[-1])
                if 'Dipole Moment (Debye)' in line:
                    line = next(file)
                    dipole = [float(val) for val in line.split()[1::2]]

        if energy is None or coords is None:
            raise ValueError("Could not find energy in file!")
        energy = energy * Hartree * mol / kJ
        return energy, np.array(dipole)*Debye, atomids, np.array(coords)

    def gradient(self, config, out_file):
        energy = None
        coords = None
        atomids = None
        gradient = None
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if 'Standard Nuclear Orientation' in line:
                    next(file)
                    next(file)
                    coords = []
                    atomids = []
                    line = next(file)
                    while '-----------' not in line:
                        _, ids, x, y, z = line.split()
                        atomids.append(ids)
                        coords.append([float(x), float(y), float(z)])
                        line = next(file)
                if 'Total energy =' in line:
                    energy = float(line.split()[-1])
                if 'Dipole Moment (Debye)' in line:
                    line = next(file)
                    dipole = [float(val) for val in line.split()[1::2]]
                if 'Gradient of SCF Energy' in line:
                    gradient = []
                    line = next(file)
                    while 'Max gradient' not in line:
                        l1, l2, l3 = next(file).split(), next(file).split(), next(file).split()
                        for x, y, z in zip(l1[1:], l2[1:], l3[1:]):
                            gradient.append([float(x), float(y), float(z)])
                        line = next(file)
                    break

        if energy is None or coords is None or gradient is None:
            raise ValueError("Could not find energy in file!")
        energy = energy * Hartree * mol / kJ
        gradient = np.array(gradient) * Hartree * mol / kJ / Bohr

        return energy, gradient, np.array(dipole)*Debye, atomids, np.array(coords)

    @staticmethod
    def _read_cm5_charges(file):
        point_charges = []
        for _ in range(3):
            file.readline()
        for line in file:
            if '-----' in line:
                break
            line = line.split()
            point_charges.append(float(line[2]))
        return np.array(point_charges)

    @staticmethod
    def _read_resp_charges(file):
        point_charges = []
        for _ in range(3):
            file.readline()
        for line in file:
            if '-----' in line:
                break
            line = line.split()
            point_charges.append(float(line[2]))
        return np.array(point_charges)


class WriteQChem(WriteABC):
    sp_rem = {'jobtype': 'sp'}
    grad_rem = {'jobtype': 'force'}
    hess_opt_rem = {'jobtype': 'opt'}
    charges_rem = {'jobtype': 'sp', 'cm5': 'true', 'resp_charges': 'true'}
    hess_freq_rem = {'jobtype': 'freq', 'cm5': 'true', 'resp_charges': 'true', 'nbo': 2,
                     'iqmol_fchk': 'true', 'vibman_print': 6, 'molden_format': 'true'}
    scan_rem = {'jobtype': 'pes_scan', 'cm5': 'true', 'resp_charges': 'true', 'sym_ignore': 'true', 'symmetry': 'false'}

    def opt(self, file, job_name, settings, coords, atnums):
        self._write_molecule(file, job_name, atnums, coords, settings.charge, settings.multiplicity)
        self._write_job_setting(file, job_name, settings, self.hess_opt_rem, is_opt=True)
        file.write('\n\n\n\n\n')

    def sp(self, file, job_name, settings, coords, atnums):
        self._write_molecule(file, job_name, atnums, coords, settings.charge, settings.multiplicity)
        self._write_job_setting(file, job_name, settings, self.sp_rem)
        file.write('\n\n\n\n\n')

    def gradient(self, file, job_name, settings, coords, atnums, extra_info=False):
        self._write_molecule(file, job_name, atnums, coords, settings.charge, settings.multiplicity, extra_info)
        self._write_job_setting(file, job_name, settings, self.grad_rem)
        file.write('\n\n\n\n\n')

    def charges(self, file, job_name, settings, coords, atnums):
        self._write_molecule(file, job_name, atnums, coords, settings.charge, settings.multiplicity)
        self._write_job_setting(file, job_name, settings, self.charges_rem)
        file.write('\n\n\n\n\n')

    def hessian(self, file, job_name, settings, coords, atnums):
        self._write_molecule(file, job_name, atnums, coords, settings.charge, settings.multiplicity)
        self._write_job_setting(file, job_name, settings, self.hess_opt_rem, is_opt=True)
        file.write('\n\n@@@\n\n\n')
        file.write('$molecule\n  read\n$end\n\n')
        self._write_job_setting(file, job_name, settings, self.hess_freq_rem, is_opt=True)
        file.write('\n$nbo\n  nbo\n  bndidx\n$end\n\n')

    def scan(self, file, job_name, settings, coords, atnums, scanned_atoms, start_angle, charge,
             multiplicity):

        direct = [1, -1]
        if start_angle + settings.scan_step_size > 180:
            direct.remove(1)
        elif start_angle - settings.scan_step_size < -180:
            direct.remove(-1)
        if not direct:
            sys.exit('ERROR: Your scan step size is too large to perform a scan.\n')

        self._write_molecule(file, job_name, atnums, coords, charge, multiplicity)
        self._write_job_setting(file, job_name, settings, self.scan_rem, is_opt=True)
        self._write_scan_info(file, scanned_atoms, start_angle, direct[0]*180,
                              direct[0]*settings.scan_step_size)

        if len(direct) == 2:
            new_start = start_angle
            while new_start < 180:
                new_start += settings.scan_step_size
            new_start -= 360

            file.write('\n\n@@@\n\n\n')
            file.write('$molecule\n  read\n$end\n\n')
            self._write_job_setting(file, job_name, settings, self.scan_rem)
            self._write_scan_info(file, scanned_atoms, new_start, start_angle-0.1,
                                  settings.scan_step_size)

    @staticmethod
    def _write_scan_info(file, scanned_atoms, start_angle, final_angle, scan_step_size):
        a1, a2, a3, a4 = scanned_atoms
        file.write('\n$scan\n')
        file.write(f'  tors {a1} {a2} {a3} {a4} {start_angle:.3f} {final_angle:.3f} '
                   f'{scan_step_size:.2f}\n')
        file.write('$end\n\n')

    @staticmethod
    def _write_molecule(file, job_name, atnums, coords, charge, multiplicity, extra_info=False):
        file.write('$comment\n')
        info_string = f'  Q-Force generated input for: {job_name}'
        if extra_info:
            info_string += extra_info
        file.write(f'{info_string}\n')
        file.write('$end\n\n')

        file.write('$molecule\n')
        file.write(f'  {charge} {multiplicity}\n')
        for atnum, coord in zip(atnums, coords):
            elem = ATOM_SYM[atnum]
            file.write(f'{elem :>3s} {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}\n')
        file.write('$end\n\n')

    def _write_job_setting(self, file, job_name, settings, job_rem, is_opt=False):
        file.write('$rem\n')
        file.write(f'  method = {self.config.method}\n')
        if self.config.basis is not None:
            file.write(f'  basis = {self.config.basis}\n')
        if self.config.dispersion is not None:
            file.write(f'  dft_d = {self.config.dispersion}\n')
        if self.config.solvent_method is not None:
            file.write(f'  solvent_method = {self.config.solvent_method}\n')
        if self.config.cis_n_roots is not None:
            file.write(f'  cis_n_roots = {self.config.cis_n_roots}\n')
        if self.config.cis_singlets is not None:
            file.write(f'  cis_singlets = {self.config.cis_singlets}\n')
        if self.config.cis_triplets is not None:
            file.write(f'  cis_triplets = {self.config.cis_triplets}\n')
        if self.config.cis_state_deriv is not None:
            file.write(f'  cis_state_deriv = {self.config.cis_state_deriv}\n')
        for key, val in job_rem.items():
            file.write(f'  {key} = {val}\n')
        file.write(f'  mem_total = {settings.memory}\n')
        file.write(f'  geom_opt_max_cycles = {self.config.max_opt_cycles}\n')
        file.write(f'  max_scf_cycles = {self.config.max_scf_cycles}\n')
        file.write(f'  xc_grid = {self.config.xc_grid}\n')
        if is_opt:
            file.write(f'  scf_convergence = {self.config.hessian_scf_convergence}\n')
            file.write(f'  thresh = {self.config.hessian_thresh}\n')
        else:
            file.write(f'  scf_convergence = {self.config.energy_scf_convergence}\n')
            file.write(f'  thresh = {self.config.energy_thresh}\n')
        if self.config.basis_lin_dep_thresh is not None:
            file.write(f'  basis_lin_dep_thresh = {self.config.basis_lin_dep_thresh}\n')
        file.write('$end\n')
