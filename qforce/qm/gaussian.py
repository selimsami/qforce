import numpy as np
from ase.units import Hartree, mol, kJ
#
from .qm_base import WriteABC, ReadABC, QMInterface
from ..elements import ATOM_SYM


class Gaussian(QMInterface):
    _user_input = """

    charge_method = cm5 :: str :: [cm5, esp]

    # QM method to be used
    method = PBEPBE :: str

    # Dispersion method - leave it empty to turn off
    dispersion = EmpiricalDispersion=GD3BJ :: str

    # QM basis set to be used - leave it empty to turn off
    basis = 6-31+G(D) :: str

    # Include implicit solvent for the complete parametrization
    solvent_method = :: str, optional

    """

    _method = ['method', 'dispersion', 'basis', 'solvent_method']

    def __init__(self, config):
        if config.solvent_method is None:
            config.solvent_method = ''
        super().__init__(config, ReadGaussian(config), WriteGaussian(config))

    def _settings(self):
        return {'method': self.config.method,
                'charge_method': self.config.charge_method,
                'dispersion': self.config.dispersion,
                'basis': self.config.basis,
                'solvent_method': self.config.solvent_method,
                }


class ReadGaussian(ReadABC):

    hessian_files = {'out_file': ['.out', '.log'],
                     'fchk_file': ['.fchk', '.fck']}
    opt_files = {'out_file': ['.out', '.log']}
    sp_files = {'out_file': ['.out', '.log']}
    charge_files = {'out_file': ['.out', '.log']}

    def opt(self, config, out_file):
        """read the log file"""

        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if 'Input orientation:' in line:
                    for _ in range(4):
                        next(file)
                    coords = self._get_input_orientation(file)
        # return the last coordinates in the file
        return coords

    def sp(self, config, out_file):
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if "SCF Done:" in line:
                    energy = round(float(line.split()[4]), 8)
                elif "EUMP2" in line:
                    # if mp2, read mp2 energy
                    energy = round(float(line.split()[-1].replace('D', 'E')), 8)
        return energy

    @staticmethod
    def _get_input_orientation(file):
        coords = []
        for line in file:
            if '---------------------------' in line:
                return coords
            x, y, z = line.split()[3:]
            coords.append([float(x), float(y), float(z)])
        raise ValueError("LogikError: end of file reached before coordinate ended")

    def hessian(self, config, out_file, fchk_file):
        b_orders, point_charges = [], []

        n_atoms, charge, multiplicity, elements, coords, hessian = self._read_fchk_file(fchk_file)

        charge_method = self.config.charge_method

        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if "Hirshfeld charges, spin densities" in line and charge_method == "cm5":
                    point_charges = self._read_cm5_charges(file, n_atoms)
                elif " ESP charges:" in line and charge_method == "esp":
                    point_charges = self._read_esp_charges(file, n_atoms)
                elif "N A T U R A L   B O N D   O R B I T A L" in line:
                    b_orders = self._read_bond_order_from_nbo_analysis(file, n_atoms)

        return n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges

    def charges(self, out_file):
        """read charge from file"""
        point_charges = None
        charge_method = self.config.charge_method
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if line.startswith(" NAtoms= "):
                    n_atoms = int(line.split()[1])
                    break

            for line in file:
                if "Hirshfeld charges, spin densities" in line and charge_method == "cm5":
                    point_charges = self._read_cm5_charges(file, n_atoms)
                elif " ESP charges:" in line and charge_method == "esp":
                    point_charges = self._read_esp_charges(file, n_atoms)
                # elif "N A T U R A L   B O N D   O R B I T A L" in line:
                #    b_orders = self._read_bond_order_from_nbo_analysis(file, n_atoms)
        if point_charges is None:
            raise ValueError("Charge not found")
        return point_charges

    def scan(self, config, file_name):
        n_atoms, angles, energies, coords, point_charges = None, [], [], [], {}
        with open(file_name, "r", encoding='utf-8') as file:
            for line in file:
                if line.startswith(" NAtoms= "):
                    n_atoms = int(line.split()[1])

                elif "following ModRedundant" in line:
                    step = 0
                    for line in file:
                        line = line.split()
                        if line == []:
                            break
                        elif line[0] == 'D' and line[5] == 'S':
                            step_size = float(line[7])

                elif "  Scan  " in line and "!" in line:
                    init_angle = float(line.split()[3])

                # Find coordinates
                elif "orientation:" in line:
                    coord = []
                    for _ in range(5):
                        line = file.readline()
                    while "--" not in line:
                        coord.append([float(a) for a in line.split()[3:6]])
                        line = file.readline()

                elif "SCF Done:" in line:
                    energy = round(float(line.split()[4]), 8)

                elif "EUMP2" in line:
                    # if mp2, read mp2 energy
                    energy = round(float(line.split()[-1].replace('D', 'E')), 8)

                # Get optimized energies, coords for each scan angle
                elif "-- Stationary" in line or '-- Number of steps exceeded' in line:
                    angle = init_angle + step * step_size
                    angles.append(angle)
                    energies.append(energy)
                    coords.append(coord)
                    step += 1

                    if '-- Number of steps exceeded' in line:
                        print('WARNING: An optimization step is unconverged in the file:\n'
                              f'         - {file_name}\n'
                              '           Double check to make sure it is fine.\n')

                elif "Hirshfeld charges, spin densities" in line:
                    point_charges['cm5'] = self._read_cm5_charges(file, n_atoms)
                elif " ESP charges:" in line:
                    point_charges['esp'] = self._read_esp_charges(file, n_atoms)

        energies = np.array(energies) * Hartree * mol / kJ
        return n_atoms, coords, angles, energies, point_charges

    @staticmethod
    def _read_cm5_charges(file, n_atoms):
        point_charges = []
        line = file.readline()
        for i in range(n_atoms):
            line = file.readline().split()
            point_charges.append(float(line[7]))
        return point_charges

    @staticmethod
    def _read_esp_charges(file, n_atoms):
        point_charges = []
        line = file.readline()
        for i in range(n_atoms):
            line = file.readline().split()
            point_charges.append(float(line[2]))
        return point_charges


class WriteGaussian(WriteABC):

    def opt(self, file, job_name, config, coords, atnums):
        self._write_opt_job_setting(job_name, config, file)
        self._write_coords(atnums, coords, file)
        file.write("\n\n\n")

    def sp(self, file, job_name, config, coords, atnums):
        self._write_sp_job_setting(job_name, config, file)
        self._write_coords(atnums, coords, file)
        file.write("\n\n\n")

    def charges(self, file, job_name, config, coords, atnums):
        self._write_charge_job_setting(job_name, config, file)
        self._write_coords(atnums, coords, file)
        file.write("\n\n\n")

    def hessian(self, file, job_name, config, coords, atnums):
        self._write_hessian_job_setting(job_name, config, file)
        self._write_coords(atnums, coords, file)
        self._write_bndix(file)

    @staticmethod
    def _write_bndix(file):
        file.write('\n$nbo BNDIDX $end\n\n')

    def scan(self, file, job_name, config, coords, atnums, scanned_atoms, start_angle, charge,
             multiplicity):
        self._write_scan_job_setting(job_name, config, file, charge, multiplicity)
        self._write_coords(atnums, coords, file)
        self._write_scanned_atoms(file, scanned_atoms, config.scan_step_size)

    @staticmethod
    def _write_scanned_atoms(file, scanned_atoms, step_size):
        a1, a2, a3, a4 = scanned_atoms
        n_steps = int(np.ceil(360/step_size))-1
        file.write(f"\nD {a1} {a2} {a3} {a4} S {n_steps} {step_size:.2f}\n\n")

    @staticmethod
    def _write_coords(atnums, coords, file):
        for atnum, coord in zip(atnums, coords):
            elem = ATOM_SYM[atnum]
            file.write(f'{elem :>3s} {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}\n')

    def _write_sp_job_setting(self, job_name, config, file):
        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p ")
        self.write_method(file, self.config)
        file.write(f"{self.config.solvent_method}\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{config.charge} {config.multiplicity}\n")

    def _write_charge_job_setting(self, job_name, config, file):
        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p ")
        self.write_method(file, self.config)
        file.write(f" pop=(CM5, ESP) {self.config.solvent_method}\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{config.charge} {config.multiplicity}\n")

    def _write_opt_job_setting(self, job_name, config, file):
        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p Opt")
        self.write_method(file, self.config)
        file.write(f"{self.config.solvent_method}\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{config.charge} {config.multiplicity}\n")

    def _write_hessian_job_setting(self, job_name, config, file):
        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p Opt Freq ")
        self.write_method(file, self.config)
        file.write(f"pop=(CM5, ESP, NBOREAD) {self.config.solvent_method}\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{config.charge} {config.multiplicity}\n")

    def _write_scan_job_setting(self, job_name, config, file, charge, multiplicity):
        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p Opt=Modredundant ")
        self.write_method(file, self.config)
        file.write(f"pop=(CM5, ESP) {self.config.solvent_method}\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{charge} {multiplicity}\n")

    @staticmethod
    def write_method(file, config):
        if config.method.strip().upper() == 'MP2':
            # no dispersion correction for mp2
            file.write(f" {config.method} {config.basis} density=current nosym ")
        else:
            file.write(f" {config.method} {config.dispersion} {config.basis} ")
            file.write(" density=current nosym ")

    @staticmethod
    def write_pop(file, string):
        file.write(string)
