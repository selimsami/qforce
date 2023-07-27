from colt import Colt
import numpy as np
from ase.units import Hartree, mol, kJ
#
from .qm_base import WriteABC, ReadABC
from ..elements import ATOM_SYM


class Gaussian(Colt):
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

    def __init__(self):
        self.required_hessian_files = {'out_file': ['.out', '.log'],
                                       'fchk_file': ['.fchk', '.fck']}
        self.read = ReadGaussian
        self.write = WriteGaussian


class ReadGaussian(ReadABC):
    def hessian(self, config, out_file, fchk_file):
        b_orders, point_charges = [], []

        n_atoms, charge, multiplicity, elements, coords, hessian = self._read_fchk_file(fchk_file)

        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if "Hirshfeld charges, spin densities" in line and config.charge_method == "cm5":
                    point_charges = self._read_cm5_charges(file, n_atoms)
                elif " ESP charges:" in line and config.charge_method == "esp":
                    point_charges = self._read_esp_charges(file, n_atoms)
                elif "N A T U R A L   B O N D   O R B I T A L" in line:
                    b_orders = self._read_bond_order_from_nbo_analysis(file, n_atoms)

        return n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges

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
    def hessian(self, file, job_name, config, coords, atnums):
        self._write_hessian_job_setting(job_name, config, file)
        self._write_coords(atnums, coords, file)
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

    @staticmethod
    def _write_hessian_job_setting(job_name, config, file):
        solvent_method = str(config.solvent_method or '')

        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}_hessian.chk\n")
        file.write(f"#Opt Freq {config.method} {config.dispersion} {config.basis} "
                   f"pop=(CM5, ESP, NBOREAD) {solvent_method}\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{config.charge} {config.multiplicity}\n")

    @staticmethod
    def _write_scan_job_setting(job_name, config, file, charge, multiplicity):
        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write(f"#Opt=Modredundant {config.method} {config.dispersion} {config.basis} "
                   f"pop=(CM5, ESP) {config.solvent_method}\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{charge} {multiplicity}\n")
