from colt import Colt
import numpy as np
from ase.units import Hartree, mol, kJ
#
from .qm_base import WriteABC, ReadABC
from ..elements import ATOM_SYM


class QChem(Colt):
    _questions = """

    charge_method = cm5 :: str :: [cm5, resp]

    # QM method to be used
    method = PBE :: str

    # Dispersion (enter "no"/"false" to turn off)
    dispersion = d3_bj :: str, optional :: [d3, d3_bj, d3_bjm, d3_zero, d3_op, empirical_grimme]

    # QM basis set to be used (enter "no"/"false" to turn off)
    basis = 6-31+G(D) :: str, optional

    # Number of maximum SCF cycles
    max_scf_cycles = 100 :: int

    # Number of maximum optimization cycles
    max_opt_cycles = 100 :: int

    # Number of CIS roots to ask
    cis_n_roots = :: int, optional

    # CIS singlets turned on or off
    cis_singlets = :: bool, optional

    # CIS triplets turned on or off
    cis_triplets = :: bool, optional

    # Sets CIS state for excited state optimizations and vibrational analysis
    cis_state_deriv = :: int, optional

    """

    _method = ['method', 'dispersion', 'basis', 'cis_n_roots', 'cis_singlets', 'cis_triplets',
               'cis_state_deriv']

    def __init__(self):
        self.required_hessian_files = {'out_file': ['out', 'log'], 'fchk_file': ['fchk', 'fck']}
        self.read = ReadQChem
        self.write = WriteQChem


class ReadQChem(ReadABC):
    def hessian(self, config, out_file, fchk_file):
        n_atoms, charge, multiplicity, elements, coords, hessian = self._read_fchk_file(fchk_file)
        n_bonds, b_orders, lone_e = self._read_nbo_analysis(out_file, n_atoms)
        point_charges = self._read_point_charges(out_file, n_atoms, config.charge_method)

        return (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds, b_orders,
                lone_e, point_charges)

    def scan(self, file_name, n_scan_steps):
        n_atoms, angles, energies, coords = None, [], [], []
        with open(file_name, "r", encoding='utf-8') as gaussout:
            angles, energies, coords = [], [], []
            found_n_atoms = False

            for line in gaussout:
                if not found_n_atoms and " NAtoms, " in line:
                    line = gaussout.readline()
                    n_atoms = int(line.split()[0])

                elif "OPTIMIZATION CONVERGED" in line:
                    coord = []
                    for _ in range(4):
                        gaussout.readline()
                    for n in range(n_atoms):
                        line = gaussout.readline().split()
                        coord.append([float(c_xyz) for c_xyz in line[2:]])
                    coords.append(coord)

                elif "Final energy is" in line:
                    energy = float(line.split()[3])

                elif "PES scan, value:" in line:
                    angles.append(float(line.split()[3]))
                    energies.append(energy)

        energies = np.array(energies) * Hartree * mol / kJ
        return n_atoms, n_scan_steps, coords, angles, energies

    @staticmethod
    def _read_point_charges(out_file, n_atoms, charge_method):
        point_charges = []
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if charge_method == "cm5" and "Charge Model 5" in line:
                    point_charges = []  # reset here because gaussian prints it multiple times
                    for _ in range(3):
                        file.readline()
                    for i in range(n_atoms):
                        line = file.readline().split()
                        point_charges.append(float(line[2]))
                elif charge_method == "resp" and "Merz-Kollman RESP Net Atomic Charges" in line:
                    point_charges = []
                    for _ in range(3):
                        file.readline()
                    for i in range(n_atoms):
                        line = file.readline().split()
                        point_charges.append(float(line[2]))
        return point_charges


class WriteQChem(WriteABC):
    hess_opt_rem = {'jobtype': 'opt'}
    hess_freq_rem = {'jobtype': 'freq', 'cm5': 'true', 'resp_charges': 'true', 'nbo': 2,
                     'iqmol_fchk': 'true'}
    scan_rem = {'jobtype': 'pes_scan'}

    def hessian(self, file, job_name, config, coords, atnums):
        self._write_molecule(file, job_name, atnums, coords, config.charge, config.multiplicity)
        self._write_job_setting(file, job_name, config, self.hess_opt_rem)
        file.write('\n\n@@@\n\n\n')
        file.write('$molecule\n  read\n$end\n\n')
        self._write_job_setting(file, job_name, config, self.hess_freq_rem)
        file.write('\n$nbo\n  nbo\n  bndidx\n$end\n\n')

    def scan(self, file, job_name, config, coords, atnums, scanned_atoms, start_angle, charge,
             multiplicity):
        self._write_molecule(file, job_name, atnums, coords, charge, multiplicity)
        self._write_job_setting(file, job_name, config, self.scan_rem)
        self._write_scanned_atoms(file, scanned_atoms, start_angle, config.scan_step_size)

    @staticmethod
    def _write_scanned_atoms(file, scanned_atoms, start_angle, scan_step_size):
        a1, a2, a3, a4 = scanned_atoms
        file.write('\n$scan\n')
        file.write(f'  tors {a1} {a2} {a3} {a4} {start_angle:.3f} {start_angle-0.01:.3f} '
                   f'{scan_step_size:.2f}\n')
        file.write('$end\n\n')

    @staticmethod
    def _write_molecule(file, job_name, atnums, coords, charge, multiplicity):
        file.write('$comment\n')
        file.write(f'  Q-Force generated input for: {job_name}\n')
        file.write('$end\n\n')

        file.write('$molecule\n')
        file.write(f'  {charge} {multiplicity}\n')
        for atnum, coord in zip(atnums, coords):
            elem = ATOM_SYM[atnum]
            file.write(f'{elem :>3s} {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}\n')
        file.write('$end\n\n')

    @staticmethod
    def _write_job_setting(file, job_name, config, job_rem):
        file.write('$rem\n')
        file.write(f'  method = {config.method}\n')
        if config.basis is not None:
            file.write(f'  basis = {config.basis}\n')
        if config.dispersion is not None:
            file.write(f'  dft_d = {config.dispersion}\n')
        if config.cis_n_roots is not None:
            file.write(f'  cis_n_roots = {config.cis_n_roots}\n')
        if config.cis_singlets is not None:
            file.write(f'  cis_singlets = {config.cis_singlets}\n')
        if config.cis_triplets is not None:
            file.write(f'  cis_triplets = {config.cis_triplets}\n')
        if config.cis_state_deriv is not None:
            file.write(f'  cis_state_deriv = {config.cis_state_deriv}\n')
        for key, val in job_rem.items():
            file.write(f'  {key} = {val}\n')
        file.write(f'  mem_total = {config.memory}\n')
        file.write(f'  geom_opt_max_cycles = {config.max_opt_cycles}\n')
        file.write(f'  max_scf_cycles = {config.max_scf_cycles}\n')
        file.write('$end\n')
