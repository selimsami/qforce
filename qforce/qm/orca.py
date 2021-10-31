from colt import Colt
import numpy as np
from ase.units import Hartree, mol, kJ
#
from .qm_base import WriteABC, ReadABC
from ..elements import ATOM_SYM


class Orca(Colt):
    _questions = """

    charge_method = CHELPG :: str :: [cm5, esp, resp]

    # QM method to be used for geometry optimisation
    opt_method = r2SCAN-3c tightSCF noautostart miniprint nopop :: str
    
    # QM method to be used for charge derivation
    c_method = HF 6-31G* noautostart miniprint nopop :: str
    
    # QM method to be used for energy calculation (e.g. dihedral scan).
    sp_method = PWPB95 D3 def2-TZVPP def2/J def2-TZVPP/C RIJCOSX tightSCF noautostart miniprint nopop :: str

    """

    _method = ['opt_method', 'c_method', 'sp_method']

    def __init__(self):
        self.required_hessian_files = {'out_file': ['out', 'log'],
                                       'hess_file': ['hess'],
                                       'pc_file': ['pc_chelpg']}
        # self.read = ReadORCA
        self.write = WriteORCA

class WriteORCA(WriteABC):
    def hessian(self, file, job_name, config, coords, atnums):
        self._write_hessian_job_setting(job_name, config, file)
        self._write_coords(atnums, coords, file)
        file.write(' *\n')

    def scan(self, file, job_name, config, coords, atnums, scanned_atoms, start_angle, charge,
             multiplicity):
        pass
        # self._write_scan_job_setting(job_name, config, file, charge, multiplicity)
        # self._write_coords(atnums, coords, file)
        # self._write_scanned_atoms(file, scanned_atoms, config.scan_step_size)

    @staticmethod
    def _write_scanned_atoms(file, scanned_atoms, step_size):
        # ORCA uses zero-based indexing
        a1, a2, a3, a4 = scanned_atoms - 1
        n_steps = int(np.ceil(360/step_size))-1
        file.write(f"\nD {a1} {a2} {a3} {a4} S {n_steps} {step_size:.2f}\n\n")

    @staticmethod
    def _write_coords(atnums, coords, file):
        for atnum, coord in zip(atnums, coords):
            elem = ATOM_SYM[atnum]
            file.write(f'{elem :>3s} {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}\n')

    @staticmethod
    def _write_hessian_job_setting(job_name, config, file):
        file.write(f"! opt freq {config.opt_method}\n")
        file.write(f"%pal nprocs  {config.n_proc} end\n")
        file.write(f"%maxcore  {config.memory}\n")
        file.write(f'%base "{job_name}_opt"\n')
        file.write(f"* xyz   {config.charge}   {config.multiplicity}\n")

    @staticmethod
    def _write_scan_job_setting(job_name, config, file, charge, multiplicity):
        file.write(f"%nprocshared={config.n_proc}\n")
        file.write(f"%mem={config.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write(f"#Opt=Modredundant {config.method} {config.dispersion} {config.basis} "
                   "pop=(CM5, ESP)\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{charge} {multiplicity}\n")

class ReadORCA(ReadABC):
    def _read_orca_hess(self, fess_file):
        with open(fess_file, 'r') as f:
            text = f.read()

        text = text[text.index('$hessian'):]
        lines = text.split('\n')[1:]
        n_atoms = int(lines[0])
        lines = lines[1:]
        hessian = np.empty((n_atoms, n_atoms))

        for _ in range(int(np.ceil(n_atoms / 5))):
            trunk = lines[: (n_atoms + 1)]
            lines = lines[(n_atoms + 1): ]
            cols = (int(col) for col in trunk[0].split())
            for row in range(n_atoms):
                row_idx, *points = trunk[1 + row].split()
                assert int(row_idx) == row
                hessian[int(row), cols] = [float(point) for point in points]
        return hessian






    def hessian(self, config, out_file, fchk_file):
        n_bonds, b_orders, lone_e, point_charges = [], [], [], []

        n_atoms, charge, multiplicity, elements, coords, hessian = self._read_fchk_file(fchk_file)

        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if "Hirshfeld charges, spin densities" in line and config.charge_method == "cm5":
                    point_charges = self._read_cm5_charges(file, n_atoms)
                elif " ESP charges:" in line and config.charge_method == "esp":
                    point_charges = self._read_cm5_charges(file, n_atoms)
                elif "N A T U R A L   B O N D   O R B I T A L" in line:
                    n_bonds, b_orders, lone_e = self._read_nbo_analysis(file, line, n_atoms)

        return (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds, b_orders,
                lone_e, point_charges)