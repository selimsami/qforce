from colt import Colt
import numpy as np
from ase.units import Hartree, mol, kJ
#
from .qm_base import WriteABC, ReadABC
from ..elements import ATOM_SYM


class Orca(Colt):
    _questions = """

    charge_method = cm5 :: str :: [cm5, esp, resp]

    # QM method to be used for geometry optimisation
    opt_method = r2SCAN-3c tightSCF noautostart miniprint nopop :: str
    
    # QM method to be used for charge derivation
    c_method = HF 6-31G* noautostart miniprint nopop :: str
    
    # QM method to be used for energy calculation (e.g. dihedral scan).
    sp_method = PWPB95 D3 def2-TZVPP def2/J def2-TZVPP/C RIJCOSX tightSCF noautostart miniprint nopop :: str

    """

    _method = ['opt_method', 'c_method', 'sp_method']

    def __init__(self):
        self.required_hessian_files = {'out_file': ['out', 'log'], 'fchk_file': ['fchk', 'fck']}
        # self.read = ReadORCA
        self.write = WriteORCA

class WriteORCA(WriteABC):
    def hessian(self, file, job_name, config, coords, atnums):
        self._write_hessian_job_setting(job_name, config, file)
        self._write_coords(atnums, coords, file)
        file.write('\n *\n')

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
        file.write(
            f"! opt freq {config.opt_method}")  # "pop=(CM5, ESP, NBOREAD)\n\n"
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
