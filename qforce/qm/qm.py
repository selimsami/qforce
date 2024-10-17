import os
from types import SimpleNamespace

from ase.io import read, write
import numpy as np
from colt import Colt

from .gaussian import Gaussian, GaussianCalculator
from .qchem import QChem, QChemCalculator
from .orca import Orca, OrcaCalculator
from .crest import Crest, CrestCalculator
from .xtb import xTB, XTBGaussian, xTBCalculator
from .qm_base import HessianOutput, ScanOutput, Calculator
from .qm_base import EnergyOutput, GradientOutput
from ..forces import get_dihed


class TorsiondriveCalculator(Calculator):

    name = 'torsiondrive'
    _user_input = ""

    @classmethod
    def from_config(cls, config):
        return cls()

    def _commands(self, filename, basename, ncores):
        return [f'bash {filename}']


implemented_qm_software = {'gaussian': Gaussian,
                           'qchem': QChem,
                           'orca': Orca,
                           'xtb': xTB,
                           'xtb-gaussian': XTBGaussian,
                           }


calculators = {
        'gaussian': GaussianCalculator,
        'xtb': xTBCalculator,
        'orca': OrcaCalculator,
        'qchem': QChemCalculator,
        'torsiondrive': TorsiondriveCalculator,
        'crest': CrestCalculator,
        }


class QM(Colt):

    _user_input = """

# software to use in preoptimization
preopt = xtb :: str

# QM software to use for the hessian calculation
# and all other energy data (e.g. scan energies)
software = gaussian :: str

# software to use for the scan optimizations, energies still computed
# at the same level as the hessian
scan_software = :: str, optional

# software to use for the charges, default same as hessian
charge_software = :: str, optional

# To turn the QM input files into job scripts
job_script = :: literal

# Step size for the dihedral scan (360 should be divisible by this number ideally)
scan_step_size = 15 :: int

# Total charge of the system
charge = 0 :: int

# Multiplicity of the system
multiplicity = 1 :: int

# Allocated memory for the QM calculation (in MB)
memory = 4000 :: int

# Number of processors to set for the QM calculation
n_proc = 1 :: int

# Scaling of the vibrational frequency for the corresponding QM method (not implemented)
vib_scaling = 1.0 :: float

# Use the internal relaxed scan method of the QM software or the Torsiondrive method using xTB
dihedral_scanner = relaxed_scan :: str :: [relaxed_scan, torsiondrive]

"""
    _method = ['charge', 'multiplicity', 'scan_step_size', 'dihedral_scanner']

    def __init__(self, job, config):
        self.job = job
        self.pathways = job.pathways
        self.config = config
        self.logger = job.logger
        self.softwares = self._get_qm_softwares(config)
        # check hessian files and if not present write the input file
        self.method = self._register_method()

    @classmethod
    def from_config(cls, config, job):
        res = SimpleNamespace(**{key: value for key, value in config.items()})
        return cls(job, res)

    @classmethod
    def _extend_user_input(cls, questions):
        questions.generate_cases("preopt", {key: software.colt_user_input for key, software in
                                            implemented_qm_software.items()})
        questions.generate_cases("software", {key: software.colt_user_input for key, software in
                                              implemented_qm_software.items()})
        questions.generate_cases("scan_software", {key: software.colt_user_input
                                                   for key, software in
                                                   implemented_qm_software.items()},
                                 )
        questions.generate_cases("charge_software", {key: software.colt_user_input
                                                     for key, software
                                                     in implemented_qm_software.items()},
                                 )
        questions.generate_block("crest", Crest.colt_user_input)

    def get_software(self, name):
        return self.softwares[name]

    @property
    def scan_software_is_defined(self):
        return not (self.softwares['scan_software'] is self.softwares['software'])

    @property
    def charge_software_is_defined(self):
        return not (self.softwares['charge_software'] is self.softwares['software'])

    def hessian_name(self, software):
        return self.pathways.hessian_filename(software, self.config.charge,
                                              self.config.multiplicity)

    def grad_name(self, software):
        return self.pathways.grad_filename(software, self.config.charge, self.config.multiplicity)

    def charge_name(self, software):
        return self.pathways.charge_filename(software, self.config.charge,
                                             self.config.multiplicity)

    def scan_name(self, software):
        return self.pathways.scan_filename(software, self.config.charge, self.config.multiplicity)

    def scan_sp_name(self, software, i):
        return self.pathways.scan_sp_filename(software, self.config.charge,
                                              self.config.multiplicity, i)

    def sp_name(self, software, i):
        return self.pathways.sp_filename(software, self.config.charge, self.config.multiplicity)

    def opt_name(self, software):
        return self.pathways.opt_filename(software, self.config.charge, self.config.multiplicity)

    def Calculation(self, filename, required_files, *, folder=None, software=None):
        return self.job.Calculation(filename, required_files, folder=folder, software=software)

    def setup_hessian_calculation(self, folder, coords, atnums, preopt=False):
        """Setup hessian calculation"""
        # if preopt is True:
        #     software = self.softwares['preopt']
        # else:
        software = self.softwares['software']

        calculation = self.Calculation(self.hessian_name(software),
                                       software.read.hessian_files,
                                       folder=folder,
                                       software=software.name)
        if not calculation.input_exists():
            with open(calculation.inputfile, 'w') as file:
                self._write_hessian(software, file, calculation.base, coords, atnums)
        return calculation

    def setup_grad_calculation(self, folder, coords, atnums):
        software = self.softwares['software']
        calculation = self.Calculation(self.grad_name(software),
                                       software.read.gradient_files,
                                       folder=folder,
                                       software=software.name)

        if not calculation.input_exists():
            with open(calculation.inputfile, 'w') as file:
                self._write_gradient(software, file, calculation.base, coords, atnums)
        return calculation

    def setup_charge_calculation(self, folder, coords, atnums):
        """Setup charge calculation"""
        software = self.softwares['charge_software']

        calculation = self.Calculation(self.charge_name(software),
                                       software.read.charge_files,
                                       folder=folder,
                                       software=software.name)

        if not calculation.input_exists():
            with open(calculation.inputfile, 'w') as file:
                self._write_charge(software, file, calculation.base, coords, atnums)
        return calculation

    def setup_scan_calculation(self, folder, scan_hash, scanned_atomids, coords, atomids):
        software = self.softwares['scan_software']

        charge = self.config.charge
        mult = self.config.multiplicity

        if self.config.dihedral_scanner == 'relaxed_scan':
            calc = self.Calculation(f'{scan_hash}.inp',
                                    software.required_scan_files,
                                    folder=folder,
                                    software=software.name)
        elif self.config.dihedral_scanner == 'torsiondrive':
            calc = self.Calculation(f'{scan_hash}_torsiondrive.inp',
                                    software.required_scan_torsiondrive_files,
                                    folder=folder,
                                    software='torsiondrive')
        else:
            raise ValueError("scanner can only be 'torsiondrive' or 'relaxed_scan'")

        if not calc.input_exists():
            equil_angle = np.degrees(get_dihed(coords[scanned_atomids])[0])
            with open(calc.inputfile, 'w') as file:
                self.write_scan(file, scan_hash, coords, atomids, scanned_atomids+1,
                                equil_angle, charge, mult)
        return calc

    def setup_opt(self, folder, coords, atnums, preopt=False):
        if preopt is True:
            software = self.softwares['preopt']
        else:
            software = self.softwares['software']
        # setup calculation
        calculation = self.Calculation(self.preopt_name(software),
                                       software.read.opt_files,
                                       folder=folder,
                                       software=software.name)
        if not calculation.input_exists():
            with open(calculation.inputfile, 'w') as file:
                self._write_opt(software, file, calculation.base, coords, atnums)
        return calculation

    def setup_crest(self, folder, coords, atnums):
        software = self.softwares['crest']
        #
        calculation = self.Calculation(self.opt_name(software),
                                       software.read.opt_files, folder=folder,
                                       software=software.name)
        if not calculation.input_exists():
            with open(calculation.inputfile, 'w') as file:
                self._write_opt(software, file, calculation.base, coords, atnums)
        return calculation

    def setup_hessian_calculations(self, parent, iterator):
        software = self.softwares['software']
        hess_calcs = []
        for i, (coords, atnums) in iterator:
            folder = parent / f'{i}_conformer'
            os.makedirs(folder, exist_ok=True)
            calculation = self.Calculation(self.hessian_name(software),
                                           software.read.hessian_files,
                                           folder=folder,
                                           software=software.name)
            if not calculation.input_exists():
                with open(calculation.inputfile, 'w') as file:
                    self._write_hessian(software, file, calculation.base, coords, atnums)

            hess_calcs.append(calculation)
        return hess_calcs

    def setup_grad_calculations(self, parent, iterator):
        software = self.softwares['software']
        grad_calcs = []
        for i, (coords, atnums) in iterator:
            folder = parent / f'{i}_conformer'
            os.makedirs(folder, exist_ok=True)
            calculation = self.Calculation(self.grad_name(software),
                                           software.read.gradient_files,
                                           folder=folder,
                                           software=software.name)
            if not calculation.input_exists():
                with open(calculation.inputfile, 'w') as file:
                    self._write_gradient(software, file, calculation.base, coords, atnums)

            grad_calcs.append(calculation)
        return grad_calcs

    def setup_energy_calculations(self, parent, iterator):
        software = self.softwares['software']
        en_calcs = []
        for i, (coords, atnums) in iterator:
            folder = parent / f'{i}_conformer'
            os.makedirs(folder, exist_ok=True)
            calculation = self.Calculation(self.sp(software),
                                           software.read.sp_ec_files,
                                           folder=folder,
                                           software=software.name)
            if not calculation.input_exists():
                with open(calculation.inputfile, 'w') as file:
                    self._write_sp(software, file, calculation.base, coords, atnums)

            en_calcs.append(calculation)
        return en_calcs

    def setup_scan_sp_calculations(self, parent, scan_out, atnums):
        """do scan sp calculations if necessary and update the scan out"""
        software = self.softwares['software']
        calculations = []
        for i in range(scan_out.n_steps):
            folder = parent / f'{i}_conformer'
            os.makedirs(folder, exist_ok=True)
            #
            calc = self.Calculation(self.scan_name(software),
                                    software.read.gradient_files,
                                    folder=folder,
                                    software=software.name)
            #
            if not calc.input_exists():
                with open(calc.inputfile, 'w') as file:
                    extra_info = f', scan angle: {scan_out.angles[i]}'
                    self._write_gradient(software, file, calc.base, scan_out.coords[i], atnums,
                                         extra_info=extra_info)
            calculations.append(calc)
        return calculations

    def read_charges(self, charge_files):
        software = self.softwares['charge_software']
        point_charges = software.read.charges(self.config, **charge_files)
        return point_charges[software.config.charge_method]

    def read_hessian(self, hessian_files, preopt=False):
        if preopt is True:
            software = self.softwares['preopt']
        else:
            software = self.softwares['software']
        qm_out = software.read.hessian(self.config, **hessian_files)
        if 'fchk_file' in hessian_files:
            fchk_file = hessian_files['fchk_file']
        else:
            fchk_file = None
        return HessianOutput(self.config.vib_scaling, fchk_file, *qm_out)

    def read_energy(self, gradient_files):
        software = self.softwares['software']
        en, dipole, atids, coords = software.read.sp_ec(self.config, **gradient_files)
        return EnergyOutput(en, dipole, atids, coords)

    def read_gradient(self, gradient_files):
        software = self.softwares['software']
        en, grad, dipole, atids, coords = software.read.gradient(self.config, **gradient_files)
        return GradientOutput(en, grad, dipole, atids, coords)

    def read_opt(self, opt_files, software='software'):
        software = self.softwares[software]
        return software.read.opt(self.config, **opt_files)

    def read_scan_data(self, files):
        software = self.softwares['scan_software']

        if self.config.dihedral_scanner == 'relaxed_scan':
            out = software.read.scan(self.config, **files)
        elif self.config.dihedral_scanner == 'torsiondrive':
            out = software.read.scan_torsiondrive(self.config, **files)

        qm_out = self._get_unique_scan_points([out])

        files = list(files.values())
        if len(files) == 0:
            file = 'unknown'
        else:
            file = files[0]

        n_scan_steps = int(np.ceil(360/self.config.scan_step_size))
        return ScanOutput(file, n_scan_steps, *qm_out)

    def read_scan(self, folder, files):
        software = self.softwares['scan_software']
        qm_outs = []
        n_scan_steps = int(np.ceil(360/self.config.scan_step_size))

        for file in files:
            if self.config.dihedral_scanner == 'relaxed_scan':
                qm_outs.append(software.read.scan(self.config, f'{folder}/{file}'))
            elif self.config.dihedral_scanner == 'torsiondrive':
                qm_outs.append(software.read.scan_torsiondrive(f'{folder}/{file}'))
        qm_out = self._get_unique_scan_points(qm_outs)

        return ScanOutput(file, n_scan_steps, *qm_out)

    def _write_opt(self, software, file, job_name, coords, atnums):
        software.write.opt(file, job_name, self.config, coords, atnums)

    def _write_sp(self, software, file, job_name, coords, atnums):
        software.write.sp(file, job_name, self.config, coords, atnums)

    def _write_charge(self, software, file, job_name, coords, atnums):
        software.write.charges(file, job_name, self.config, coords, atnums)

    def _write_hessian(self, software, file, job_name, coords, atnums):
        software.write.hessian(file, job_name, self.config, coords, atnums)

    def _write_gradient(self, software, file, job_name, coords, atnums, extra_info=False):
        software.write.gradient(file, job_name, self.config, coords, atnums, extra_info)

    def write_scan(self, file, scan_id, coords, atnums, scanned_atoms,
                   start_angle, charge, multiplicity):
        """
        Generate the input file for the dihedral scan.
        Parameters
        ----------
        file : file
            The file handler for writing the input file. e.g. file.write('test')
        scan_id : string
            The name of the fragment.
        coords : numpy.array
            Array of the coordinates in the same of (n,3), where n is the
            number of atoms.
        atnums : list
            A list of n integers representing the atomic number, where n is
            the number of atoms.
        scanned_atoms : list
            A list of 4 integers representing the one-based atom index of the
            dihedral.
        start_angle : float
            The dihedral angle of the current conformation.
        charge : int
            The total charge of the fragment.
        multiplicity : int
            The multiplicity of the molecule.
        """

        software = self.softwares['scan_software']

        if self.config.dihedral_scanner == 'relaxed_scan':
            software.write.scan(file, scan_id, self.config, coords,
                                atnums, scanned_atoms, start_angle,
                                charge, multiplicity)
        elif self.config.dihedral_scanner == 'torsiondrive':
            software.write.scan_torsiondrive(file, scan_id, self.config, coords,
                                             atnums, scanned_atoms, start_angle,
                                             charge, multiplicity)

    def _get_unique_scan_points(self, qm_outs):
        all_angles, all_energies, all_coords, all_dipoles = [], [], [], []
        all_angles_rounded = []

        for n_atoms, coords, angles, energies, dipoles, point_charges in qm_outs:
            angles = [round(a % 360, 3) for a in angles]

            for angle, coord, energy, dipole in zip(angles, coords, energies, dipoles):
                angle_rounded = round(angle)
                if angle_rounded not in all_angles_rounded:
                    all_angles.append(angle)
                    all_energies.append(energy)
                    all_coords.append(coord)
                    all_dipoles.append(dipole)
                    all_angles_rounded.append(angle_rounded)
                else:
                    idx = all_angles_rounded.index(angle_rounded)
                    if energy < all_energies[idx]:
                        all_energies[idx] = energy
                        all_coords[idx] = coord
                        all_dipoles[idx] = dipole

        return n_atoms, all_coords, all_angles, all_energies, all_dipoles, point_charges

    def _read_coord_file(self, filename):
        molecule = read(filename)
        coords = molecule.get_positions()
        atnums = molecule.get_atomic_numbers()
        return molecule, coords, atnums

    def _read_molecules(self, filename):
        return read(filename, index=':')

    def _read_init_file(self, filename):
        molecules = read(filename, index=':')
        for molecule in molecules:
            yield molecule.get_positions(), molecule.get_atomic_numbers()

    def _write_xyzfile(self, molecule, filename, comment=None):
        write(filename, molecule, plain=True, comment=comment)

    def _get_qm_softwares(self, config):
        default = self._set_qm_software(config.software)
        #
        softwares = {
                'software': default,
                'crest': Crest(SimpleNamespace(**config.crest)),
                'preopt': self._set_qm_software(config.preopt),
        }

        defaults = {
                'charge_software': default,
                'scan_software': default,
        }
        scanner = config.dihedral_scanner
        # do it twice, once load the settings, once set the defaults
        for option, default in defaults.items():
            if getattr(config, option).value is not None:
                softwares[option] = self._set_qm_software(getattr(config, option))
            else:
                softwares[option] = default

        if scanner == 'torsiondrive' and softwares['scan_software'] is default:
            selection = xTB.generate_user_input().get_answers()
            xtbsoftware = implemented_qm_software['xtb']
            softwares['scan_software'] = xtbsoftware(SimpleNamespace(**selection))

        if scanner == 'torsiondrive' and softwares['scan_software'].has_torsiondrive is False:
            self.logger.error("TorsionDrive not supported for scan_software "
                              f"'{softwares['scan_software'].name}'")
        self.logger.info(self._get_software_text(softwares))

        return softwares

    def _get_software_text(self, softwares):
        delim = " -------------------------------------------------\n"
        txt = (delim + "    Selected Electronic Structure Softwares\n" + delim)

        for typ, software in softwares.items():
            if typ == 'crest':
                continue
            if software is not None:
                txt += "    %15s | %15s\n" % (typ, software.name)
        txt += delim
        txt += "\n\n"
        return txt

    def _set_qm_software(self, selection):
        try:
            software = implemented_qm_software[selection.value](SimpleNamespace(**selection))
        except KeyError:
            raise KeyError(f'"{selection.value}" software is not implemented.')
        software.name = selection.value
        return software

    def _register_method(self):
        software = self.softwares['software']
        method_list = self._method + software._method
        method = {key: val for key, val in self.config.__dict__.items() if key in method_list}
        method.update({key: val.upper() for key, val in method.items() if isinstance(val, str)})
        method['software'] = self.config.software
        return method
