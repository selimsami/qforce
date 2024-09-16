import os
from types import SimpleNamespace
from copy import deepcopy
from shutil import copy2 as copy

from ase.io import read, write
import numpy as np
from colt import Colt
from calkeeper import check, CalculationIncompleteError

from .gaussian import Gaussian, GaussianCalculator
from .qchem import QChem, QChemCalculator
from .orca import Orca
from .crest import Crest, CrestCalculator
from .xtb import xTB, XTBGaussian, xTBCalculator
from .qm_base import scriptify, HessianOutput, ScanOutput, Calculator
from .qm_base import EnergyOutput, GradientOutput


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
                           'crest': Crest,
                           }


calculators = {
        'gaussian': GaussianCalculator,
        'xtb': xTBCalculator,
        'qchem': QChemCalculator,
        'torsiondrive': TorsiondriveCalculator,
        'crest': CrestCalculator,
        }


class QM(Colt):

    _user_input = """

# QM software to use for the hessian calculation
# and all other energy data (e.g. scan energies)
software = gaussian :: str

# software to use in preoptimization
preopt = :: str, optional

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

[addstructures]
en_struct = :: existing_file, optional
grad_struct = :: existing_file, optional
hess_struct = :: existing_file, optional
"""
    _method = ['scan_step_size', 'dihedral_scanner']

    def __init__(self, job, config):
        self.job = job
        self.pathways = job.pathways
        self.config = config
        self.logger = job.logger
        self.softwares = self._get_qm_softwares(config)
        # check hessian files and if not present write the input file
        self.method = self._register_method()

    def get_scan_software(self):
        return self.softwares['scan_software']

    def hessian_name(self, software):
        return self.pathways.hessian_filename(software, self.config.charge,
                                              self.config.multiplicity)

    def hessian_charge_name(self, software):
        return self.pathways.hessian_charge_filename(software, self.config.charge,
                                                     self.config.multiplicity)

    def charge_name(self, software):
        return self.pathways.charge_filename(software, self.config.charge,
                                             self.config.multiplicity)

    def scan_sp_name(self, software, i):
        return self.pathways.scan_sp_filename(software, self.config.charge,
                                              self.config.multiplicity, i)

    def preopt_name(self, software):
        return self.pathways.preopt_filename(software, self.config.charge,
                                             self.config.multiplicity)

    def Calculation(self, filename, required_files, *, folder=None, software=None):
        return self.job.Calculation(filename, required_files, folder=folder, software=software)

    def do_hessian_calculations(self, parent, iterator):
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
                    self.write_hessian(file, calculation.base, coords, atnums)

            hess_calcs.append(calculation)
        return hess_calcs

    def do_grad_calculations(self, parent, iterator):
        software = self.softwares['software']
        grad_calcs = []
        for i, (coords, atnums) in iterator:
            folder = parent / f'{i}_conformer'
            os.makedirs(folder, exist_ok=True)
            calculation = self.Calculation(self.hessian_name(software),
                                           software.read.gradient_files,
                                           folder=folder,
                                           software=software.name)
            if not calculation.input_exists():
                with open(calculation.inputfile, 'w') as file:
                    self.write_gradient(file, calculation.base, coords, atnums)

            grad_calcs.append(calculation)
        return grad_calcs

    def do_sp_calculations(self, parent, iterator):
        software = self.softwares['software']
        en_calcs = []
        for i, (coords, atnums) in iterator:
            folder = parent / f'{i}_conformer'
            os.makedirs(folder, exist_ok=True)
            calculation = self.Calculation(self.hessian_name(software),
                                           software.read.sp_ec_files,
                                           folder=folder,
                                           software=software.name)
            if not calculation.input_exists():
                with open(calculation.inputfile, 'w') as file:
                    self.write_sp(file, calculation.base, coords, atnums)

            en_calcs.append(calculation)
        return en_calcs

    def get_hessian(self):
        """Setup hessian files, and if present read the hessian information"""

        software = self.softwares['software']
        folder = self.pathways.getdir("hessian", create=True)

        hessians = []
        for i, (coords, atnums) in enumerate(self._read_init_file(self.pathways['init.xyz'])):
            folder = self.pathways.getdir("hessian_step", i, create=True)
            calculation = self.Calculation(self.hessian_name(software),
                                           software.read.hessian_files,
                                           folder=folder,
                                           software=software.name)
            if not calculation.input_exists():
                with open(calculation.inputfile, 'w') as file:
                    self.write_hessian(file, calculation.base, coords, atnums)
            hessians.append(calculation)

        folder = self.pathways.getdir("hessian")
        #
        for calculation in hessians:
            try:
                hessian_files = calculation.check()
            except CalculationIncompleteError:
                self.logger.exit(f"Required Hessian output file(s) not found in '{folder}' .\n"
                                 'Creating the necessary input file and exiting...\nPlease run the '
                                 'calculation and put the output files in the same directory.\n'
                                 'Necessary Hessian output files and the corresponding extensions '
                                 f"are:\n{calculation.missing_as_string()}\n\n\n")
        #
        results = []
        for calculation in hessians:
            hessian_files = calculation.check()
            results.append(self._read_hessian(hessian_files))

        # update output with charge calculation if necessary
        charge_software = self.softwares['charge_software']
        if charge_software is None:
            return results
        #
        output = results[0]

        folder = self.pathways.getdir("hessian_charge", create=True)

        calculation = self.Calculation(self.hessian_charge_name(charge_software),
                                       charge_software.read.charge_files,
                                       folder=folder,
                                       software=charge_software.name)

        if not calculation.input_exists():
            with open(calculation.inputfile, 'w') as file:
                self.write_charge(file, calculation.base, output.coords,
                                  output.atomids, charge_software)
        # if files exist
        try:
            charge_files = calculation.check()
        except CalculationIncompleteError:
            self.logger.exit(f"Required Hessian Charge output file(s) not found in '{folder}' .\n"
                             'Creating the necessary input file and exiting...\nPlease run the '
                             'calculation and put the output files in the same directory.\n'
                             'Necessary Hessian output files and the corresponding extensions '
                             f"are:\n{calculation.missing_as_string()}\n\n\n")
        #
        point_charges = charge_software.read.charges(self.config, **charge_files)
        #
        output.point_charges = output.check_type_and_shape(
                    point_charges[charge_software.config.charge_method], 'point_charges', float,
                    (output.n_atoms,))
        #
        return results

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

    def do_scan_sp_calculations(self, parent, scan_id, scan_out, atnums):
        """do scan sp calculations if necessary and update the scan out"""
        software = self.softwares['software']
        scan_software = self.softwares['scan_software']
        # check if sp should be computed
        # do_sp = not (scan_software is software)
        do_sp = True

        # setup sp calculations
        if do_sp is True:
            software = self.softwares['software']
            folder = parent
            os.makedirs(folder, exist_ok=True)
            calculations = [self.Calculation(self.scan_sp_name(software, i),
                                             software.read.gradient_files,
                                             folder=f'{folder}/step_{i}',
                                             software=software.name)
                            for i in range(scan_out.n_steps)]

            # setup files
            for i, calc in enumerate(calculations):
                if not calc.input_exists():
                    os.makedirs(calc.folder, exist_ok=True)
                    with open(calc.inputfile, 'w') as file:
                        extra_info = f', scan angle: {scan_out.angles[i]}'
                        self.write_gradient(file, calc.base, scan_out.coords[i], atnums, extra_info=extra_info)

            try:
                out_files = check(calculations)
            except CalculationIncompleteError:
                self.logger.exit(f"Required output file(s) not found in '{parent}/step_XX'.\n"
                                 'Creating the necessary input file and exiting...\n'
                                 'Please run the calculation and put the output files in the '
                                 'same directory.\nNecessary output files and the '
                                 'corresponding extensions are:\n'
                                 f"{calculations[0].missing_as_string()}")

            energies, forces, dipoles, coords = [], [], [], []
            for files in out_files:
                energy, force, dipole, atomids, coord = software.read.gradient(self.config, **files)
                energies.append(energy)
                forces.append(force)
                dipoles.append(dipole)
                coords.append(coord)

            scan_out.energies = np.array(energies, dtype=np.float64)
            scan_out.forces = np.array(forces, dtype=np.float64)
            scan_out.dipoles = np.array(dipoles, dtype=np.float64)
            scan_out.coords = np.array(coords, dtype=np.float64)
        return scan_out

    def xtb_md(self, folder, xtbinput):
        initfile = self.pathways.getfile('init.xyz')
        copy(initfile, folder / 'xtb.xyz')
        with open(folder / 'md.inp', 'w') as fh:
            fh.write("$md\n")
            for key, value in xtbinput.items():
                fh.write(f"{key} = {value}\n")
            fh.write("$end\n")

        with open(folder / 'xtb.inp', 'w') as fh:
            fh.write("xtb xtb.xyz --input md.inp --md")
        calc = self.Calculation('xtb.inp',
                                {'traj': ['xtb.trj']},
                                folder=folder,
                                software='xtb')
        return calc

    @scriptify
    def write_preopt(self, file, job_name, coords, atnums):
        software = self.softwares['preopt']
        software.write.opt(file, job_name, self.config, coords, atnums)

    @scriptify
    def write_sp(self, file, job_name, coords, atnums):
        software = self.softwares['software']
        software.write.sp(file, job_name, self.config, coords, atnums)

    @scriptify
    def write_scan_sp(self, file, job_name, coords, atnums):
        software = self.softwares['software']
        software.write.sp(file, job_name, self.config, coords, atnums)

    @scriptify
    def write_charge(self, file, job_name, coords, atnums, software=None):
        if software is None:
            software = self.softwares['charge_software']
        software.write.charges(file, job_name, self.config, coords, atnums)

    @scriptify
    def write_hessian(self, file, job_name, coords, atnums):
        software = self.softwares['software']
        software.write.hessian(file, job_name, self.config, coords, atnums)

    @scriptify
    def write_gradient(self, file, job_name, coords, atnums, extra_info=False):
        software = self.softwares['software']
        software.write.gradient(file, job_name, self.config, coords, atnums, extra_info)

    @scriptify
    def write_scan(self, file, scan_id, coords, atnums, scanned_atoms, start_angle, charge, multiplicity):
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

    def _read_energy(self, gradient_files):
        software = self.softwares['software']
        en, dipole, atids, coords = software.read.sp_ec(self.config, **gradient_files)
        return EnergyOutput(en, dipole, atids, coords)

    def _read_gradient(self, gradient_files):
        software = self.softwares['software']
        en, grad, dipole, atids, coords = software.read.gradient(self.config, **gradient_files)
        return GradientOutput(en, grad, dipole, atids, coords)

    def _read_hessian(self, hessian_files):
        software = self.softwares['software']
        qm_out = software.read.hessian(self.config, **hessian_files)
        if 'fchk_file' in hessian_files:
            fchk_file = hessian_files['fchk_file']
        else:
            fchk_file = None
        return HessianOutput(self.config.vib_scaling, fchk_file, *qm_out)

    def _read_opt(self, opt_files):
        software = self.softwares['preopt']
        return software.read.opt(self.config, **opt_files)

    def _get_unique_scan_points(self, qm_outs):
        all_angles, all_energies, all_coords, all_dipoles, chosen_point_charges, final_e = [], [], [], [], {}, 0
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

    def preopt(self):
        molecule, coords, atnums = self._read_coord_file(self.job.coord_file)
        software = self.softwares['preopt']

        if self.softwares['preopt'] is None:
            molecules = self._read_molecules(self.job.coord_file)
            self._write_xyzfile(molecules, self.pathways['init.xyz'],
                                comment=f'{self.job.name} - input geometry for hessian')
            return
        # create Preopt directory
        folder = self.pathways.getdir("preopt", create=True)
        self._write_xyzfile(molecule, self.pathways['preopt.xyz'],
                            comment=f'{self.job.name} - input geometry for preopt')
        # setup calculation
        calculation = self.Calculation(self.preopt_name(software),
                                       software.read.opt_files, folder=folder,
                                       software=software.name)
        #
        if not calculation.input_exists():
            _, coords, atnums = self._read_coord_file(self.pathways['preopt.xyz'])
            #
            with open(calculation.inputfile, 'w') as file:
                self.write_preopt(file, calculation.base, coords, atnums)
        # check_preopt_output
        try:
            preopt_files = calculation.check()
        except CalculationIncompleteError:
            self.logger.exit('Required Preopt output file(s) not found in the job directory.\n'
                             'Creating the necessary input file and exiting...\nPlease run the '
                             'calculation and put the output files in the same directory.\n')
        #
        coords = self._read_opt(preopt_files)
        molecules = []
        for coord in coords:
            mol = deepcopy(molecule)
            mol.set_positions(coord)
            molecules.append(mol)

        self._write_xyzfile(molecules, self.pathways['init.xyz'],
                            comment=f'{self.job.name} - input geometry for hessian')

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
        }

        defaults = {
                'preopt': None,
                'charge_software': None,
                'scan_software': default,
        }
        scanner = config.dihedral_scanner
        # do it twice, once load the settings, once set the defaults
        for option, default in defaults.items():
            if getattr(config, option).value is None:
                if isinstance(default, str):
                    softwares[option] = softwares[default]
                else:
                    softwares[option] = default
            else:
                softwares[option] = self._set_qm_software(getattr(config, option))

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
