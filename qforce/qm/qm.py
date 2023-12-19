import os
from types import SimpleNamespace

from ase.io import read, write
import numpy as np
from colt import Colt

from .gaussian import Gaussian
from .qchem import QChem
from .orca import Orca
from .xtb import xTB, XTBGaussian
from .qm_base import scriptify, HessianOutput, ScanOutput
from .qm_base import Calculation, CalculationIncompleteError, check
from .torsiondrive_xtb import TorsiondrivexTB


implemented_qm_software = {'gaussian': Gaussian,
                           'qchem': QChem,
                           'orca': Orca,
                           'xtb': xTB,
                           'xtb-gaussian': XTBGaussian
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
"""
    _method = ['scan_step_size']

    def __init__(self, job, config):
        self.job = job
        self.config = config
        self.softwares = self._get_qm_softwares(config)
        # check hessian files and if not present write the input file
        self.method = self._register_method()

    def get_scan_software(self):
        return self.softwares['scan_software']

    def preopt_dir(self, only=False):
        path = '0_preopt'
        if only is True:
            return path
        return os.path.join(self.job.dir, path)

    def hessian_dir(self, only=False):
        path = '1_hessian'
        if only is True:
            return path
        return os.path.join(self.job.dir, path)

    def hessian_charge_dir(self, only=False):
        path = '1_hessian_charge'
        if only is True:
            return path
        return os.path.join(self.job.dir, path)

    def fragment_dir(self, only=False):
        path = '2_fragments'
        if only is True:
            return path
        return os.path.join(self.job.dir, path)

    def _basename(self, software):
        return f'{self.job.name}_{software.hash(self.config.charge, self.config.multiplicity)}'

    def hessian_name(self, software):
        return f'{self._basename(software)}_hessian'

    def hessian_charge_name(self, software):
        return f'{self._basename(software)}_hessian_charge'

    def charge_name(self, software):
        return f'{self._basename(software)}_charge'

    def scan_sp_name(self, software, i):
        return f'{self._basename(software)}_sp_step_{i:02d}'

    def get_hessian(self):
        """Setup hessian files, and if present read the hessian information"""

        software = self.softwares['software']
        folder = self.hessian_dir()
        os.makedirs(folder, exist_ok=True)
        calculation = Calculation(f'{self.hessian_name(software)}.inp',
                                  software.read.hessian_files,
                                  folder=folder,
                                  software=software.name)
        if not calculation.input_exists():
            _, coords, atnums = self._read_coord_file(f'{self.job.dir}/init.xyz')
            with open(calculation.inputfile, 'w') as file:
                self.write_hessian(file, calculation.base, coords, atnums)
            raise CalculationIncompleteError(
                        f"Required Hessian output file(s) not found in '{folder}' .\n"
                        'Creating the necessary input file and exiting...\nPlease run the '
                        'calculation and put the output files in the same directory.\n'
                        f'Selected QM Software: "{self.config.software}"\n'
                        'Necessary Hessian output files and the corresponding extensions are:\n'
                        f"{calculation.missing_as_string()}"
                            )

        hessian_files = calculation.check()
        output = self._read_hessian(hessian_files)
        # update output with charge calculation if necessary
        charge_software = self.softwares['charge_software']
        if charge_software is None:
            return output
        #
        folder = self.hessian_charge_dir()
        os.makedirs(folder, exist_ok=True)

        calculation = Calculation(f'{self.hessian_charge_name(charge_software)}.inp',
                                  charge_software.read.charge_files,
                                  folder=folder,
                                  software=charge_software.name)

        if not calculation.input_exists():
            with open(calculation.inputfile, 'w') as file:
                self.write_charge(file, calculation.base, output.coords,
                                  output.elements, charge_software)
            raise CalculationIncompleteError(
                        f"Required Hessian Charge output file(s) not found in '{folder}' .\n"
                        'Creating the necessary input file and exiting...\nPlease run the '
                        'calculation and put the output files in the same directory.\n'
                        f'Selected QM Software: "{self.config.software}"\n'
                        'Necessary Hessian output files and the corresponding extensions are:\n'
                        f"{calculation.missing_as_string()}"
                            )
        # if files exist
        charge_files = calculation.check()
        #
        point_charges = charge_software.read.charges(self.config, **charge_files)
        #
        output.point_charges = output.check_type_and_shape(point_charges, 'point_charges', float,
                                                           (output.n_atoms,))
        #
        return output

    def read_scan(self, folder, files):
        software = self.softwares['scan_software']
        qm_outs = []
        n_scan_steps = int(np.ceil(360/self.config.scan_step_size))

        for file in files:
            if self.config.dihedral_scanner == 'relaxed_scan':
                qm_outs.append(software.read.scan(self.config, f'{folder}/{file}'))
            elif self.config.dihedral_scanner == 'torsiondrive':
                qm_outs.append(software.read.scan_torsiondrive(f'{folder}/{file}'))
        qm_out = self._get_unique_scan_points(qm_outs, n_scan_steps)

        return ScanOutput(file, n_scan_steps, *qm_out)

    def do_scan_sp_calculations(self, parent, scan_id, scan_out, atnums):
        """do scan sp calculations if necessary and update the scan out"""
        software = self.softwares['software']
        scan_software = self.softwares['scan_software']
        # check if sp should be computed
        do_sp = not (scan_software is software)
        charge_software = self.softwares['charge_software']
        charge_calc = None
        #
        if charge_software is None and do_sp is True:
            charge_software = software

        if charge_software is None and self.config.dihedral_scanner == 'torsiondrive':
            # always do charge calculations in case of torsiondrive!
            charge_software = software

        # setup charge calculations
        if charge_software is not None:
            folder = f'{parent}/charge'
            charge_calc = Calculation(f'{self.charge_name(charge_software)}.inp',
                                      charge_software.read.charge_files, folder=folder,
                                      software=charge_software.name)
            os.makedirs(folder, exist_ok=True)
            with open(charge_calc.inputfile, 'w') as file:
                self.write_charge(file, charge_calc.base, scan_out.coords[0],
                                  atnums, charge_software)
        # setup sp calculations
        if do_sp is True:
            software = self.softwares['software']
            folder = parent
            os.makedirs(folder, exist_ok=True)
            calculations = [Calculation(f'{self.scan_sp_name(software, i)}.inp',
                                        software.read.sp_files,
                                        folder=f'{folder}/step_{i}',
                                        software=software.name)
                            for i in range(scan_out.n_steps)]

            # setup files
            for i, calc in enumerate(calculations):
                if not calc.input_exists():
                    os.makedirs(calc.folder, exist_ok=True)
                    with open(calc.inputfile, 'w') as file:
                        self.write_scan_sp(file, calc.base, scan_out.coords[i], atnums)
            #
            if charge_calc is not None:
                out_files = check(calculations + [charge_calc])[:-1]
            else:
                out_files = check(calculations)
            energies = []
            # do not do che charge
            for files in out_files:
                energies.append(charge_software.read.sp(self.config, **files))
            scan_out.energies = np.array(energies, dtype=np.float64)
        # check and read charge software
        if charge_software is not None:
            charge_files = charge_calc.check()
            point_charges = charge_software.read.charges(self.config, **charge_files)
            scan_out.charges = {self.config.charge_method: list(point_charges)}

        return scan_out

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
    def write_charge(self, file, job_name, coords, atnums, software):
        software.write.charges(file, job_name, self.config, coords, atnums)

    @scriptify
    def write_hessian(self, file, job_name, coords, atnums):
        software = self.softwares['software']
        software.write.hessian(file, job_name, self.config, coords, atnums)

    @scriptify
    def write_scan(self, file, scan_id, coords, atnums, scanned_atoms, start_angle, charge,
                   multiplicity):
        '''Generate the input file for the dihedral scan.
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
        '''
        software = self.softwares['scan_software']
        if self.config.dihedral_scanner == 'relaxed_scan':
            software.write.scan(file, scan_id, self.config, coords,
                                atnums, scanned_atoms, start_angle,
                                charge, multiplicity)
        elif self.config.dihedral_scanner == 'torsiondrive':
            software.write.scan_torsiondrive(file, scan_id, self.config, coords,
                                             atnums, scanned_atoms, start_angle,
                                             charge, multiplicity)

    def _read_hessian(self, hessian_files):
        software = self.softwares['software']
        qm_out = software.read.hessian(self.config, **hessian_files)
        return HessianOutput(self.config.vib_scaling, *qm_out)

    def _read_opt(self, opt_files):
        software = self.softwares['preopt']
        return software.read.opt(self.config, **opt_files)

    def _get_unique_scan_points(self, qm_outs, n_scan_steps):
        all_angles, all_energies, all_coords, chosen_point_charges, final_e = [], [], [], {}, 0
        all_angles_rounded = []

        for n_atoms, coords, angles, energies, point_charges in qm_outs:
            angles = [round(a % 360, 3) for a in angles]

            for angle, coord, energy in zip(angles, coords, energies):
                angle_rounded = round(angle)
                if angle_rounded not in all_angles_rounded:
                    all_angles.append(angle)
                    all_energies.append(energy)
                    all_coords.append(coord)
                    all_angles_rounded.append(angle_rounded)
                else:
                    idx = all_angles_rounded.index(angle_rounded)
                    if energy < all_energies[idx]:
                        all_energies[idx] = energy
                        all_coords[idx] = coord

        if not chosen_point_charges:
            chosen_point_charges = point_charges
            final_e = energies[-1]
        elif energies[-1] < final_e:
            chosen_point_charges = point_charges

        return n_atoms, all_coords, all_angles, all_energies, chosen_point_charges

    def preopt(self):
        molecule, coords, atnums = self._read_coord_file(self.job.coord_file)
        software = self.softwares['preopt']
        if self.softwares['preopt'] is None:
            self._write_xyzfile(molecule, 'init.xyz',
                                comment=f'{self.job.name} - input geometry for hessian')
            return
        # create Preopt directory
        folder = self.preopt_dir()
        os.makedirs(folder, exist_ok=True)
        self._write_xyzfile(molecule, f'{self.preopt_dir(True)}/preopt.xyz',
                            comment=f'{self.job.name} - input geometry for preopt')
        # setup calculation
        calculation = Calculation(f"{self.job.name}_opt.inp",
                                  software.read.opt_files, folder=folder,
                                  software=software.name)
        if not calculation.input_exists():
            _, coords, atnums = self._read_coord_file(f'{folder}/preopt.xyz')
            with open(calculation.inputfile, 'w') as file:
                self.write_preopt(file, calculation.base, coords, atnums)
                raise CalculationIncompleteError(
                        'Required Preopt output file(s) not found in the job directory.\n'
                        'Creating the necessary input file and exiting...\nPlease run the '
                        'calculation and put the output files in the same directory.\n')
        # check_preopt_output
        preopt_files = calculation.check()
        coords = self._read_opt(preopt_files)
        molecule.set_positions(coords)
        self._write_xyzfile(molecule, 'init.xyz',
                            comment=f'{self.job.name} - input geometry for hessian')

    def _check_scan_sp_output(self, folder):
        software = self.softwares['software']
        scan_sp_files = {}
        folders = [folder for folder in os.listdir(folder)
                   if folder.startswith('step_')]

        for subfolder in folders:
            files = {}
            scan_sp_files[subfolder] = files
            all_files = os.listdir(f'{folder}/{subfolder}')
            for name, tails in software.read.sp_files.items():
                found_files = [file for file in all_files
                               if any(file.endswith(f'{tail}') for tail in tails)]
                n_files = len(found_files)
                if n_files == 0:
                    scan_sp_files[subfolder] = SystemExit(
                                'Required Scan SP output file(s) not found in the job directory.\n'
                                'Creating the necessary input file and exiting...\nPlease run the '
                                'calculation and put the output files in the same directory.\n')
                elif n_files > 1:
                    scan_sp_files[subfolder] = SystemExit(
                                    'There are multiple files in the job directory '
                                    'with the expected Hessian output extensions.\n'
                                    'Please remove the undesired ones.\n')

                else:
                    files[name] = f'{folder}/{subfolder}/{found_files[0]}'

        msg = ''
        for folder, error in scan_sp_files.items():
            if isinstance(error, SystemExit):
                msg += f'\n---------------------\nError in {folder}:\n---------------------\n\n'
                msg += error.code

        if msg != '':
            raise SystemExit(msg)

        return scan_sp_files

    def _read_coord_file(self, filename):
        molecule = read(filename)
        coords = molecule.get_positions()
        atnums = molecule.get_atomic_numbers()
        return molecule, coords, atnums

    def _write_xyzfile(self, molecule, filename, comment=None):
        write(f'{self.job.dir}/{filename}', molecule, plain=True,
              comment=comment)

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
        # do it twice, once load the settings, once set the defaults
        for option, default in defaults.items():
            if getattr(config, option).value is None:
                if isinstance(default, str):
                    softwares[option] = softwares[default]
                else:
                    softwares[option] = default
            else:
                softwares[option] = self._set_qm_software(getattr(config, option))

        if config.dihedral_scanner == 'torsiondrive' and softwares['scan_software'] is default:
            selection = xTB.generate_user_input().get_answers()
            softwares['scan_software'] = implemented_qm_software['xtb'](SimpleNamespace(**selection))

        if config.dihedral_scanner == 'torsiondrive' and softwares['scan_software'].has_torsiondrive is False:
            raise SystemExit("Error: TorsionDrive not supported for scan_software '{softwares['scan_software'].name}'")
        self._print_softwares(softwares)

        return softwares

    def _print_softwares(self, softwares):
        print('Selected Electronic Structure Softwares: "')
        for typ, software in softwares.items():
            if software is not None:
                print("%s: %s" % (typ, software.name))

    def _set_qm_software(self, selection):
        try:
            software = implemented_qm_software[selection.value](SimpleNamespace(**selection))
        except KeyError:
            raise KeyError(f'"{selection.value}" software is not implemented.')
        software.name = selection.value
        return software

    @staticmethod
    def _print_selected(selection, required_hessian_files):
        print(f'Selected QM Software: "{selection}"\n'
              'Necessary Hessian output files and the corresponding extensions are:')
        for req, ext in required_hessian_files.items():
            print(f'- {req}: {ext}')
        print()

    def _register_method(self):
        software = self.softwares['software']
        method_list = self._method + software._method
        method = {key: val for key, val in self.config.__dict__.items() if key in method_list}
        method.update({key: val.upper() for key, val in method.items() if isinstance(val, str)})
        method['software'] = self.config.software
        return method
