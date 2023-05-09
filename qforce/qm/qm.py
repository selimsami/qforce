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
scan_step_size = 15.0 :: float

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
dihedral_scanner = relaxed_scan :: str :: [relaxed_scan, xtb-torsiondrive]
"""
    _method = ['scan_step_size']

    def __init__(self, job, config):
        self.job = job
        self.config = config
        self.softwares = self._get_qm_softwares(config)
        # check hessian files and if not present write the input file
        self.method = self._register_method()

    def _read_opt(self, opt_files):
        software = self.softwares['preopt']
        return software.read.opt(self.config, **opt_files)

    def get_hessian(self):
        """Setup hessian files, and if present read the hessian information"""
        hessian_files = self._check_hessian_output()
        output = self._read_hessian(hessian_files)
        charge_software = self.softwares['charge_software']
        if charge_software is None:
            return output

        os.makedirs(f'{self.job.dir}/hessian_charge', exist_ok=True)

        with open(f'{self.job.dir}/hessian_charge/{self.job.name}_hessian_charge.inp', 'w') as file: 
            filename = self.write_charge(file, output.coords, output.elements, charge_software)
        #
        charge_files = self._check_hessian_charge_output()
        point_charges = charge_software.read.charges(self.config, **charge_files)
        #
        output.point_charges = output.check_type_and_shape(point_charges, 'point_charges', float,
                                                           (output.n_atoms,))
        #
        return output

    def _read_hessian(self, hessian_files):
        software = self.softwares['software']
        qm_out = software.read.hessian(self.config, **hessian_files)
        return HessianOutput(self.config.vib_scaling, *qm_out)

    def read_scan(self, files):
        software = self.softwares['scan_software']
        qm_outs = []
        n_scan_steps = int(np.ceil(360/self.config.scan_step_size))

        for file in files:
            if self.config.dihedral_scanner == 'relaxed_scan':
                qm_outs.append(software.read.scan(self.config, f'{self.job.frag_dir}/{file}'))
            elif self.config.dihedral_scanner == 'xtb-torsiondrive':
                qm_outs.append(TorsiondrivexTB.read(f'{self.job.frag_dir}/{file}'))
        qm_out = self._get_unique_scan_points(qm_outs, n_scan_steps)

        return ScanOutput(file, n_scan_steps, *qm_out)


    def do_scan_sp_calculations(self, scan_id, scan_out, atnums):
        """do scan sp calculations if necessary and update the scan out"""
        software = self.softwares['software']
        scan_software = self.softwares['scan_software']
        # check if sp should be computed 
        do_sp = not (scan_software is software)
        charge_software = self.softwares['charge_software']
        #
        if charge_software is None and do_sp is True:
            charge_software = software
            
        # setup charge calculations
        if charge_software is not None:
            folder = f'{self.job.frag_dir}/{scan_id}/charge'
            os.makedirs(folder, exist_ok=True)
            with open(f'{folder}/{self.job.name}_{scan_id}.inp', 'w') as file: 
                filename = self.write_charge(file, scan_out.coords[0], atnums, charge_software)
        # setup sp calculations
        if do_sp is True:
            folder = f'{self.job.frag_dir}/{scan_id}'
            os.makedirs(folder, exist_ok=True)
            #
            for i in range(scan_out.n_steps):
                os.makedirs(f'{folder}/step_{i}', exist_ok=True)
                with open(f'{folder}/step_{i}/{self.job.name}.inp', 'w') as file:
                    self.write_sp(file, scan_out.coords[i], atnums)
            #
            scan_sp_files = self._check_scan_sp_output(f'{self.job.frag_dir}/{scan_id}')
            energies = []
            for i in range(scan_out.n_steps):
                files = scan_sp_files[f'step_{i}']
                energies.append(charge_software.read.sp(self.config, **files))
            scan_out.energies = np.array(energies, dtype=np.float64)
        # check and read charge software
        if charge_software is not None:
            charge_files = self._check_hessian_charge_output(f'{self.job.frag_dir}/{scan_id}/charge')
            point_charges = charge_software.read.charges(self.config, **charge_files)
            scan_out.charges = {self.config.charge_method: list(point_charges)}

        return scan_out

    @scriptify
    def write_preopt(self, file, coords, atnums):
        software = self.softwares['preopt']
        software.write.opt(file, self.job.name, self.config, coords, atnums)

    @scriptify
    def write_sp(self, file, coords, atnums):
        software = self.softwares['software']
        software.write.sp(file, self.job.name, self.config, coords, atnums)

    @scriptify
    def write_charge(self, file, coords, atnums, software):
        software.write.charges(file, self.job.name, self.config, coords, atnums)

    @scriptify
    def write_hessian(self, file, coords, atnums):
        software = self.softwares['software']
        software.write.hessian(file, self.job.name, self.config, coords, atnums)

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
        elif self.config.dihedral_scanner == 'xtb-torsiondrive':
            TorsiondrivexTB.write(self.config, file, self.job.frag_dir,
                                  scan_id, coords, atnums, scanned_atoms,
                                  charge, multiplicity)

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

    @property
    def preopt_dir(self):
        return os.path.join(self.job.dir, 'preopt')

    def preopt(self):
        molecule, coords, atnums = self._read_coord_file(self.job.coord_file)
        software = self.softwares['preopt']
        if self.softwares['preopt'] is None:
            self._write_xyzfile(molecule, 'init.xyz',
                                comment=f'{self.job.name} - input geometry for hessian')
            return

        # create Preopt directory
        preopt_dir = self.preopt_dir
        os.makedirs(preopt_dir, exist_ok=True)
        self._write_xyzfile(molecule, 'preopt/preopt.xyz',
                            comment=f'{self.job.name} - input geometry for preopt')
        # check_preopt_output
        preopt_files = self._check_preopt_output()
        coords = self._read_opt(preopt_files)
        molecule.set_positions(coords)
        self._write_xyzfile(molecule, 'init.xyz',
                            comment=f'{self.job.name} - input geometry for hessian')

    def _check_preopt_output(self):
        software = self.softwares['preopt']
        if software is None:
            raise ValueError("preopt cannot be None")
        preopt_files = {}
        folder = self.preopt_dir
        all_files = os.listdir(folder)

        for req, tails in software.required_preopt_files.items():
            files = [file for file in all_files if any(file.endswith(f'{tail}') for tail in tails)]
            n_files = len(files)

            if n_files == 0 and self.job.coord_file:
                _, coords, atnums = self._read_coord_file(f'{folder}/preopt.xyz')
                file_name = f'{self.job.dir}/{self.job.name}_hessian.inp'
                print('Required Preopt output file(s) not found in the job directory.\n'
                      'Creating the necessary input file and exiting...\nPlease run the '
                      'calculation and put the output files in the same directory.\n')
                file_name = f'{self.preopt_dir}/{self.job.name}_preopt.inp'
                with open(file_name, 'w') as file:
                    self.write_preopt(file, coords, atnums)
                raise SystemExit
            elif n_files == 0:
                print('Required Preopt output file(s) not found in the job directory\n'
                      'and no coordinate file was provided to create input files.\nExiting...\n')
                raise SystemExit
            elif n_files > 1:
                print('There are multiple files in the job directory with the expected Preopt'
                      ' output extensions.\nPlease remove the undesired ones.\n')
                raise SystemExit
            else:
                preopt_files[req] = f'{folder}/{files[0]}'
        return preopt_files

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
                found_files = [file for file in all_files if any(file.endswith(f'{tail}') for tail in tails)]
                n_files = len(found_files)
                if n_files == 0:
                    scan_sp_files[subfolder] = SystemExit('Required Scan SP output file(s) not found in the job directory.\n'
                                'Creating the necessary input file and exiting...\nPlease run the '
                                'calculation and put the output files in the same directory.\n')
                elif n_files > 1: 
                    scan_sp_files[subfolder] = SystemExit('There are multiple files in the job directory with the expected Hessian'
                                     ' output extensions.\nPlease remove the undesired ones.\n')

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

    def _check_hessian_charge_output(self, folder=None):
        software = self.softwares['charge_software']
        charge_files = {}
        if folder is None:
            all_files = os.listdir(f'{self.job.dir}/hessian_charge')
        else:
            all_files = os.listdir(folder)

        for name, tails in software.read.charge_files.items():
            files = [file for file in all_files if any(file.endswith(f'{tail}') for tail in tails)]
            n_files = len(files)
            if n_files == 0:
                raise SystemExit('Required Hessian Charge output file(s) not found in the job directory.\n'
                            'Creating the necessary input file and exiting...\nPlease run the '
                            'calculation and put the output files in the same directory.\n')
            elif n_files > 1: 
                raise SystemExit('There are multiple files in the job directory with the expected Hessian'
                                 ' output extensions.\nPlease remove the undesired ones.\n')

            else:
                charge_files[name] = f'{self.job.dir}/hessian_charge/{files[0]}'
        return charge_files

    def _check_hessian_output(self):
        software = self.softwares['software']
        hessian_files = {}
        all_files = os.listdir(self.job.dir)

        for req, tails in software.required_hessian_files.items():
            files = [file for file in all_files if any(file.endswith(f'{tail}') for tail in tails)]
            n_files = len(files)

            if n_files == 0 and self.job.coord_file:
                _, coords, atnums = self._read_coord_file(f'{self.job.dir}/init.xyz')
                file_name = f'{self.job.dir}/{self.job.name}_hessian.inp'
                with open(file_name, 'w') as file:
                    self.write_hessian(file, coords, atnums)
                raise SystemExit('Required Hessian output file(s) not found in the job directory.\n'
                      'Creating the necessary input file and exiting...\nPlease run the '
                      'calculation and put the output files in the same directory.\n')
            elif n_files == 0:
                print('Required Hessian output file(s) not found in the job directory\n'
                      'and no coordinate file was provided to create input files.\nExiting...\n')
                raise SystemExit
            elif n_files > 1:
                print('There are multiple files in the job directory with the expected Hessian'
                      ' output extensions.\nPlease remove the undesired ones.\n')
                raise SystemExit
            else:
                hessian_files[req] = f'{self.job.dir}/{files[0]}'
        return hessian_files

    def _read_coord_file(self, filename):
        molecule = read(filename)
        coords = molecule.get_positions()
        atnums = molecule.get_atomic_numbers()
        return molecule, coords, atnums

    def _write_xyzfile(self, molecule, filename, comment=None):
        print(f'{self.job.dir}/{filename}')
        write(f'{self.job.dir}/{filename}', molecule, plain=True,
              comment=comment)

    def _get_qm_softwares(self, config):
        default = self._set_qm_software(config.software)
        self._print_selected(config.software.value, default.required_hessian_files)
        #
        softwares = {}
        defaults = {
                'preopt': None,
                'software': default,
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
