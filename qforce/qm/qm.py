import os

from ase.io import read, write
import numpy as np
from colt import Colt

from .gaussian import Gaussian
from .qchem import QChem
from .orca import Orca
from .xtb import xTB 
from .qm_base import scriptify, HessianOutput, ScanOutput
from .torsiondrive_xtb import TorsiondrivexTB


implemented_qm_software = {'gaussian': Gaussian,
                           'qchem': QChem,
                           'orca': Orca,
                           'xtb': xTB}


class QM(Colt):
    _user_input = """
# QM software to use
software = gaussian :: str

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
        self.software = self._set_qm_software(config.software)
        self.hessian_files = self._check_hessian_output()
        self.method = self._register_method()

    def read_hessian(self):
        qm_out = self.software.read().hessian(self.config, **self.hessian_files)
        return HessianOutput(self.config.vib_scaling, *qm_out)

    def read_scan(self, files):
        qm_outs = []
        n_scan_steps = int(np.ceil(360/self.config.scan_step_size))

        for file in files:
            if self.config.dihedral_scanner == 'relaxed_scan':
                qm_outs.append(self.software.read().scan(self.config, f'{self.job.frag_dir}/{file}'))
            elif self.config.dihedral_scanner == 'xtb-torsiondrive':
                qm_outs.append(TorsiondrivexTB.read(f'{self.job.frag_dir}/{file}'))
        qm_out = self._get_unique_scan_points(qm_outs, n_scan_steps)

        return ScanOutput(file, n_scan_steps, *qm_out)

    @scriptify
    def write_hessian(self, file, coords, atnums):
        self.software.write().hessian(file, self.job.name, self.config, coords, atnums)

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
        if self.config.dihedral_scanner == 'relaxed_scan':
            self.software.write().scan(file, scan_id, self.config, coords,
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

    def _check_hessian_output(self):
        hessian_files = {}
        all_files = os.listdir(self.job.dir)

        for req, tails in self.software.required_hessian_files.items():
            files = [file for file in all_files if any(file.endswith(f'{tail}') for tail in tails)]
            n_files = len(files)

            if n_files == 0 and self.job.coord_file:
                coords, atnums = self._read_coord_file()
                file_name = f'{self.job.dir}/{self.job.name}_hessian.inp'
                print('Required Hessian output file(s) not found in the job directory.\n'
                      'Creating the necessary input file and exiting...\nPlease run the '
                      'calculation and put the output files in the same directory.\n')
                with open(file_name, 'w') as file:
                    self.write_hessian(file, coords, atnums)
                raise SystemExit
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

    def _read_coord_file(self):
        molecule = read(self.job.coord_file)
        coords = molecule.get_positions()
        atnums = molecule.get_atomic_numbers()
        write(f'{self.job.dir}/init.xyz', molecule, plain=True,
              comment=f'{self.job.name} - input geometry')
        return coords, atnums

    def _set_qm_software(self, selection):
        try:
            software = implemented_qm_software[selection]()
        except KeyError:
            raise KeyError(f'"{selection}" software is not implemented.')
        self._print_selected(selection, software.required_hessian_files)
        return software

    @staticmethod
    def _print_selected(selection, required_hessian_files):
        print(f'Selected QM Software: "{selection}"\n'
              'Necessary Hessian output files and the corresponding extensions are:')
        for req, ext in required_hessian_files.items():
            print(f'- {req}: {ext}')
        print()

    def _register_method(self):
        method_list = self._method + self.software._method
        method = {key: val for key, val in self.config.__dict__.items() if key in method_list}
        method.update({key: val.upper() for key, val in method.items() if isinstance(val, str)})
        method['software'] = self.config.software
        return method
