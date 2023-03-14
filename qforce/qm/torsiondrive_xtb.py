import shutil
import os
from warnings import warn

import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.units import Hartree, mol, kJ


class TorsiondrivexTB():
    @staticmethod
    def read(log_file):
        '''Read the TorsionDrive output.

        Parameters
        ----------
        config : config
            A configparser object with all the parameters.

        Returns
        -------
        n_atoms : int
            The number of atoms in the molecule.
        coords : list
            A list of array of float. The list has the length of the number
            of steps and the array has the shape of (n_atoms, 3).
        angles : list
            A list (length: steps) of the angles that is being scanned.
        energies : list
            A list (length: steps) of the energy.
        point_charges : dict
            A dictionary with key in charge_method and the value to be a
            list of float of the size of n_atoms.
        '''
        frames = read(log_file, index=':', format='extxyz')
        n_atoms = len(frames[0])
        energy_list = []
        coord_list = []
        angle_list = []
        for frame in frames:
            coord_list.append(frame.positions)
            _, angle, _, energy = list(frame.info.keys())
            angle = float(angle[1:-2])
            angle_list.append(angle)
            energy = float(energy)
            energy_list.append(energy)

        point_charges = np.loadtxt(os.path.splitext(log_file)[0]+'.charges')
        energies = np.array(energy_list) * Hartree * mol / kJ
        return n_atoms, coord_list, angle_list, energies, {'xtb': point_charges}

    @staticmethod
    def write(config, file, dir, scan_id, coords, atnums, scanned_atoms, charge, multiplicity):
        '''Create the TorsionDrive input.

        Parameters
        ----------
        config : config
            A configparser object with all the parameters.
        file : file
            The file object to write the torsiondrive command
        dir : string
            The path to the current directory.
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
        charge : int
            The total charge of the fragment.
        multiplicity : int
            The multiplicity of the molecule.
         '''
        name = f'{dir}/{scan_id}_torsiondrive'

        if os.path.exists(name):
            warn(f'Folder {name} exists, remove the folder.')
            shutil.rmtree(name)
        os.makedirs(name)

        # Create the coordinate input file
        uhf = multiplicity - 1
        cmd = f'xTB arguments: --opt --chrg {charge} --uhf {uhf} --gfn 2 --parallel 1'

        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{dir}/{scan_id}_torsiondrive/input.xyz', mol, plain=True,
              comment=cmd)

        with open(f"{dir}/{scan_id}_torsiondrive/dihedrals.txt", 'w') as f:
            f.write('{} {} {} {}'.format(*scanned_atoms))

        # Write the torsiondrive input command
        file.write(f'cd {scan_id}_torsiondrive\n')
        file.write(f'torsiondrive-launch input.xyz dihedrals.txt '
                   f'-g {int(config.scan_step_size)} '
                   f'-e xtb --native_opt -v \n')
        file.write(f'cp scan.xyz ../{scan_id}.log\n')
        file.write(f'cp opt_tmp/gid_+000/1/charges ../{scan_id}.charges\n')
        # Ensure that at least one charge file is being copied.
        file.write(f'cp opt_tmp/gid_+180/1/charges ../{scan_id}.charges\n')
        file.write('cd ..\n')
