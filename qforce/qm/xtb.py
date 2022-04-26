import os.path

from colt import Colt
import numpy as np
from ase.units import Hartree, mol, kJ, Bohr
from ase.io import read, write
from ase import Atoms
#
from .qm_base import WriteABC, ReadABC


class xTB(Colt):
    _user_input = """
    # xTB only allows Mulliken charge.
    charge_method = xtb :: str ::

    # Extra command line passed to the xtb executable
    xtb_command = --gfn 2 :: str ::

    """

    _method = []

    def __init__(self):
        self.required_hessian_files = {'hess_file': ['hessian'],
                                       'pc_file': ['charges'],
                                       'wbo_file': ['wbo'],
                                       'coord_file': ['xtbopt.xyz'], }
        self.read = ReadxTB
        self.write = WritexTB


class WritexTB(WriteABC):
    def hessian(self, file, job_name, config, coords, atnums):
        """ Write the input file for hessian and charge calculation.

        Parameters
        ----------
        file : file
            The file object to write the command line.
        job_name : string
            The name of the job.
        config : config
            A configparser object with all the parameters.
        coords : array
            A coordinates array of shape (N,3), where N is the number of atoms.
        atnums : list
            A list of atom elements represented as atomic number.
        """
        name = file.name
        base, filename = os.path.split(name)
        # Given that the xTB input has to be given in the command line.
        # We create the xTB command template here.
        cmd = f'xtb {job_name}_input.xyz --ohess --chrg {config.charge} ' \
              f'--uhf {config.multiplicity - 1} ' \
              f'--namespace {job_name}_hessian --parallel {config.n_proc} ' \
              f'{config.xtb_command}'
        # Write the hessian.inp which is the command line input
        file.write(cmd)
        # Write the coordinates, which is the standard xyz file.
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

    def scan(self, file, job_name, config, coords, atnums, scanned_atoms,
             start_angle, charge, multiplicity):
        """ Write the input file for the dihedral scan and charge calculation.

        Parameters
        ----------
        file : file
            The file object to write the input.
        job_name : string
            The name of the job.
        config : config
            A configparser object with all the parameters.
        coords : array
            A coordinates array of shape (N,3), where N is the number of atoms.
        atnums : list
            A list of atom elements represented as atomic number.
        scanned_atoms : list
            A list of 4 integer (one-based index) of the dihedral
        start_angle : float
            The starting angle in degree.
        charge : int
            The total charge of the molecule.
        multiplicity : int
            The multiplicity of the molecule.
        """
        # Write the xTB input file
        name = file.name
        base, filename = os.path.split(name)

        # Generate the command line
        cmd = f'xtb {job_name}_input.xyz --opt --chrg {config.charge} ' \
              f'--uhf {config.multiplicity - 1} ' \
              f'--namespace {job_name} --parallel {config.n_proc} ' \
              f'--input {job_name}.dat {config.xtb_command}'

        # Create the scan input file
        a1, a2, a3, a4 = np.array(scanned_atoms)
        step_num = int(360 // config.scan_step_size)
        end_angle = start_angle + 360 - config.scan_step_size

        with open(f'{base}/{job_name}.dat', 'w') as f:
            f.write('$constrain\n')
            f.write('  force constant=15.0\n')
            f.write('$scan\n')
            f.write(f'  dihedral: {a1},{a2},{a3},{a4},{start_angle:.2f}; '
                    f'{start_angle:.2f},{end_angle:.2f},{step_num}\n')
            f.write('$end\n')

        # Write the coordiante file in the xyz file format
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

        file.write(cmd)


class ReadxTB(ReadABC):
    @staticmethod
    def _read_xtb_hess(hess_file, n_atoms):
        """ Read the hessian matrix.

        For xTB jobs, the file contain the hessian information is stored
        with the extension of .hessian.

        Parameters
        ----------
        hess_file : string
            The name of the hess file.
        n_atoms : int
            The number of atoms in the molecule.

        Returns
        -------
        out_hessian : array
            An array of float of the size of ((n_atoms*3)**2+n_atoms*3)/2),
            which is the lower triangle of the hessian matrix. Unit： kJ/mol
        """
        with open(hess_file, 'r') as f:
            text = f.read()

        text = text[text.index('$hessian'):]
        lines = text.split('\n')[1:]
        # number of atoms * 3 for the x, y and z axis
        n_atoms *= 3
        hessian = np.empty((n_atoms, n_atoms))

        for i in range(n_atoms):
            # Find the row
            trunk = lines[: int(np.ceil(n_atoms / 5))]
            lines = lines[int(np.ceil(n_atoms / 5)):]
            # Split into columns
            row = []
            for line in trunk:
                row.extend(line.split())
            hessian[i, :] = row

        # Output the lower triangle of the hessian matrix to match the
        # format adopted by Gaussian and Qchem.
        out_hessian = []
        for i in range(len(hessian)):
            for j in range(i + 1):
                hes = (hessian[i, j] + hessian[j, i]) / 2
                out_hessian.append(hes)
        return np.array(out_hessian) * Hartree * mol / kJ / Bohr ** 2

    @staticmethod
    def _read_xtb_charge(pc_file):
        """ Read the point charge file.

        For xTB jobs, there is no other option to compute the point charge and
        the user should note that this point charge is not suitable for MM
        calculations. The xTB point chareg is stored in a file with the
        extension of .charges.

        Parameters
        ----------
        pc_file : string
            The name of the point charge file.

        Returns
        -------
        n_atoms : int
            The number of atoms in the molecule.
        point_charges : float
            A list of float of the size of n_atoms.
        """
        point_charges = np.loadtxt(pc_file)
        return len(point_charges), point_charges

    @staticmethod
    def _read_xtb_xyz(coord_file):
        """ Read the optimised coordinate xyz file.

        For xTB jobs, the optimised geometry will be stored as a file
        with the extension .xtbopt.xyz.

        Parameters
        ----------
        coord_file : string
            The name of the coordinates file.

        Returns
        -------
        n_atoms : int
            The number of atoms in the molecule.
        elements : array
            A np.array of integer of the atomic number of the atoms.
        coords : array
            An array of float of the shape (n_atoms, 3).
        """
        mol = read(coord_file)
        n_atoms = len(mol)
        elements = np.array([atom.number for atom in mol])
        coords = mol.positions
        return n_atoms, elements, coords

    @staticmethod
    def _read_xtb_scan_log(coord_file):
        """ Read the xTB scan log file.

        For xTB jobs, the optimised geometry will be stored as in the xyz
        format with the energy in the comment line with the extension log.

        Parameters
        ----------
        coord_file : string
            The name of the .xtbscan.log file.

        Returns
        -------
        elements : array
            A np.array of integer of the atomic number of the atoms.
        energy_list : list
            A list of energy for each conformations.
        coord_list : array
            An array of float of the shape (n_atoms, 3).
        """
        frames = read(coord_file, index=':', format='extxyz')
        energy_list = []
        coord_list = []
        for frame in frames:
            coord_list.append(frame.positions)
            energy_list.append(float(list(frame.info.keys())[1]))
        elements = [atom.number for atom in frame]
        return elements, energy_list, coord_list

    @staticmethod
    def _read_xtb_wbo_analysis(out_file, elements):
        """ Read the wbo analysis from xTB.

        Parameters
        ----------
        out_file : string
            The name of the orca output file.
        n_atoms : int
            The number of atoms in the molecule.

        Returns
        -------
        b_orders : list
            A list (length: n_atoms) of list (length: n_atoms) of float.
            representing the bond order between each atom pair.

        """
        n_atoms = len(elements)

        b_orders = [[0 for _ in range(n_atoms)] for _ in range(n_atoms)]

        file = np.loadtxt(out_file)
        for x, y, bo in file:
            b_orders[int(x) - 1][int(y) - 1] = bo
            b_orders[int(y) - 1][int(x) - 1] = bo

        return b_orders

    @staticmethod
    def _read_xtb_input_angle(in_file):
        '''Read the angles from the xTB input file.

        Given that the xTB will not output the scan dihedral angles, we would
        read in the input file to got the dihedral angles of each conformation.

        Parameters
        ----------
        in_file : string
            The name of the xTB input file.

        Returns
        -------
        angles : list
            A list of float of angles for each conformation.
        '''
        with open(in_file, 'r') as f:
            text = f.read().strip()
        angle_line = text[text.index('$scan'):].split('\n')[1]
        _, _, angle_range = angle_line.split()
        start, end, step_num = angle_range.split(',')
        return np.linspace(float(start), float(end), int(step_num))

    def hessian(self, config, hess_file, pc_file, coord_file, wbo_file):
        """ Extract hessian information from all the relevant files.

        Parameters
        ----------
        config : config
            A configparser object with all the parameters.
        hess_file : string
            File name of the xTB hess file for hessian information.
        pc_file : string
            File name of the xTB point charge file.
        coord_file : string
            File name of the xTB geometry optimised coordinate file.

        Returns
        -------
        n_atoms : int
            The number of atoms in the molecule.
        charge : int
            The total charge of the molecule.
        multiplicity : int
            The multiplicity of the molecule.
        elements : array
            A np.array of integer of the atomic number of the atoms.
        coords : array
            An array of float of the shape (n_atoms, 3).
        out_hessian : array
            An array of float of the size of ((n_atoms*3)**2+n_atoms*3)/2),
            which is the lower triangle of the hessian matrix. Unit： kJ/mol
        b_orders : list
            A list (length: n_atoms) of list (length: n_atoms) of float.
            representing the bond order between each atom pair.
        point_charges : float
            A list of float of the size of n_atoms.
        """
        n_atoms, point_charges = self._read_xtb_charge(pc_file)
        n_atoms, elements, coords = self._read_xtb_xyz(coord_file)
        charge = config.charge
        multiplicity = config.multiplicity
        b_orders = self._read_xtb_wbo_analysis(wbo_file, elements)
        hessian = self._read_xtb_hess(hess_file, n_atoms)
        return (n_atoms, charge, multiplicity, elements, coords, hessian,
                b_orders, point_charges)

    def scan(self, config, file_name):
        """ Read data from the scan file.

        Parameters
        ----------
        config : config
            A configparser object with all the parameters.
        file_name : string
            File name of the ORCA log file.

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
        """
        file_name_base, ext = os.path.splitext(file_name)
        base, name = os.path.split(file_name_base)
        name = name.split('.')[0]

        point_charges = {}
        n_atoms, charges = self._read_xtb_charge(
                '{}.charges'.format(os.path.join(base, name)))
        point_charges["xtb"] = charges
        elements, energies, coords = self._read_xtb_scan_log(
            '{}.xtbscan.log'.format(os.path.join(base, name)))
        angles = self._read_xtb_input_angle(
            '{}.dat'.format(os.path.join(base, name)))

        energies = np.array(energies) * Hartree * mol / kJ
        return n_atoms, coords, angles, energies, point_charges
