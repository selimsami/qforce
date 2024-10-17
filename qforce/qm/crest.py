import os.path
from ase.io import read, write
from ase import Atoms
import numpy as np
#
from .qm_base import WriteABC, ReadABC, QMInterface, Calculator


class CrestCalculator(Calculator):

    name = 'crest'
    _user_input = ""

    @classmethod
    def from_config(cls, config):
        return cls()

    def _commands(self, filename, basename, ncores):
        return [f'bash {filename} > {basename}.out']


class Crest(QMInterface):

    _user_input = """
    # Extra command line passed to the xtb executable
    xtb_command = --gfn2 :: str 

    """

    name = 'crest'
    has_torsiondrive = False

    _method = ['xtb_command']

    def __init__(self, config):
        super().__init__(config, ReadCrest(config), WriteCrest(config))


class ReadCrest(ReadABC):

    hessian_files = {}

    opt_files = {'coord_file': ['crest_conformers.xyz'], 'wbo_file': ['wbo']}

    sp_files = {}

    charge_files = {}

    scan_files = {}

    def opt(self, config, coord_file, wbo_file):
        coords = self._read_xyzs(coord_file)
        n_atoms = len(coords[0])
        bond_orders = self._read_crest_wbo_analysis(wbo_file, n_atoms)
        return coords, np.array(bond_orders)

    def sp(self, config, sp_file):
        raise NotImplementedError

    def charges(self, config, pc_file):
        raise NotImplementedError

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
        energy: float
            Energy of the system at the hessian opt structure.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def _read_xyzs(coord_file):
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
        mols = read(coord_file, index=':')
        return [mol.get_positions() for mol in mols]

    @staticmethod
    def _read_crest_wbo_analysis(out_file, n_atoms):
        """ Read the wbo analysis from CREST.

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

        b_orders = [[0 for _ in range(n_atoms)] for _ in range(n_atoms)]

        file = np.loadtxt(out_file)
        for x, y, bo in file:
            b_orders[int(x) - 1][int(y) - 1] = bo
            b_orders[int(y) - 1][int(x) - 1] = bo

        return b_orders


class WriteCrest(WriteABC):

    def opt(self, file, job_name, settings, coords, atnums):
        """ Write the input file for optimization

        Parameters
        ----------
        file : file
            The file object to write the command line.
        job_name : string
            The name of the job.
        settings: SimpleNamespace
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
        cmd = f'crest {job_name}_input.xyz --chrg {settings.charge} ' \
              f'--uhf {settings.multiplicity - 1} ' \
              f'{self.config.xtb_command} -T {settings.n_proc} -alpb water\n'
        # Write the hessian.inp which is the command line input
        file.write(cmd)
        # Write the coordinates, which is the standard xyz file.
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

    def hessian(self, file, job_name, settings, coords, atnums):
        """ Write the input file for hessian and charge calculation.

        Parameters
        ----------
        file : file
            The file object to write the command line.
        job_name : string
            The name of the job.
        settings: SimpleNamespace
            A configparser object with all the parameters.
        coords : array
            A coordinates array of shape (N,3), where N is the number of atoms.
        atnums : list
            A list of atom elements represented as atomic number.
        """
        raise NotImplementedError

    def sp(self, file, job_name, settings, coords, atnums):
        raise NotImplementedError

    def charges(self, file, job_name, settings, coords, atnums):
        raise NotImplementedError

    def scan(self, file, job_name, settings, coords, atnums, scanned_atoms,
             start_angle, charge, multiplicity):
        """ Write the input file for the dihedral scan and charge calculation.

        Parameters
        ----------
        file : file
            The file object to write the input.
        job_name : string
            The name of the job.
        settings: SimpleNamespace
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
        raise NotImplementedError
