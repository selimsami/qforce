import os.path
import re
import subprocess

import numpy as np
from ase.units import Hartree, mol, kJ, Bohr
from ase.io import read
#
from .qm_base import WriteABC, ReadABC, QMInterface, Calculator

from ..elements import ATOM_SYM


class Orca(QMInterface):

    name = 'orca'
    has_torsiondrive = False

    _user_input = """

    charge_method = hirshfeld :: str :: [hirshfeld, esp]

    # QM method to be used for hessian calculation
    # Note: The accuracy of this method determines the accuracy of bond,
    # angle and improper dihedral.
    method = PBE

    # basis set to be used
    basis = 6-31+G(D) :: str

    # dispersion
    dispersion = D3BJ :: str, optional :: [D2, D3, D3BJ, D3ZERO, D4]

    # additional options
    options =  :: str, optional
    """

    _method = ['method', 'basis', 'dispersion', 'options']

    def __init__(self, config):
        if not config.options:
            config.options = ''
        super().__init__(config, ReadORCA(config), WriteORCA(config))


class OrcaCalculator(Calculator):

    name = 'orca'
    _user_input = ""
    
    @classmethod
    def from_config(cls, config):
        return cls()

    def _commands(self, filename, basename, ncores):
        raise NotImplementedError



class WriteORCA(WriteABC):

    def opt(self, file, job_name, settings, coords, atnums):
        self._write_coordinates_and_defaults(file, settings, atnums, coords)
        file.write(f"! opt {self.config.method} {self.config.basis} ")
        file.write(f" {self.config.options} {self.config.dispersion} nopop\n")
        file.write(f'%base "{job_name}"\n')

    def sp(self, file, job_name, settings, coords, atnums):
        self._write_coordinates_and_defaults(file, settings, atnums, coords)
        file.write(f"! {self.config.method} {self.config.basis} ")
        file.write(f" {self.config.options} {self.config.dispersion} nopop\n\n")

    def charges(self, file, job_name, settings, coords, atnums):
        self._write_coordinates_and_defaults(file, settings, atnums, coords)
        file.write(f"! {self.config.method} {self.config.basis} ")
        file.write(f" {self.config.options} {self.config.dispersion} chelpg Hirshfeld nopop\n")
        file.write(f'%base "{job_name}"\n\n')

    def hessian(self, file, job_name, settings, coords, atnums):
        """ Write the input file for hessian and charge calculation.

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
        """

        self._write_coordinates_and_defaults(file, settings, atnums, coords)
        # Start compound job
        file.write('%Compound\n\n')

        # Do the hessian calculation
        file.write('New_Step\n')
        file.write("! opt freq ")
        file.write(f"{self.config.method} {self.config.basis} ")
        file.write(f"{self.config.dispersion} {self.config.options}")
        file.write(" PModel nopop\n")
        file.write(f'%base "{job_name}_opt"\n')
        file.write('STEP_END\n\n')

        # Do the nbo calculation
        file.write('New_Step\n')
        file.write(f"! {self.config.method} {self.config.basis} ")
        file.write(f" {self.config.dispersion} {self.config.options} \n")
        file.write(f'%base "{job_name}_nbo"\n')
        file.write('%method\n  MAYER_BONDORDERTHRESH 0\nend\n')
        file.write('STEP_END\n\n')

        # Write the charge calculation input
        file.write('New_Step\n')
        file.write(f"! {self.config.method} {self.config.basis} chelpg Hirshfeld nopop\n")
        file.write(f'%base "{job_name}_charge"\n')
        file.write('STEP_END\n\n')
        file.write('END\n')

    def scan(self, file, job_name, settings, coords, atnums, scanned_atoms, start_angle, charge,
             multiplicity):
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
        self._write_coordinates_and_defaults(file, settings, atnums, coords)
        # Start compound job
        file.write('%Compound\n\n')

        # Do the scan
        file.write('New_Step\n')
        file.write(f"! opt {self.config.method} {self.config.basis} ")
        file.write(f" {self.config.options} {self.config.dispersion} nopop\n")
        file.write(f'%base "{job_name}_scan"\n')
        self._write_scanned_atoms(file, scanned_atoms, start_angle, settings.scan_step_size)
        file.write('STEP_END\n\n')

        # Get charge
        file.write('New_Step\n')
        file.write(f"! {self.config.method} {self.config.basis} ")
        file.write(" chelpg Hirshfeld PModel nopop\n")
        file.write(f'%base "{job_name}_charge"\n')
        file.write(f"*xyzfile {charge} {multiplicity} {job_name}_scan.001.xyz\n")
        file.write('STEP_END\n\n')

        # Close compound block
        file.write('END\n')

    def _write_coordinates_and_defaults(self, file, settings, atnums, coords):
        # Using the ORCA compound functionality
        # Write the coordinates
        file.write(f"* xyz   {settings.charge}   {settings.multiplicity}\n")
        self._write_coords(atnums, coords, file)
        file.write(' *\n\n')

        file.write(f"%pal nprocs  {settings.n_proc} end\n")
        # ORCA uses MPI parallelization and a factor of 0.75 is used to
        # avoid ORCA using more than it is available.
        file.write(f'%maxcore  {int(settings.memory / settings.n_proc * 0.75)}\n\n')

    @staticmethod
    def _write_scanned_atoms(file, scanned_atoms, start_angle, step_size):
        """ Write the input line for dihedral scan.

        Parameters
        ----------
        file : file
            The file object to write the input.
        scanned_atoms : list
            A list of 4 integer (one-based index) of the dihedral
        start_angle : float
            The starting angle in degree.
        step_size : float
            The size of the step.
        """
        # ORCA uses zero-based indexing
        a1, a2, a3, a4 = np.array(scanned_atoms) - 1
        start_angle = float(start_angle)
        # Remove the last point as it is the same as the first point
        end_angle = start_angle + 360 - step_size
        n_steps = int(np.ceil(360 / step_size))
        file.write("%geom Scan\n")
        file.write(f"D {a1} {a2} {a3} {a4} = {start_angle:.2f},"
                   f" {end_angle:.2f},"
                   f" {n_steps}\n")
        file.write("end\n")
        file.write("end\n")

    @staticmethod
    def _write_constrained_atoms(file, scanned_atoms):
        """ Write the input line for the constrained dihedral.

        Parameters
        ----------
        file : file
            The file object to write the input.
        scanned_atoms : list
            A list of 4 integer (one-based index) of the dihedral
        """
        # ORCA uses zero-based indexing
        a1, a2, a3, a4 = np.array(scanned_atoms) - 1
        file.write("%geom\n")
        file.write("Constraints\n")
        file.write(f"{{D {a1} {a2} {a3} {a4} C }}\n")
        file.write("end\n")
        file.write("end\n")

    @staticmethod
    def _write_coords(atnums, coords, file):
        """ Write the input coordinates.

        Parameters
        ----------
        atnums : list
            A list (N) of atom elements represented as atomic number.
        coords : array
            A coordinates array of shape (N,3), where N is the number of atoms.
        file : file
            The file object to write the input.
        """
        for atnum, coord in zip(atnums, coords):
            elem = ATOM_SYM[atnum]
            file.write(f'{elem :>3s} {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}\n')


class ReadORCA(ReadABC):

    # TODO: Check if pc_file is always present, or only in special cases?
    hessian_files = {'out_file': ['${base}.out', '${base}.log'],
                     'hess_file': ['${base}_opt.hess'],
                     # 'pc_file': ['${base}_charge.pc_chelpg'],
                     'coord_file': ['${base}_opt.xyz']}

    opt_files = {'coord_file': ['${base}_opt.xyz']}
    sp_files = {'out_file': ['${base}.out', '${base}.log']}
    charge_files = {'out_file': ['${base}.out', '${base}.log']}
    scan_files = {'out_file': ['${base}.out', '${base}.log'],
                  'scan_file': ['${base}_scan.allxyz'],
                  # possible add _scan.relaxscanact.dat etc.
                  }

    def opt(self, settings, coord_file):
        n_atoms, elements, coords = self._read_orca_xyz(coord_file)
        return [coords]

    def sp(self, settings, out_file):
        energy = None
        with open(out_file, 'r') as fh:
            for line in fh:
                if 'TOTAL SCF ENERGY' in line:
                    next(fh)
                    next(fh)
                    line = next(fh).split()
                    energy = float(line[3])

        if energy is not None:
            return energy * Hartree * mol / kJ
        raise ValueError("Could not parse orca file")

    def charges(self, settings, out_file):
        if self.config.charge_method == "hirshfeld":
            n_atoms, point_charges = self._read_orca_hirshfeld(out_file)
        elif self.config.charge_method == "esp":
            base, ext = os.path.splitext(out_file)
            n_atoms, point_charges = self._read_orca_esp(f'{base}_charge.pc_chelpg')
        else:
            raise ValueError("charge method unknown!")
        return {self.config.charge_method: point_charges}

    def hessian(self, settings, out_file, hess_file, coord_file):
        """ Extract information from all the relevant files.

        Parameters
        ----------
        settings: SimpleNamespace
            A configparser object with all the parameters.
        out_file : string
            File name of the ORCA log file.
        hess_file : string
            File name of the ORCA hess file for hessian information.
        coord_file : string
            File name of the ORCA geometry optimised coordinate file.

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
        hessian = self._read_orca_hess(hess_file)
        #
        if self.config.charge_method == "hirshfeld":
            n_atoms, point_charges = self._read_orca_hirshfeld(out_file)
        elif self.config.charge_method == "esp":
            base, ext = os.path.splitext(out_file)
            n_atoms, point_charges = self._read_orca_esp(f'{base}_charge.pc_chelpg')
        #
        n_atoms, elements, coords = self._read_orca_xyz(coord_file)
        charge = settings.charge
        multiplicity = settings.multiplicity
        b_orders = self._read_orca_bond_order(out_file, n_atoms)
        return n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges

    def scan(self, settings, out_file, scan_file):
        """ Read data from the scan file.

        Parameters
        ----------
        settings: SimpleNamespace
            A configparser object with all the parameters.
        out_file: string
            File name of the ORCA log file.
        scan_file: string
            File name of the ORCA scan file.

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
        base, ext = os.path.splitext(out_file)
        point_charges = {}
        #
        if self.config.charge_method == "hirshfeld":
            n_atoms, charges = self._read_orca_hirshfeld(out_file)
            point_charges["hirshfeld"] = charges
        elif self.config.charge_method == "esp":
            n_atoms, charges = self._read_orca_esp(f'{base}_charge.pc_chelpg')
            point_charges["esp"] = charges
        #
        n_atoms, elements, coords = self._read_orca_allxyz(scan_file)
        angles, energies = self._read_orca_dat(f'{base}_scan.relaxscanact.dat')
        if os.path.isfile(f'{base}_sp.xyzact.dat'):
            _, energies = self._read_orca_dat(f'{base}_sp.xyzact.dat')
        energies = np.array(energies) * Hartree * mol / kJ
        return n_atoms, coords, angles, energies, point_charges

    @staticmethod
    def _read_orca_hess(hess_file):
        """ Read the hessian matrix.

        For ORCA jobs, the file contain the hessian information is stored
        with the extension of hess.

        Parameters
        ----------
        hess_file : string
            The name of the hess file.

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

        n_atoms_times_3 = int(lines[0])
        lines = lines[1:]
        hessian = np.empty((n_atoms_times_3, n_atoms_times_3))

        for _ in range(int(np.ceil(n_atoms_times_3 / 5))):
            trunk = lines[:(n_atoms_times_3 + 1)]
            lines = lines[(n_atoms_times_3 + 1):]
            cols = [int(col) for col in trunk[0].split()]
            for row in range(n_atoms_times_3):
                row_idx, *points = trunk[1 + row].split()
                assert int(row_idx) == row
                for point, col in zip(points, cols):
                    hessian[row, col] = point

        # Output the lower triangle of the hessian matrix to match the
        # format adopted by Gaussian and Qchem.
        out_hessian = []
        for i in range(len(hessian)):
            for j in range(i + 1):
                hes = (hessian[i, j] + hessian[j, i]) / 2
                out_hessian.append(hes)
        return np.array(out_hessian) * Hartree * mol / kJ / Bohr**2

    @staticmethod
    def _read_orca_esp(pc_file):
        """ Read the point charge file.

        For ORCA jobs, the point charge computed by chelpg method will
        yield a file contain the charge information stored with the extension
        of pc_chelpg.

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
        with open(pc_file, 'r') as f:
            lines = f.read().split('\n')
        n_atoms = int(lines[0])
        point_charges = []
        for line in lines[2: 2+n_atoms]:
            point_charges.append(float(line.split()[1]))
        return n_atoms, point_charges

    @staticmethod
    def _read_orca_hirshfeld(out_file):
        """ Read the HIRSHFELD charge file.

        Parameters
        ----------
        out_file : string
            The name of the calculation log file.

        Returns
        -------
        n_atoms : int
            The number of atoms in the molecule.
        point_charges : float
            A list of float of the size of n_atoms.
        """
        with open(out_file, 'r') as file:
            line = file.readline()
            while 'HIRSHFELD ANALYSIS' not in line:
                line = file.readline()

            while 'ATOM     CHARGE      SPIN' not in line:
                line = file.readline()

            charges = []

            while 'TOTAL' not in line:
                line = file.readline()
                if len(line.split()) == 4:
                    atom_id, element, charge, _ = line.split()
                    charges.append(float(charge))
        # atom_id is zero-based index
        return int(atom_id) + 1, charges

    @staticmethod
    def _read_orca_coords(coord_text):
        """ Read the optimised coordinate string.

        The string is in the format of the standard xyz file.

        Parameters
        ----------
        coord_text : string
            The content of the xyz file file.

        Returns
        -------
        n_atoms : int
            The number of atoms in the molecule.
        elements : array
            A np.array of integer of the atomic number of the atoms.
        coords : array
            An array of float of the shape (n_atoms, 3).
        """
        lines = coord_text.split('\n')
        n_atoms = int(lines[0])
        coords = np.empty((n_atoms, 3))
        elements = []
        for i, line in enumerate(lines[2: 2+n_atoms]):
            element, x, y, z = line.split()
            elements.append(ATOM_SYM.index(element))
            coords[i, :] = (x, y, z)
        return n_atoms, np.array(elements), coords

    @staticmethod
    def _read_orca_xyz(coord_file):
        """ Read the optimised coordinate xyz file.

        For ORCA jobs, the optimised geometry will be stored as a file
        with the extension xyz.

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
    def _read_orca_allxyz(coord_file):
        """ Read the optimised coordinate allxyz file.

        For ORCA jobs, the scan optimised geometry will be stored as a file
        with the extension allxyz.

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
        coords : list
            A list of array of float. The list has the length of the number
            of steps and the array has the shape of (n_atoms, 3).
        """
        with open(coord_file, 'r') as f:
            coord_text = f.read()
        coords = []
        for text in coord_text.split('>\n'):
            n_atoms, elements, coord = ReadORCA._read_orca_coords(text)
            coords.append(coord)
        return n_atoms, elements, coords

    @staticmethod
    def _read_orca_bond_order(out_file, n_atoms):
        """ Read the bond order analysis.

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
        item_match = re.compile(r'^\(\s*(\d+)-\w{1,2}\s*,\s*(\d+)-\w{1,2}\s*\)\s*:\s*(-?\w.+)$')
        b_orders = [[0, ] * n_atoms for _ in range(n_atoms)]

        with open(out_file, 'r') as file:
            line = file.readline()
            # Skip to the step after geometry optimisation
            while 'Mayer bond orders larger than' not in line:
                line = file.readline()

            line = file.readline()
            while "-------" not in line:
                items = line.split('B')
                for item in items:
                    if item.strip():
                        _m = re.match(item_match, item)
                        i = int(_m.group(1))
                        j = int(_m.group(2))
                        bond_order = float(_m.group(3))
                        b_orders[i][j] = bond_order
                        b_orders[j][i] = bond_order
                line = file.readline()

        return b_orders

    @staticmethod
    def _read_orca_dat(dat):
        """ Read the scan parameter and energy from ORCA relaxscanact.dat or
        xyzact.dat file.

        Parameters
        ----------
        filename : string
            File name of the ORCA dat file for scan information.

        Returns
        -------
        parameter : list
            A list (length: steps) of the parameter that is being scanned.
        energies : list
            A list (length: steps) of the energy.
        """
        with open(dat, 'r') as f:
            coord_text = f.read().strip()
        parameter = []
        energies = []
        for line in coord_text.split('\n'):
            parameter.append(float(line.split()[0]))
            energies.append(float(line.split()[1]))
        return parameter, energies
