import os.path

import numpy as np
from ase.units import Hartree, mol, kJ, Bohr
from ase.io import read, write
from ase import Atoms
#
from .qm_base import WriteABC, ReadABC, QMInterface, Calculator
#
from .gaussian import Gaussian, ReadGaussian, WriteGaussian


class WriteXTBGaussian(WriteGaussian):

    @staticmethod
    def write_method(file, config):
        file.write('external="gauext-xtb " ')

    @staticmethod
    def write_pop(file, string):
        pass

    @staticmethod
    def _write_bndix(file):
        file.write("\n\n\n")

    def _write_scan_job_setting(self, job_name, settings, file, charge, multiplicity):
        file.write(f"%nprocshared={settings.n_proc}\n")
        file.write(f"%mem={settings.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p Opt=(Modredundant,nomicro) ")
        self.write_method(file, self.config)
        file.write("\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{charge} {multiplicity}\n")

    def _write_charge_job_setting(self, job_name, settings, file, charge, multiplicity):
        file.write(f"%nprocshared={settings.n_proc}\n")
        file.write(f"%mem={settings.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p ")
        self.write_method(file, self.config)
        file.write("\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{charge} {multiplicity}\n")

    def _write_opt_job_setting(self, job_name, settings, file):
        file.write(f"%nprocshared={settings.n_proc}\n")
        file.write(f"%mem={settings.memory}MB\n")
        file.write(f"%chk={job_name}_hessian.chk\n")
        file.write("#p Opt=(nomicro) ")
        self.write_method(file, self.config)
        file.write("\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{settings.charge} {settings.multiplicity}\n")

    def _write_hessian_job_setting(self, job_name, settings, file):
        file.write(f"%nprocshared={settings.n_proc}\n")
        file.write(f"%mem={settings.memory}MB\n")
        file.write(f"%chk={job_name}_hessian.chk\n")
        file.write("#p Opt=nomicro Freq ")
        self.write_method(file, self.config)
        file.write("\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{settings.charge} {settings.multiplicity}\n")

    def _write_sp_job_setting(self, job_name, settings, file):
        file.write(f"%nprocshared={settings.n_proc}\n")
        file.write(f"%mem={settings.memory}MB\n")
        file.write(f"%chk={job_name}_hessian.chk\n")
        file.write("#p ")
        self.write_method(file, self.config)
        file.write("\n\n")
        file.write(f"{job_name}\n\n")
        file.write(f"{settings.charge} {settings.multiplicity}\n")

    def _write_scan_torsiondrive_job_settings(self, job_name, settings, file, charge, multiplicity):
        file.write(f"%nprocshared={settings.n_proc}\n")
        file.write(f"%mem={settings.memory}MB\n")
        file.write(f"%chk={job_name}.chk\n")
        file.write("#p Opt=ModRedundant nosym ")
        self.write_method(file, self.config)


class ReadXTBGaussian(ReadGaussian):

    @staticmethod
    def _read_charges(file, n_atoms):
        point_charges = []
        for _ in range(n_atoms):
            line = file.readline()
            point_charges.append(float(line))
        return point_charges

    @classmethod
    def _read_esp_charges(cls, file, n_atoms):
        return cls._read_charges(file, n_atoms)

    @classmethod
    def _read_cm5_charges(cls, file, n_atoms):
        return cls._read_charges(file, n_atoms)

    @staticmethod
    def _read_nbo_analysis(file, line, n_atoms):
        found_wiberg = False
        lone_e = np.zeros(n_atoms, dtype=int)
        n_bonds = []
        b_orders = [[] for _ in range(n_atoms)]
        while "--- END WBO ANALYSIS ---" not in line:
            line = file.readline()
            if ("bond index matrix" in line and not found_wiberg):
                for _ in range(int(np.ceil(n_atoms/9))):
                    for atom in range(-3, n_atoms):
                        line = file.readline().split()
                        if atom >= 0:
                            order = [float(line_cut) for line_cut in line[2:]]
                            b_orders[atom].extend(order)
            if ("bond index, Totals" in line and not found_wiberg):
                found_wiberg = True
                for i in range(-3, n_atoms):
                    line = file.readline()
                    if i >= 0:
                        n_bonds.append(int(round(float(line.split()[2]), 0)))
            if "Natural Bond Orbitals (Summary)" in line:
                while "Total Lewis" not in line:
                    line = file.readline()
                    if " LP " in line:
                        atom = int(line[19:23])
                        occ = int(round(float(line[40:48]), 0))
                        if occ > 0:
                            lone_e[atom-1] += occ
        return n_bonds, b_orders, lone_e


class XTBGaussian(Gaussian):

    name = 'gaussian'

    _user_input = """

    charge_method = xtb :: str :: xtb

    # QM method to be used
    method = gnf2 :: str :: gnf2

    """

    def __init__(self, config):
        read = ReadXTBGaussian(config)
        write = WriteXTBGaussian(config)
        self._setup(config, read, write)


class xTB(QMInterface):

    _user_input = """
    # xTB only allows Mulliken charge.
    charge_method = xtb :: str :: xtb

    # Extra command line passed to the xtb executable
    xtb_command = --gfn 2 :: str ::

    """

    name = 'xtb'
    has_torsiondrive = True

    _method = ['xtb_command', 'charge_method']

    def __init__(self, config):
        super().__init__(config, ReadxTB(config), WritexTB(config))


class xTBCalculator(Calculator):

    name = 'xtb'
    _user_input = ""

    @classmethod
    def from_config(cls, config):
        return cls()

    def _commands(self, filename, basename, ncores):
        return [f'bash {filename} > {filename}.shellout']


class WritexTB(WriteABC):

    def gradient(self, file, job_name, settings, coords, atnums, extra_info=False):
        """ Write the input file for sp gradient

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
        cmd = f'xtb {job_name}_input.xyz --grad --chrg {settings.charge} ' \
              f'--uhf {settings.multiplicity - 1} ' \
              f'--namespace {job_name} --parallel {settings.n_proc} ' \
              f'{self.config.xtb_command} ' \
              f'> {job_name}.out'

        # Write the hessian.inp which is the command line input
        file.write(cmd)
        # Write the coordinates, which is the standard xyz file.
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

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
        cmd = f'xtb {job_name}_input.xyz --opt --chrg {settings.charge} ' \
              f'--uhf {settings.multiplicity - 1} ' \
              f'--namespace {job_name} --parallel {settings.n_proc} ' \
              f'{self.config.xtb_command} ' \
              f'> {job_name}.out'

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
        name = file.name
        base, filename = os.path.split(name)
        # Given that the xTB input has to be given in the command line.
        # We create the xTB command template here.
        cmd = f'xtb {job_name}_input.xyz --ohess --chrg {settings.charge} ' \
              f'--uhf {settings.multiplicity - 1} ' \
              f'--namespace {job_name} --parallel {settings.n_proc} ' \
              f'{self.config.xtb_command} ' \
              f'> {job_name}.out'
        # Write the hessian.inp which is the command line input
        file.write(cmd)
        # Write the coordinates, which is the standard xyz file.
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

    def sp(self, file, job_name, settings, coords, atnums):
        name = file.name
        base, filename = os.path.split(name)
        # Given that the xTB input has to be given in the command line.
        # We create the xTB command template here.
        cmd = f'xtb {job_name}_input.xyz --chrg {settings.charge} ' \
              f'--uhf {settings.multiplicity - 1} ' \
              f'--namespace {job_name} --parallel {settings.n_proc} ' \
              f'{self.config.xtb_command} ' \
              f'> {job_name}.sp.inp.out'
        # Write the hessian.inp which is the command line input
        file.write(cmd)
        # Write the coordinates, which is the standard xyz file.
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

    def charges(self, file, job_name, settings, coords, atnums):
        name = file.name
        base, _ = os.path.split(name)
        # Given that the xTB input has to be given in the command line.
        # We create the xTB command template here.
        cmd = f'xtb {job_name}_input.xyz --chrg {settings.charge} ' \
              f'--uhf {settings.multiplicity - 1} ' \
              f'--namespace {job_name} --parallel {settings.n_proc} ' \
              f'{self.config.xtb_command} ' \
              f'> {job_name}_sp.out'
        # Write the hessian.inp which is the command line input
        file.write(cmd)
        # Write the coordinates, which is the standard xyz file.
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

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
        # Write the xTB input file
        name = file.name
        base, filename = os.path.split(name)

        # Generate the command line
        cmd = f'xtb {job_name}_input.xyz --opt --chrg {settings.charge} ' \
              f'--uhf {settings.multiplicity - 1} ' \
              f'--namespace {job_name} --parallel {settings.n_proc} ' \
              f'--input {job_name}.dat {self.config.xtb_command} ' \
              f'> {job_name}.out'

        # Create the scan input file
        a1, a2, a3, a4 = np.array(scanned_atoms)
        step_num = int(360 // settings.scan_step_size)
        end_angle = start_angle + 360 - settings.scan_step_size

        with open(f'{base}/{job_name}.dat', 'w') as f:
            f.write('$constrain\n')
            f.write('  force constant=0.1\n')
            f.write('$scan\n')
            f.write(f'  dihedral: {a1},{a2},{a3},{a4},{start_angle:.2f}; '
                    f'{start_angle:.2f},{end_angle:.2f},{step_num}\n')
            f.write('$end\n')

        # Write the coordiante file in the xyz file format
        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)

        file.write(cmd)

    def scan_torsiondrive(self, file, job_name, settings, coords, atnums, scanned_atoms,
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
        base, _ = os.path.split(file.name)

        self._scan_torsiondrive_helper(file, job_name, settings, scanned_atoms, 'xtb')

        cmd = (f'xTB arguments: --opt --chrg {settings.charge} --uhf 0 '
               f'{self.config.xtb_command} --parallel {settings.n_proc} ')

        mol = Atoms(positions=coords, numbers=atnums)
        write(f'{base}/{job_name}_input.xyz', mol, plain=True,
              comment=cmd)


class ReadxTB(ReadABC):

    hessian_files = {'hess_file': ['${base}.hessian'],
                     'pc_file': ['${base}.charges'],
                     'wbo_file': ['${base}.wbo'],
                     'coord_file': ['${base}.xtbopt.xyz'],
                     }

    opt_files = {'coord_file': ['${base}.xtbopt.xyz'], }

    gradient_files = {'grad_file': ['${base}.gradient'], 'xyz_file': ['${base}_input.xyz']}

    sp_files = {'sp_file': ['${base}.sp.inp.out'], }
    sp_ec_files = {'sp_file': ['${base}.sp.inp.out'], 'xyz_file': ['${base}_input.xyz']}

    charge_files = {'pc_file': ['${base}.charges']}

    scan_files = {
            'charge_file': ['${base}.charges'],
            'scan_log':  ['${base}.xtbscan.log'],
            'input_dat':  ['${base}.dat'],
            }

    def opt(self, config, coord_file):
        _, elements, coords = self._read_xtb_xyz(coord_file)
        return [coords]

    def sp(self, config, sp_file):
        with open(sp_file, 'r') as fh:
            for line in fh:
                if 'TOTAL ENERGY' in line:
                    energy = float(line.split()[3])
        energy = energy * Hartree * mol / kJ
        return energy

    def sp_ec(self, config, sp_file, xyz_file):
        molecule = read(xyz_file)
        with open(sp_file, 'r') as fh:
            for line in fh:
                if 'TOTAL ENERGY' in line:
                    energy = float(line.split()[3])
        energy = energy * Hartree * mol / kJ
        dipol = None
        return energy, dipol, molecule.get_atomic_numbers(), molecule.get_positions()

    def charges(self, config, pc_file):
        _, point_charges = self._read_xtb_charge(pc_file)
        return {'xtb': np.array(point_charges)}

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
        n_atoms, elements, coords, energy = self._read_xtb_xyz_and_energy(coord_file)
        #
        charge = config.charge
        multiplicity = config.multiplicity
        b_orders = self._read_xtb_wbo_analysis(wbo_file, elements)
        hessian = self._read_xtb_hess(hess_file, n_atoms)
        # not defined
        dip_ders = None
        energy = energy * Hartree * mol / kJ

        return (n_atoms, charge, multiplicity, elements, coords, energy, hessian,
                b_orders, point_charges, dip_ders)

    def gradient(self, config, grad_file, xyz_file):
        """ Read data from the grad file.

        Parameters
        ----------
        config : config
            A configparser object with all the parameters.

        grad_file : string
            File name of the xtb grad file.

        xyz_file : string
            File name of the xtb input file.

        Returns
        -------
        energy : float
            The scf energy of the system

        gradient : array(dtype=float)
            The gradient of the molecule.

        """
        molecule = read(xyz_file)
        with open(grad_file, 'r') as fh:
            # skip first line
            next(fh)
            # read energy
            energy = float(next(fh).split()[6])
            natoms = 0
            cols = next(fh).split()
            while len(cols) == 4:
                natoms += 1
                cols = next(fh).split()

            grad = []

            while len(cols) == 3:
                natoms += 1
                grad.append([float(val) for val in cols])
                cols = next(fh).split()

        energy = energy * Hartree * mol / kJ
        grad = np.array(grad) * Hartree * mol / kJ / Bohr
        dipol = None
        return energy, grad, dipol, molecule.get_atomic_numbers(), molecule.get_positions()

    def scan_new(self, config, charge_file, scan_log, input_dat):
        """ Read data from the scan file.

        Parameters
        ----------
        config : config
            A configparser object with all the parameters.

        charge_file : string
            File name of the xTB charges.

        scan_log : string
            File name of the xTB scan log

        input_dat : string
            input of the dihedral scan

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
        point_charges = {}
        n_atoms, charges = self._read_xtb_charge(charge_file)
        point_charges["xtb"] = charges

        elements, energies, coords = self._read_xtb_scan_log(scan_log)

        angles = self._read_xtb_input_angle(input_dat)

        energies = np.array(energies) * Hartree * mol / kJ
        dipoles = [[None, None, None] for _ in angles]

        return n_atoms, coords, angles, energies, dipoles, point_charges

    def scan(self, config, file_name):
        """ Read data from the scan file.

        Parameters
        ----------
        config : config
            A configparser object with all the parameters.

        file_name : string
            File name of the xTB log file.

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
        dipoles = [[None, None, None] for _ in angles]

        return n_atoms, coords, angles, energies, dipoles, point_charges

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
    def _read_xtb_xyz_and_energy(coord_file):
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
        with open(coord_file, 'r') as fh:
            next(fh)
            comment = next(fh).split()
            energy = float(comment[1])
        mol = read(coord_file)

        n_atoms = len(mol)
        elements = np.array([atom.number for atom in mol])
        coords = mol.positions
        return n_atoms, elements, coords, energy

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
