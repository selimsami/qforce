import math
import os
import sys
import json
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from warnings import warn
from contextlib import contextmanager
import hashlib
import subprocess

import numpy as np
from ase.units import Hartree, mol, kJ, Bohr
from ase.io import read
from colt import Colt


class QMInterface(Colt):
    """Basic Logic for an QM Interface"""

    # Please specify the name of the qm interface
    name = None
    has_torsiondrive = False

    def __init__(self, config, read, write):
        self._setup(config, read, write)

    def hash(self, charge, mult):
        """Returns a unique hash for the given interface"""
        return self._dct_to_hash(self.settings(charge, mult))

    def _settings(self):
        """Every QMInterface needs this, it defines the unique keys
        should not contain information like number of cores or size of memory
        but only relevant information for the calculation (basisset, functional etc.)
        """
        raise NotImplementedError("Please provide settings method")

    def settings(self, charge, mult):
        """Returns the unique settings of the qm interface"""
        settings = self._settings()
        settings['charge'] = charge
        settings['multiplicity'] = mult
        return settings

    @staticmethod
    def _dct_to_hash(settings):
        return hashlib.md5(''.join(f'{key.lower()}:{str(value).lower()}'
                                   for key, value in settings.items()
                                   ).encode()).hexdigest()

    def _setup(self, config, read, write):
        """Setup qm interface"""
        self.read = read
        # get required files
        self.required_hessian_files = read.hessian_files
        self.required_opt_files = read.opt_files
        self.required_sp_files = read.sp_files
        self.required_charge_files = read.charge_files
        self.required_scan_files = read.scan_files
        self.required_scan_torsiondrive_files = read.scan_torsiondrive_files
        #
        self.write = write
        self.config = config


class WriteABC(ABC):
    """Abstract baseclass of QM Interface Write classes"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def hessian(self, ):
        ...

    @abstractmethod
    def scan(self, ):
        ...

    @abstractmethod
    def opt(self, ):
        ...

    @abstractmethod
    def sp(self, ):
        ...

    def scan_torsiondrive(self, ):
        ...

    def _scan_torsiondrive_helper(self, file, job_name, config, scanned_atoms, engine):
        """ Write the input file for the dihedral scan and charge calculation.

        Parameters
        ----------
        file : file
            The file object to write the submission.
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
        engine: str
            The engine to use in torsion-drive
        """
        base, _ = os.path.split(file.name)
        # Given that the xTB input has to be given in the command line.
        # We create the xTB command template here.
        # Write the hessian.inp which is the command line input
        file.write(f"torsiondrive-launch {job_name}_input.xyz {job_name}_dihedrals.txt -g {int(config.scan_step_size)} -e {engine} --native_opt -v > {job_name}.log ")
        # Write the coordinates, which is the standard xyz file.
        with open(f'{base}/{job_name}_dihedrals.txt', 'w') as fh:
            fh.write('{} {} {} {}'.format(*scanned_atoms))


class ReadABC(ABC):
    """Abstract baseclass of QM Interface Read classes"""

    scan_torsiondrive_files = {'xyz': ['scan.xyz']}

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def hessian(self, ):
        ...

    @abstractmethod
    def scan(self, ):
        ...

    def scan_torsiondrive(self, log_file):
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
        folder, _ = os.path.split(log_file)
        frames = read(os.path.join(folder, 'scan.xyz'), index=':', format='extxyz')
        n_atoms = len(frames[0])
        energy_list = []
        coord_list = []
        angle_list = []
        # load energies
        for frame in frames:
            coord_list.append(frame.positions)
            _, angle, _, energy = list(frame.info.keys())
            angle = float(angle[1:-2])
            angle_list.append(angle)
            energy = float(energy)
            energy_list.append(energy)
        # convert to gromacs units
        energies = np.array(energy_list) * Hartree * mol / kJ
        return n_atoms, coord_list, angle_list, energies, {}

    @abstractmethod
    def opt(self, ):
        ...

    @abstractmethod
    def sp(self, ):
        ...
    @staticmethod
    def _read_fchk_file(fchk_file):
        n_atoms, charge, multiplicity, elements, coords, hessian = None, None, None, [], [], []
        with open(fchk_file, "r", encoding='utf-8') as file:
            for line in file:
                if "Charge                                     I" in line:
                    charge = int(line.split()[2])
                if "Multiplicity                               I" in line:
                    multiplicity = int(line.split()[2])
                if "Number of atoms  " in line:
                    n_atoms = int(line.split()[4])
                if "Atomic numbers  " in line:
                    n_line = math.ceil(n_atoms/6)
                    for _ in range(n_line):
                        line = file.readline()
                        ids = [int(i) for i in line.split()]
                        elements.extend(ids)
                if "Current cartesian coordinates   " in line:
                    n_line = math.ceil(3*n_atoms/5)
                    for _ in range(n_line):
                        line = file.readline()
                        coords.extend(line.split())
                if "Cartesian Force Constants  " in line:
                    n_line = math.ceil(3*n_atoms*(3*n_atoms+1)/10)
                    for _ in range(n_line):
                        line = file.readline()
                        hessian.extend(line.split())

        coords = np.asfarray(coords, float)
        hessian = np.asfarray(hessian, float)
        coords = np.reshape(coords, (-1, 3))
        elements = np.array(elements)
        coords = coords * Bohr
        hessian = hessian * Hartree * mol / kJ / Bohr**2
        return n_atoms, charge, multiplicity, elements, coords, hessian

    @staticmethod
    def _read_bond_order_from_nbo_analysis(file, n_atoms):
        b_orders = [[] for _ in range(n_atoms)]

        for line in file:
            if "bond index matrix" in line:
                for _ in range(int(np.ceil(n_atoms/9))):
                    for _ in range(3):
                        next(file)
                    for atom in range(n_atoms):
                        line = file.readline().split()
                        order = [float(line_cut) for line_cut in line[2:]]
                        b_orders[atom].extend(order)
                return b_orders
        return None


class HessianOutput():
    def __init__(self, vib_scaling, n_atoms, charge, multiplicity, elements, coords, hessian,
                 b_orders, point_charges, lone_e=None, n_bonds=None):

        self.n_atoms = self.check_type(n_atoms, 'n_atoms', int)
        self.charge = self.check_type(charge, 'charge', int)
        self.multiplicity = self.check_type(multiplicity, 'n_atoms', int)
        self.elements = self.check_type_and_shape(elements, 'elements', int, (n_atoms,))
        self.coords = self.check_type_and_shape(coords, 'coords', float, (n_atoms, 3))
        self.hessian = self.check_type_and_shape(hessian, 'hessian', float,
                                                 (((n_atoms*3)**2+n_atoms*3)/2,)) * vib_scaling**2
        self.b_orders = self.check_type_and_shape(b_orders, 'b_orders', float, (n_atoms, n_atoms))
        self.n_bonds = self.b_orders.sum(axis=1).round().astype(int)
        self.point_charges = self.check_type_and_shape(point_charges, 'point_charges', float,
                                                       (n_atoms,))

        if lone_e is not None or n_bonds is not None:
            warn('HessianOutput no longer needs "lone_e" or "n_bonds" arguments and they will be '
                 'removed in a future release.', DeprecationWarning, stacklevel=2)

    @staticmethod
    def check_type(value, name, expected_type):
        if not isinstance(value, expected_type):
            sys.exit(f'WARNING: A valid "{name}" property was not found in the hessian output'
                     ' file(s). Exiting...\n\n')
        return value

    @staticmethod
    def check_type_and_shape(value, name, expected_type, expected_shape):
        value = np.asarray(value)
        if value.size == 0:
            sys.exit(f'ERROR: No data found in the QM Hessian output file(s) for "{name}".'
                     ' Exiting...\n\n')
        elif value.dtype != np.dtype(expected_type):
            raise TypeError(f'"{name}" property expected a type of "{np.dtype(expected_type)}",'
                            f' but got "{value.dtype}" for the QM Hessian output.')
        elif value.shape != expected_shape:
            sys.exit(f'ERROR: "{name}" property expected a shape of "{expected_shape}", but got '
                     f'"{value.shape}" for the QM Hessian output. Exiting...\n\n')
        return value


class ScanOutput:
    """Store the output of a scan calculation"""

    def __init__(self, file,  n_steps, n_atoms, coords, angles, energies, charges):
        self.n_atoms = n_atoms
        self.n_steps = n_steps
        angles, energies, coords, self.charges, self.mismatch = self.check_shape(angles, energies,
                                                                                 coords, charges,
                                                                                 file)
        self._angles, self._energies, self.coords = self._rearrange(angles, energies, coords)

    @property
    def angles(self):
        return self._angles

    @property
    def energies(self):
        return self._energies

    @energies.setter
    def energies(self, energies):
        energies = np.asarray(energies, dtype=self._energies.dtype)
        if energies.size != self.n_steps:
            raise ValueError("Number of energies incomplete!")
        self._energies = energies

    def check_shape(self, angles, energies, coords, charges, file):
        mismatched = []
        if not isinstance(self.n_atoms, int):
            mismatched.append('n_atoms')
        if not isinstance(self.n_steps, int):
            mismatched.append('n_steps')

        angles, coords, energies = np.asarray(angles), np.asarray(coords), np.asarray(energies)

        for prop, name, shape in [(angles, 'angles', (self.n_steps,)),
                                  (energies, 'energies', (self.n_steps,)),
                                  (coords, 'coords', (self.n_steps, self.n_atoms, 3))]:
            if prop.shape != shape:
                mismatched.append(name)

        for key, val in charges.items():
            if len(val) == self.n_atoms:
                charges[key] = list(val)
            else:
                mismatched.append('charges')

        if mismatched:
            print(f'WARNING: {mismatched} properties have missing/extra data in the file:\n{file}')
        return angles, energies, coords, charges, mismatched

    @staticmethod
    def _rearrange(angles, energies, coords):
        if energies.size != 0:
            angles = (angles % 360).round(4)
            order = np.argsort(angles)
            angles = angles[order]
            coords = coords[order]
            energies = energies[order]
            energies -= energies.min()
        return angles, energies, coords


class CalculationIncompleteError(SystemExit):
    """Indicates that a calculation is incomplete and still files are missing"""


class Calculation:
    """Hold information of a calculation"""

    def __init__(self, inputfile, required_output_files, *, folder=None, software=None):
        self.folder = Path(folder) if folder is not None else Path("")
        self.required_output_files = required_output_files
        self.inputfile = self.folder / Path(inputfile)
        self.base = self.inputfile.name[:-len(self.inputfile.suffix)]
        self.software = software
        # register itself
        SubmitKeeper.add(self)

    @property
    def filename(self):
        return self.inputfile.name

    def _render(self, option):
        if '$' in option:
            option = Template(option)
            return option.substitute(base=self.base)
        return self.base + option

    def as_pycode(self):
        return (f'Calculation("{self.inputfile.name}", '
                f'{json.dumps(self.required_output_files)}, '
                f'folder="{str(self.folder)}", '
                f'software="{str(self.software)}")')

    @contextmanager
    def within(self):
        """Perform a set of option within the folder of the system"""
        current = os.getcwd()
        os.chdir(self.folder)
        yield
        os.chdir(current)

    def input_exists(self):
        """check if the inputfile is already present"""
        return self.inputfile.exists()

    def _get_files_dct(self):
        """gets a basic dictionary with all setted and missing files"""
        outfiles = {}

        for name, options in self.required_output_files.items():
            outfiles[name] = None
            for option in options:
                option = self._render(option)
                filename = self.folder / option
                if filename.exists():
                    outfiles[name] = filename
                    break
        return outfiles

    def missing_as_string(self):
        outfiles = self._get_files_dct()
        return "\n".join(f'    - {req}: {self.required_output_files[req]}'
                         for req, ext in outfiles.items()
                         if ext is None)

    def check(self):
        """Checks if all required files are present, if not raises CalculationIncompleteError"""

        outfiles = self._get_files_dct()

        error = self.missing_as_string()

        if error != '':
            raise CalculationIncompleteError((f"For folder: '{str(self.folder)}' following "
                                              f" files missing:\n{error}\n\n"))
        return outfiles


def check(calculations):
    """Check multiple calculations, if false Raises CalculationIncompleteError"""
    files = []
    error = ""
    for calc in calculations:
        try:
            files.append(calc.check())
        except CalculationIncompleteError as err:
            error += err.code + "\n\n"
    if error != "":
        raise CalculationIncompleteError(error)
    return files


class CalculationFailed(SystemExit):
    """Error passed if the direct calculation failed"""


def do_xtb(calculation):
    """Perform a xtb calculation, raises CalculationFailed error in case of an error"""
    with calculation.within():
        name = calculation.filename
        try:
            subprocess.run(f"bash {name} > {name}.shellout", shell=True, check=True)
        except subprocess.CalledProcessError as err:
            raise CalculationFailed(f"subprocess registered error '{err.code}'") from None

    try:
        calculation.check()
    except CalculationIncompleteError:
        raise CalculationFailed("Not all necessary files could be generated for calculation"
                                f" '{calculation.inputfile}'"
                                ) from None


def perform_calculations(calculations):

    methods = {'xtb': do_xtb, }

    for calculation in calculations:
        method = methods.get(calculation.software)
        if method is None:
            raise ValueError("Software not suppored")
        try:
            method(calculation)
        except CalculationFailed:
            raise SystemExit("Calculation failed!") from None


class SubmitKeeper:
    """Singleton that keeps track over all calculations that need to be submitted"""

    calculations = []

    @classmethod
    def add(cls, calculation):
        """Add a new Calculation"""
        assert isinstance(calculation, Calculation)
        cls.calculations.append(calculation)

    @classmethod
    def clear(cls, clear_all=False):
        """Remove all calculations from the list

        Args:
            clear_all: bool, optional
                default: False
                True: remove all calculations
                False: remove all incomplete calculations
        """
        if clear_all is False:
            cls.calculations = cls._get_incomplete()
        else:
            cls.calculations = []

    @classmethod
    def do_calculations(cls):
        calculations = cls._get_incomplete()
        perform_calculations(calculations)

    @classmethod
    def check(cls):
        """Check if calculations need to be performed

        Raises:
            CalculationIncompleteError in case some calculations are still missing
        """
        check(cls.calculations)

    @classmethod
    def write_check(cls, filename, *, only_incomplete=True):
        """Write check function to check if calculations are there

        Args:
            filename: str/Path
                name of the python file to be written
            only_incomplete: bool, optional
                default: True
                True: write out only the incomplete calculations
                False: write out all created calculations
        """

        if only_incomplete is False:
            calculations = cls.calculations
        else:
            calculations = cls._get_incomplete()

        calculations = ',\n'.join(calc.as_pycode() for calc in cls.calculations)

        out = f"""from qforce.cli import Calculation, Option


# currently missing calculations
calculations = [{calculations}]


if __name__ == '__main__':
    option = Option.from_commandline(calculations)
    option.run()
        """

        with open(filename, 'w') as fh:
            fh.write(out)

    @classmethod
    def _get_incomplete(cls):
        """Return a list of all incompleted calculations"""
        calculations = []
        for calc in cls.calculations:
            try:
                calc.check()
            except CalculationIncompleteError:
                calculations.append(calc)
        return calculations


def scriptify(writer):
    def wrapper(self, *args, **kwargs):
        pre_input_script, post_input_script = [], []
        job_script = self.config.job_script

        if job_script:
            file = args[0]

            if writer.__name__ == 'write_hessian':
                job_name = f'{self.job.name}_hessian'
            elif writer.__name__ == 'write_scan':
                job_name = args[1]
            elif writer.__name__ == 'write_charge':
                job_name = f'{self.job.name}_hessian_charge'
            else:
                job_name = self.job.name

            job_script = job_script.replace('<jobname>', f'{job_name}')
            job_script = job_script.split('\n')
            if '<input>' in job_script:
                inp_line = job_script.index('<input>')
            else:
                inp_line = len(job_script)

            pre_input_script = job_script[:inp_line]
            post_input_script = job_script[inp_line+1:]

        for line in pre_input_script:
            file.write(f'{line}\n')
        writer(self, *args, **kwargs)
        for line in post_input_script:
            file.write(f'{line}\n')
    return wrapper
