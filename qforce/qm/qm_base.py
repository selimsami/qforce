import math
import sys
import json
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from warnings import warn
import hashlib

import numpy as np
from ase.units import Hartree, mol, kJ, Bohr
from colt import Colt


class WriteABC(ABC):

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


class ReadABC(ABC):

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
                    for i in range(n_line):
                        line = file.readline()
                        ids = [int(i) for i in line.split()]
                        elements.extend(ids)
                if "Current cartesian coordinates   " in line:
                    n_line = math.ceil(3*n_atoms/5)
                    for i in range(n_line):
                        line = file.readline()
                        coords.extend(line.split())
                if "Cartesian Force Constants  " in line:
                    n_line = math.ceil(3*n_atoms*(3*n_atoms+1)/10)
                    for i in range(n_line):
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


class QMInterface(Colt):
    """Basic Logic for an QM Interface"""

    def __init__(self, config, read, write):
        self.read = read
        # get required files
        self.required_hessian_files = read.hessian_files
        self.required_opt_files = read.opt_files
        self.required_sp_files = read.sp_files
        self.required_charge_files = read.charge_files
        #
        self.write = write
        self.config = config

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


class ScanOutput():
    def __init__(self, file,  n_steps, n_atoms, coords, angles, energies, charges):
        self.n_atoms = n_atoms
        self.n_steps = n_steps
        angles, energies, coords, self.charges, self.mismatch = self.check_shape(angles, energies,
                                                                                 coords, charges,
                                                                                 file)
        self.angles, self.energies, self.coords = self._rearrange(angles, energies, coords)

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
    pass


class Calculation:
    """Hold information of a calculation"""

    def __init__(self, inputfile, required_output_files, *, folder=None):
        self.folder = Path(folder) if folder is not None else Path("")
        self.required_output_files = required_output_files
        self.inputfile = self.folder / Path(inputfile)
        self.base = self.inputfile.name[:-len(self.inputfile.suffix)]
        # register itself
        SubmitKeeper.add(self)

    def _render(self, option):
        if '$' in option:
            option = Template(option)
            return option.substitute(base=self.base)
        return self.base + option

    def as_pycode(self):
        return (f'Calculation("{self.inputfile.name}", '
                f'{json.dumps(self.required_output_files)}, '
                f'folder="{str(self.folder)}")')

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
    for i, calc in enumerate(calculations):
        try:
            files.append(calc.check())
        except CalculationIncompleteError as e:
            error += e.code + "\n\n"
    if error != "":
        raise CalculationIncompleteError(error)
    return files


class SubmitKeeper:
    """Singleton that keeps track over all calculations that need to be submitted"""

    calculations = []

    @classmethod
    def add(self, calculation):
        assert isinstance(calculation, Calculation)
        self.calculations.append(calculation)

    @classmethod
    def clear(self):
        self.calculations = []

    @classmethod
    def check(self):
        check(self.calculations)

    @classmethod
    def write_check(self, filename, only_incomplete=False):
        """Write check function to check if calculations are there"""

        if only_incomplete is False:
            calculations = ',\n'.join(calc.as_pycode() for calc in self.calculations)
        else:
            calculations = []
            for calc in self.calculations:
                try:
                    calc.check()
                except CalculationIncompleteError:
                    calculations.append(calc)
            calculations = ',\n'.join(calc.as_pycode() for calc in calculations)

        out = f"""from qforce.cli import Calculation, Option


# currently missing calculations
calculations = [{calculations}]


if __name__ == '__main__':
    option = Option.from_commandline(calculations)
    option.run()
        """

        with open(filename, 'w') as fh:
            fh.write(out)


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
