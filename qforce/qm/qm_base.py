import math
import numpy as np
import sys
from ase.units import Hartree, mol, kJ, Bohr
from abc import ABC, abstractmethod
from warnings import warn


class WriteABC(ABC):
    @abstractmethod
    def hessian(self, ):
        ...

    @abstractmethod
    def scan(self, ):
        ...


class ReadABC(ABC):

    @abstractmethod
    def hessian(self, ):
        ...

    @abstractmethod
    def scan(self, ):
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


def scriptify(writer):
    def wrapper(*args, **kwargs):
        pre_input_script, post_input_script = [], []
        job_script = args[0].config.job_script

        if job_script:
            file = args[1]

            if writer.__name__ == 'write_hessian':
                job_name = f'{args[0].job.name}_hessian'
            elif writer.__name__ == 'write_scan':
                job_name = args[2]
            else:
                job_name = args[0].job.name

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
        writer(*args, **kwargs)
        for line in post_input_script:
            file.write(f'{line}\n')
    return wrapper


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
