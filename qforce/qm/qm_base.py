import math
import numpy as np
import sys
import inspect
from ase.units import Hartree, mol, kJ, Bohr
from abc import ABC, abstractmethod


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
        elements, coords, hessian = [], [], []
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
    def _read_nbo_analysis(out_file, n_atoms):
        with open(out_file, "r", encoding='utf-8') as file:
            for line in file:
                if "N A T U R A L   B O N D   O R B I T A L" in line:
                    found_wiberg = False
                    lone_e = np.zeros(n_atoms, dtype=int)
                    n_bonds = []
                    b_orders = [[] for _ in range(n_atoms)]
                    while "Calling FoFJK" not in line and "Charge unit " not in line:
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
                                    atom = int(line.split()[5])
                                    occ = int(round(float(line.split()[6]), 0))
                                    if occ > 0:
                                        lone_e[atom-1] += occ
        b_orders = np.round(np.array(b_orders)*2)/2
        return n_bonds, b_orders, lone_e


def scriptify(writer):
    def wrapper(*args, **kwargs):
        pre_input_script, post_input_script = [], []
        job_script = args[0].config.job_script

        if job_script:
            file = args[1]
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


def check_array(name, expected_type, expected_shape):
    private_name = '_' + name

    @property
    def prop(self):
        return getattr(self, private_name)

    @prop.setter
    def prop(self, value):
        value = np.asarray(value)
        shape_input  = [getattr(self, att) for att in inspect.getfullargspec(expected_shape).args]

        if value.size == 0:
            sys.exit(f'No data found in the QM output file for "{name}". Exiting...\n')
        elif value.dtype != np.dtype(expected_type):
            raise TypeError(f'"{name}" expected a type of "{np.dtype(expected_type)}",'
                            f' but got "{value.dtype}"')
        elif value.shape != expected_shape(*shape_input):
            raise ValueError(f'"{name}" expected a shape of "{expected_shape(*shape_input)}",'
                             f' but got "{value.shape}"')
        setattr(self, private_name, value)
    return prop


def check_type(name, expected_type):
    private_name = '_' + name

    @property
    def prop(self):
        return getattr(self, private_name)

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError(f'"{name}" expected a type of "{expected_type}", '
                            f'but got "{value.dtype}"')
        setattr(self, private_name, value)
    return prop


class HessianOutput():
    n_atoms = check_type('n_atoms', int)
    charge = check_type('charge', int)
    multiplicity = check_type('multiplicity', int)
    elements = check_array('elements', int, lambda n_atoms: (n_atoms,))
    coords = check_array('coords', float, lambda n_atoms: (n_atoms, 3))
    hessian = check_array('hessian', float, lambda n_atoms: (((n_atoms*3)**2+n_atoms*3)/2,))
    n_bonds = check_array('n_bonds', int, lambda n_atoms: (n_atoms,))
    b_orders = check_array('b_orders', float, lambda n_atoms: (n_atoms, n_atoms))
    lone_e = check_array('lone_e', int, lambda n_atoms: (n_atoms,))
    point_charges = check_array('point_charges', float, lambda n_atoms: (n_atoms,))

    def __init__(self, n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds, b_orders,
                 lone_e, point_charges):
        self.n_atoms = n_atoms
        self.charge = charge
        self.multiplicity = multiplicity
        self.elements = elements
        self.coords = coords
        self.hessian = hessian
        self.n_bonds = n_bonds
        self.b_orders = b_orders
        self.lone_e = lone_e
        self.point_charges = point_charges


class ScanOutput():
    n_atoms = check_type('n_atoms', int)
    n_steps = check_type('n_steps', int)
    coords = check_array('coords', float, lambda n_atoms, n_steps: (n_steps, n_atoms, 3))
    angles = check_array('angles', float, lambda n_steps: (n_steps,))
    energies = check_array('energies', float, lambda n_steps: (n_steps,))

    def __init__(self, n_atoms, n_steps, coords, angles, energies):
        angles = (np.asarray(angles) % 360).round(4)
        order = np.argsort(angles)
        angles = angles[order]
        coords = np.asarray(coords)[order]
        energies = np.asarray(energies)[order]
        energies -= energies.min()

        self.n_atoms = n_atoms
        self.n_steps = n_steps
        self.coords = coords
        self.angles = angles
        self.energies = energies
