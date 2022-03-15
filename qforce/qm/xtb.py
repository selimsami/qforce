import numpy as np
#
from .gaussian import Gaussian, ReadGaussian, WriteGaussian


class WriteXTBGaussian(WriteGaussian):

    @staticmethod
    def write_method(file, config):
        file.write(f'external="gauext-xtb " ')

    @staticmethod
    def write_pop(file, string):
        pass

    @staticmethod
    def _write_bndix(file):
        pass


class ReadXTBGaussian(ReadGaussian):

    @staticmethod
    def _read_charges(file, n_atoms):
        point_charges = []
        for i in range(n_atoms):
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
            print(line)
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

class XTB(Gaussian):

    _user_input = """

    charge_method = cm5 :: str :: [cm5, esp]

    # QM method to be used
    method = gnf2 :: str :: gnf2

    """

    def __init__(self):
        self.required_hessian_files = {'out_file': ['.out', '.log'],
                                       'fchk_file': ['.fchk', '.fck']}
        self.read = ReadXTBGaussian
        self.write = WriteXTBGaussian
