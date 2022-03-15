from .gaussian import Gaussian, ReadGaussian, WriteGaussian


class WriteXTBGaussian(WriteGaussian):

    @staticmethod
    def write_method(file, config):
        file.write(f'external="gauext-xtb " ')

    @staticmethod
    def write_pop(file, string):
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
        return self._read_charges(file, n_atoms)

    @classmethod
    def _read_cm5_charges(cls, file, n_atoms):
        return self._read_charges(file, n_atoms)


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
