from .gaussian import Gaussian, ReadGaussian, WriteGaussian


class WriteXTBGaussian(WriteGaussian):

    @staticmethod
    def write_method(file, config):
        file.write(f'external="gauext-xtb " ')

    @staticmethod
    def write_pop(file, string):
        pass


class XTB(Gaussian):

    _user_input = """

    charge_method = cm5 :: str :: [cm5, esp]

    # QM method to be used
    method = gnf2 :: str :: gnf2

    """

    def __init__(self):
        self.required_hessian_files = {'out_file': ['.out', '.log'],
                                       'fchk_file': ['.fchk', '.fck']}
        self.read = ReadGaussian
        self.write = WriteXTBGaussian
