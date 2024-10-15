from calkeeper import CalculationIncompleteError
from colt import Colt


class CalculationStorage:
    """Basic class to store calculations and calculation results"""

    def __init__(self, calculations=None, results=None, as_dict=False, **metadata):
        if as_dict is True:
            container = dict
        else:
            container = list
        #
        if calculations is None:
            calculations = container()
        self.calculations = calculations
        #
        if results is None:
            results = container()
        self.results = results
        #
        self._meta = metadata

    def get(self, arg, default=None):
        return self._meta.get(arg, default)

    def __getitem__(self, arg):
        return self._meta[arg]


class CustomStructureCreator(Colt):
    """Creator class to generate structures for the fitting procedure
    Basic idea to call the creator in this way:
    """

    def __init__(self, weight, folder=None):
        self.weight = weight
        self._folder = folder

    @property
    def folder(self):
        if self._folder is None:
            raise ValueError("Please setup main folder")
        return self._folder

    @staticmethod
    def _check(calculations):
        for calc in calculations:
            try:
                _ = calc.check()
            except CalculationIncompleteError:
                return calc
        return None

    @folder.setter
    def folder(self, value):
        self._folder = value

    def run(self, qm):
        # pre calculations
        self.setup_pre(qm)
        cal = self.check_pre()
        if cal is not None:
            qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                           'Creating the necessary input file and exiting...\nPlease run the '
                           'calculation and put the output files in the same directory.\n'
                           'Necessary output files and the corresponding extensions '
                           f"are:\n{cal.missing_as_string()}\n\n\n")
        self.parse_pre(qm)
        # main calculations
        self.setup_main(qm)
        cal = self.check_main()
        if cal is not None:
            qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                           'Creating the necessary input file and exiting...\nPlease run the '
                           'calculation and put the output files in the same directory.\n'
                           'Necessary output files and the corresponding extensions '
                           f"are:\n{cal.missing_as_string()}\n\n\n")
        self.parse_main(qm)
        # post calculations
        self.setup_post(qm)
        cal = self.check_post()
        if cal is not None:
            qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                           'Creating the necessary input file and exiting...\nPlease run the '
                           'calculation and put the output files in the same directory.\n'
                           'Necessary output files and the corresponding extensions '
                           f"are:\n{cal.missing_as_string()}\n\n\n")
        self.parse_post(qm)

    def setup_pre(self, qm):
        pass

    def check_pre(self):
        pass

    def parse_pre(self, qm):
        pass

    def setup_post(self, qm):
        pass

    def check_post(self):
        pass

    def parse_post(self, qm):
        pass

    def setup_main(self, qm):
        raise NotImplementedError

    def check_main(self):
        raise NotImplementedError

    def parse_main(self, qm):
        raise NotImplementedError

    def enouts(self):
        raise NotImplementedError

    def gradouts(self):
        raise NotImplementedError

    def hessouts(self):
        raise NotImplementedError
