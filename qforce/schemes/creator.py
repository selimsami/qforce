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


class CostumStructureCreator(Colt):
    """Creator class to generate structures for the fitting procedure
    Basic idea to call the creator in this way:

    pre: optional
    main: main step, needs to be implemented

    """

    def __init__(self, weight):
        self.weight = weight
        # will be overwritten!
        self._folder = None

    @property
    def folder(self):
        if self._folder is None:
            raise ValueError("Please setup main folder")
        return self._folder

    @folder.setter
    def folder(self, value):
        self._folder = value

    def run(self, qm):
        self.setup_pre(qm)
        self.check_pre()
        self.parse_pre(qm)
        #
        self.setup_main(qm)
        self.check_main()
        self.parse_main(qm)
        #
        self.setup_post(qm)
        self.check_post()
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
