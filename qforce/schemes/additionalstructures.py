import os
from shutil import copy2 as copy
from itertools import chain
#
from ase.io import read
from calkeeper import CalculationIncompleteError
from .creator import CalculationStorage, CostumStructureCreator


class AdditionalStructureCreator(CostumStructureCreator):

    name = None
    __classes = {}
    __predefined = ['dihedrals', 'hessian']
    _user_input = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name in cls.__predefined:
            raise ValueError(f"Cannot create class with name {cls.name}")
        cls.__classes[cls.name] = cls
        cls._user_input += ("\n# Weight of the structures in the forcefield fit"
                            "\n weight = 1 :: float \n")

    @classmethod
    def classes(cls):
        return cls.__classes

    @classmethod
    def get(cls, name):
        return cls.__classes.get(name)

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


class StructuresFromFile(AdditionalStructureCreator):

    name = 'fromfile'

    _user_input = """
        en_struct = :: existing_file, optional
        grad_struct = :: existing_file, optional
        hess_struct = :: existing_file, optional
    """

    def __init__(self, weight, en_struct, grad_struct, hess_struct):
        super().__init__(weight)
        self._energy = CalculationStorage(file=en_struct)
        self._grad = CalculationStorage(file=grad_struct)
        self._hess = CalculationStorage(file=hess_struct)

    @classmethod
    def from_config(cls, config):
        if not config['en_struct'] and not config['grad_struct'] and not config['hess_struct']:
            return None
        return cls(config['weight'], config['en_struct'], config['grad_struct'],
                   config['hess_struct'])

    @staticmethod
    def _fileiter(filename):
        molecules = read(filename, index=':')
        for i, molecule in enumerate(molecules):
            yield i, (molecule.get_positions(), molecule.get_atomic_numbers())

    def setup_main(self, qm):
        parent = self.folder

        if self._energy['file'] is not None:
            folder = parent / 'en_structs'
            os.makedirs(folder, exist_ok=True)
            self._energy.calculations = qm.do_sp_calculations(folder,
                                                              self._fileiter(self._energy['file']))
        if self._grad['file'] is not None:
            folder = parent / 'grad_structs'
            os.makedirs(folder, exist_ok=True)
            self._grad.calculations = qm.do_grad_calculations(folder,
                                                              self._fileiter(self._grad['file']))
        if self._hess['file'] is not None:
            folder = parent / 'hess_structs'
            os.makedirs(folder, exist_ok=True)
            self._hess.calculations = qm.do_hessian_calculations(folder,
                                                                 self._fileiter(self._hess['file']))

    def check_main(self):
        for calculation in chain(self._energy.calculations, self._grad.calculations,
                                 self._hess.calculations):
            try:
                _ = calculation.check()
            except CalculationIncompleteError:
                return calculation
        return None

    def parse_main(self, qm):
        results = []
        for calculation in self._hess.calculations:
            hessian_files = calculation.check()
            results.append(qm.read_hessian(hessian_files))

        en_results = []
        for calculation in self._energy.calculations:
            files = calculation.check()
            en_results.append(qm.read_energy(files))

        grad_results = []
        for calculation in self._grad.calculations:
            files = calculation.check()
            grad_results.append(qm.read_gradient(files))

        self._energy.results = en_results
        self._grad.results = grad_results
        self._hess.results = results

    def enouts(self):
        return self._energy.results

    def gradouts(self):
        return self._grad.results

    def hessouts(self):
        return self._hess.results


class MetaDynamics(AdditionalStructureCreator):

    name = 'metadynamics'

    _user_input = """
    # What to compute:
    #   none: do not do metadynamics, default
    #     en: do metadynamics and do single point energy calculations
    #   grad: do metadynamics and do single point energy+ gradient calculations
    #
    compute = none :: str :: [none, en, grad]
    # Number of frames used for fitting
    n_fit = 100 :: int
    # Number of frames used for validation
    n_valid = 0 :: int
    # temperature (K)
    temp = 800 :: float :: >0
    # interval for trajectory printout (fs)
    dump = 500.0 :: float :: >0
    # time step for propagation (fs)
    step = 2.0 :: float :: >0
    # Bond constraints (0: none, 1: H-only, 2: all)
    shake = 0
    """

    def __init__(self, weight, compute, config):
        self.compute = compute
        self.weight = weight
        self.n_fit = config['n_fit']
        self.n_valid = config['n_valid']
        self.total_frames = self.n_fit + self.n_valid

        time = config['dump'] * self.total_frames / 1e3

        self.xtbinput = {'time': time}
        for item in ['temp', 'dump', 'step', 'shake']:
            self.xtbinput[item] = config[item]

        self._energy = CalculationStorage()
        self._grad = CalculationStorage()
        # not used so far
        self._hess = CalculationStorage()
        self._md = CalculationStorage()

    @classmethod
    def from_config(cls, config):
        if config['compute'] == 'none':
            return None
        return cls(config['weight'], config['compute'], config)

    @staticmethod
    def _fileiter(filename):
        molecules = read(filename, index=':')
        for i, molecule in enumerate(molecules):
            yield i, (molecule.get_positions(), molecule.get_atomic_numbers())

    def setup_pre(self, qm):
        """setup md calculation"""
        folder = self.folder / '0_md'
        os.makedirs(folder, exist_ok=True)

        # setup
        initfile = qm.pathways.getfile('init.xyz')
        copy(initfile, folder / 'xtb.xyz')
        with open(folder / 'md.inp', 'w') as fh:
            fh.write("$md\n")
            for key, value in self.xtbinput.items():
                fh.write(f"{key} = {value}\n")
            fh.write("$end\n")

        with open(folder / 'xtb.inp', 'w') as fh:
            fh.write("xtb xtb.xyz --input md.inp --md")

        #
        calc = self.Calculation('xtb.inp',
                                {'traj': ['xtb.trj']},
                                folder=folder,
                                software='xtb')
        self._md.calculations = [calc]

    def check_pre(self):
        """check that the md calculation was finished"""
        for calc in self._md.calculations:
            try:
                _ = calc.check()
            except CalculationIncompleteError:
                return calc
        return None

    def parse_pre(self, qm):
        files = self._md.calculations[0].check()
        traj = files['traj']
        filename = self.folder / 'xtbmd.xyz'
        copy(traj, filename)
        self._md.results = [{'file': filename}]

    def setup_main(self, qm):
        # setupt
        parent = self.folder
        # currently only one result there
        filename = self._md.results[0]['file']

        traj = self._fileiter(filename)

        if self.compute == 'en':
            folder = parent / '1_en_structs'
            os.makedirs(folder, exist_ok=True)
            self._energy.calculations = qm.do_sp_calculations(folder, traj)
        elif self.compute == 'grad':
            folder = parent / '1_grad_structs'
            os.makedirs(folder, exist_ok=True)
            self._grad.calculations = qm.do_grad_calculations(folder, traj)
        else:
            raise ValueError("do not know compute method!")

    def check_main(self):
        for calculation in chain(self._energy.calculations, self._grad.calculations):
            try:
                _ = calculation.check()
            except CalculationIncompleteError:
                return calculation
        return None

    def parse_main(self, qm):
        en_results = []
        for i, calculation in enumerate(self._energy.calculations):
            files = calculation.check()
            if i < self.n_fit:
                en_results.append(qm.read_energy(files))

        grad_results = []
        for i, calculation in enumerate(self._grad.calculations):
            files = calculation.check()
            if i < self.n_fit:
                grad_results.append(qm.read_gradient(files))

        self._energy.results = en_results
        self._grad.results = grad_results

    def enouts(self):
        return self._energy.results

    def gradouts(self):
        return self._grad.results

    def hessouts(self):
        return self._hess.results
