import os
from shutil import copy2 as copy
from itertools import chain
import numpy as np
#
from colt import Colt
from ase.io import read
from calkeeper import CalculationIncompleteError


class CalculationStorage:
    """Basic class to store calculations and calculation results"""

    def __init__(self, calculations=None, results=None, **metadata):
        if calculations is None:
            calculations = []
        self.calculations = calculations
        if results is None:
            results = []
        self.results = results
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

    def setup_pre(self, qm):
        pass

    def check_pre(self):
        pass

    def parse_pre(self, qm):
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
        self._md.calculations = [qm.xtb_md(folder, self.xtbinput)]

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


class HessianOutput(CostumStructureCreator):

    def __init__(self, weight, hessout):
        super().__init__(weight)
        if not isinstance(hessout, (tuple, list)):
            hessout = [hessout]
        self._hessout = hessout

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return self._hessout

    def setup_main(self, qm):
        pass

    def check_main(self):
        pass

    def parse_main(self, qm):
        pass


class DihedralOutput(CostumStructureCreator):

    def __init__(self, weight, gradouts):
        super().__init__(weight)
        self._gradouts = gradouts

    def enouts(self):
        return []

    def gradouts(self):
        return self._gradouts

    def hessouts(self):
        return []

    def setup_main(self, qm):
        pass

    def check_main(self):
        pass

    def parse_main(self, qm):
        pass

def minval(itr, value=None):
    try:
        return min(itr)
    except ValueError:
        return value


class AdditionalStructures(Colt):

    _user_input = """
    energy_element_weights = 1 :: float
    gradient_element_weights = 1 :: float
    hessian_element_weights = 1 :: float

    hessian_weight = 1 :: float
    dihedral_weight = 1 :: float

    """

    def __init__(self, creators, energy_ele_weight, gradient_ele_weight, hessian_ele_weight,
                 hessian_weight, dihedral_weight):
        self.creators = creators
        self.energy_weight = energy_ele_weight
        self.gradient_weight = gradient_ele_weight
        self.hessian_weight = hessian_ele_weight
        #
        self._hessian_weight = hessian_weight
        self._dihedral_weight = dihedral_weight

    def add_hessians(self, hessout):
        """add hessian"""
        self.creators['hessian'] = HessianOutput(self._hessian_weight, hessout)

    def add_dihedrals(self, scans):
        """add computed dihedrals"""
        self.creators['dihedrals'] = DihedralOutput(self._dihedral_weight, scans)

    def register(self, name, creator):
        """add a new creator"""
        if not isinstance(creator, CostumStructureCreator):
            raise ValueError(f"Can not register '{type(creator)}' "
                             "only register CostumStructureCreator instances")
        if name == 'dihedrals':
            creator.weight = self._dihedral_weight
        elif name == 'hessian':
            creator.weight = self._hessian_weight
        self.creators[name] = creator

    @classmethod
    def from_config(cls, config):
        creators = {}
        for name, clss in AdditionalStructureCreator.classes().items():
            _cls = clss.from_config(getattr(config, name))
            if _cls is not None:
                creators[name] = _cls
        return cls(creators, config.energy_element_weights,
                   config.gradient_element_weights, config.hessian_element_weights,
                   config.hessian_weight, config.dihedral_weight)

    @classmethod
    def _extend_user_input(cls, questions):
        for name, clss in AdditionalStructureCreator.classes().items():
            questions.generate_block(name, clss.colt_user_input)

    def create(self, qm):
        parent = qm.pathways.getdir("addstruct", create=True)

        for i, (name, creator) in enumerate(self.creators.items()):
            folder = parent / f'{i}_{creator.name}'
            os.makedirs(folder, exist_ok=True)
            creator.folder = folder

        # prestep
        for name, creator in self.creators.items():
            creator.setup_pre(qm)

        for name, creator in self.creators.items():
            cal = creator.check(prestep=False)
            if cal is not None:
                qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                               'Creating the necessary input file and exiting...\nPlease run the '
                               'calculation and put the output files in the same directory.\n'
                               'Necessary output files and the corresponding extensions '
                               f"are:\n{cal.missing_as_string()}\n\n\n")

        for name, creator in self.creators.items():
            creator.parse_pre(qm)

        # mainstep
        for name, creator in self.creators.items():
            creator.setup_main(qm)

        for name, creator in self.creators.items():
            cal = creator.check_main()
            if cal is not None:
                qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                               'Creating the necessary input file and exiting...\nPlease run the '
                               'calculation and put the output files in the same directory.\n'
                               'Necessary output files and the corresponding extensions '
                               f"are:\n{cal.missing_as_string()}\n\n\n")

        for name, creator in self.creators.items():
            creator.parse_main(qm)

    def _get_lowest_energy_and_coords(self):
        minimum = np.inf
        coords = None

        for creator in self.creators.values():
            for calctype in [creator.enouts(), creator.gradouts(), creator.hessouts()]:
                if calctype:
                    argmin = np.argmin((out.energy for out in calctype))
                    lowest = calctype[argmin].energy
                    if lowest < minimum:
                        minimum = lowest
                        coords = calctype[argmin].coords
        return minimum, coords

    def normalize(self):
        """set all energies to the minimum one"""
        emin, coords = self._get_lowest_energy_and_coords()
        self.subtract_energy(emin)
        return emin, coords

    def subtract_energy(self, energy):
        """subtract energy from all qm energies, forces and hessian are not affected"""
        for creator in self.creators.values():
            for out in creator.enouts():
                out.energy -= energy
            for out in creator.gradouts():
                out.energy -= energy
            for out in creator.hessouts():
                out.energy -= energy

    def enitr(self):
        for creator in self.creators.values():
            weight = creator.weight * self.energy_weight
            for out in creator.enouts():
                yield weight, out

    def graditr(self):
        for creator in self.creators.values():
            weight = creator.weight * self.gradient_weight
            for out in creator.gradouts():
                yield weight, out

    def hessitr(self):
        for creator in self.creators.values():
            weight = creator.weight * self.hessian_weight
            for out in creator.hessouts():
                yield weight, out
