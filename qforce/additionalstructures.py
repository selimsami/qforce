import os
from shutil import copy2 as copy
from itertools import chain
#
from colt import Colt
from ase.io import read
from calkeeper import CalculationIncompleteError


class AdditionalStructureCreator(Colt):

    name = None
    __classes = {}

    def __init__(self, weight):
        self.weight = weight

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__classes[cls.name] = cls
        cls._user_input += ("\n# Weight of the structures in the forcefield fit"
                            "\n weight = 1 :: int :: >1 \n")

    @classmethod
    def classes(cls):
        return cls.__classes

    @classmethod
    def get(cls, name):
        return cls.__classes.get(name)

    def create(self, qm):
        raise NotImplementedError

    def check(self):
        raise NotImplementedError

    def parse(self, qm):
        raise NotImplementedError


class StructuresFromFile(AdditionalStructureCreator):

    name = 'fromfile'

    _user_input = """
        en_struct = :: existing_file, optional
        grad_struct = :: existing_file, optional
        hess_struct = :: existing_file, optional
    """

    def __init__(self, weight, en_struct, grad_struct, hess_struct):
        self.weight = weight
        self._data = {
                'en': {
                       'file': en_struct,
                       'calculations': [],
                       'results': [],
                      },
                'grad': {
                         'file': grad_struct,
                         'calculations': [],
                         'results': [],
                        },
                'hess': {
                         'file': hess_struct,
                         'calculations': [],
                         'results': [],
                        },
        }

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

    def create(self, parent, qm):
        en = self._data['en']
        grad = self._data['grad']
        hess = self._data['hess']

        if en['file'] is not None:
            folder = parent / 'en_structs'
            os.makedirs(folder, exist_ok=True)
            en['calculations'] = qm.do_sp_calculations(folder, self._fileiter(en['file']))
        if grad['file'] is not None:
            folder = parent / 'grad_structs'
            os.makedirs(folder, exist_ok=True)
            grad['calculations'] = qm.do_grad_calculations(folder, self._fileiter(grad['file']))
        if hess['file'] is not None:
            folder = parent / 'hess_structs'
            os.makedirs(folder, exist_ok=True)
            hess['calculations'] = qm.do_hessian_calculations(folder, self._fileiter(hess['file']))

    def check(self):
        en = self._data['en']
        grad = self._data['grad']
        hess = self._data['hess']
        for calculation in chain(en['calculations'], grad['calculations'], hess['calculations']):
            try:
                _ = calculation.check()
            except CalculationIncompleteError:
                return calculation
        return None

    def parse(self, qm):
        en = self._data['en']
        grad = self._data['grad']
        hess = self._data['hess']

        results = []
        for calculation in hess['calculations']:
            hessian_files = calculation.check()
            results.append(qm._read_hessian(hessian_files))

        en_results = []
        for calculation in en['calculations']:
            files = calculation.check()
            en_results.append(qm._read_energy(files))

        grad_results = []
        for calculation in grad['calculations']:
            files = calculation.check()
            grad_results.append(qm._read_gradient(files))

        en['results'] = en_results
        grad['results'] = grad_results
        hess['results'] = results

    def enouts(self):
        return self._data['en']['results']

    def gradouts(self):
        return self._data['grad']['results']

    def hessouts(self):
        return self._data['hess']['results']


class MetaDynamics(AdditionalStructureCreator):

    name = 'metadynamics'

    _user_input = """
    compute = none :: str :: [none, en, grad]

    # temperature in Kelvin
    temp = 1000 :: float :: >0
    # total simulation time in ps
    time = 5.0 :: float :: >0
    # interval for trajectory printout in fs
    dump = 5.0 :: float :: >0
    # time step for propagation in fs
    step = 2.0 :: float :: >0
    """

    def __init__(self, weight, compute, config):
        self.compute = compute
        self.weight = weight
        self.xtbinput = {key: value for key, value in config.items()
                         if key not in ('weight', 'compute')}
        self._data = {
                'en':  {'calculations': [], 'results': []},
                'grad':  {'calculations': [], 'results': []},
                'hess':  {'calculations': [], 'results': []},
        }

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

    def create(self, parent, qm):
        folder = parent / '0_md'
        os.makedirs(folder, exist_ok=True)

        calc = qm.xtb_md(folder, self.xtbinput)

        try:
            files = calc.check()
        except CalculationIncompleteError:
            qm.logger.exit(f"Required xtb trajectory file not found in '{folder}'.\n"
                           'Creating the necessary input file and exiting...\n'
                           )

        traj = files['traj']
        filename = parent / 'xtbmd.xyz'
        copy(traj, filename)

        traj = self._fileiter(filename)

        if self.compute == 'en':
            folder = parent / '1_en_structs'
            os.makedirs(folder, exist_ok=True)
            self._data['en']['calculations'] = qm.do_sp_calculations(folder, traj)
        elif self.compute == 'grad':
            folder = parent / '1_grad_structs'
            os.makedirs(folder, exist_ok=True)
            self._data['grad']['calculations'] = qm.do_grad_calculations(folder, traj)
        else:
            raise ValueError("do not know compute method!")

    def check(self):
        en = self._data['en']
        grad = self._data['grad']
        for calculation in chain(en['calculations'], grad['calculations']):
            try:
                _ = calculation.check()
            except CalculationIncompleteError:
                return calculation
        return None

    def parse(self, qm):
        en = self._data['en']
        grad = self._data['grad']

        en_results = []
        for calculation in en['calculations']:
            files = calculation.check()
            en_results.append(qm._read_energy(files))

        grad_results = []
        for calculation in grad['calculations']:
            files = calculation.check()
            grad_results.append(qm._read_gradient(files))

        en['results'] = en_results
        grad['results'] = grad_results

    def enouts(self):
        return self._data['en']['results']

    def gradouts(self):
        return self._data['grad']['results']

    def hessouts(self):
        return self._data['hess']['results']


class HessianOutput:

    def __init__(self, weight, hessout):
        self.weight = weight
        if not isinstance(hessout, (tuple, list)):
            hessout = [hessout]
        self.hessout = hessout

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return self.hessout

    def create(self, parent, qm):
        pass

    def parse(self, qm):
        pass

    def check(self):
        pass


class DihedralOutput:

    def __init__(self, weight, gradouts):
        self.weight = weight
        self._gradouts = gradouts

    def enouts(self):
        return []

    def gradouts(self):
        return self._gradouts

    def create(self, parent, qm):
        pass

    def parse(self, qm):
        pass

    def check(self):
        pass

    def hessouts(self):
        return []


def minval(itr, value=None):
    try:
        return min(itr)
    except ValueError:
        return value


class AdditionalStructures(Colt):

    _user_input = """
    energy_element_weights = 1 :: int :: >1
    gradient_element_weights = 1 :: int :: >1
    hessian_element_weights = 1 :: int :: >1

    hessian_weight = 1 :: int :: >1
    dihedral_weight = 1 :: int :: >1

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
        self.creators['hessian'] = HessianOutput(self._hessian_weight, hessout)

    def add_dihedrals(self, scans):
        self.creators['dihedrals'] = DihedralOutput(self._dihedral_weight, scans)

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
            creator.create(folder, qm)

        for name, creator in self.creators.items():
            cal = creator.check()
            if cal is not None:
                qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                               'Creating the necessary input file and exiting...\nPlease run the '
                               'calculation and put the output files in the same directory.\n'
                               'Necessary output files and the corresponding extensions '
                               f"are:\n{cal.missing_as_string()}\n\n\n")

        for name, creator in self.creators.items():
            creator.parse(qm)

    def lowest_energy(self, only_hessian=True):
        minimum = 10000.0
        if only_hessian is True:
            for creator in self.creators.values():
                min_h = minval((out.energy for out in creator.hessouts()), minimum)
                if min_h < minimum:
                    minimum = min_h
            return minimum

        for creator in self.creators.values():
            min_e = minval((out.energy for out in creator.enouts()), minimum)
            if min_e < minimum:
                minimum = min_e
            min_g = minval((out.energy for out in creator.gradouts()), minimum)
            if min_g < minimum:
                minimum = min_g
            min_h = minval((out.energy for out in creator.hessouts()), minimum)
            if min_h < minimum:
                minimum = min_h
        return minimum

    def normalize(self):
        emin = self.lowest_energy(only_hessian=True)
        self.subtract_energy(emin)

    def subtract_energy(self, energy):
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
