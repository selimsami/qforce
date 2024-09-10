import os
#
from colt import Colt
from ase.io import read
from calkeeper import check, CalculationIncompleteError


class AdditionalStructureCreator(Colt):

    __classes = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__classes[cls.name] = cls

    @classmethod
    def classes(cls):
        return cls.__classes

    @classmethod
    def get(cls, name):
        return cls.__classes.get(name)

    def run(self, qm):
        raise NotImplementedError


class StructuresFromFile(AdditionalStructureCreator):

    name = 'fromfile'

    _user_input = """
        en_struct = :: existing_file, optional
        grad_struct = :: existing_file, optional
        hess_struct = :: existing_file, optional
    """

    def __init__(self, en_struct, grad_struct, hess_struct):
        self._data = {
                'en':  {'file': en_struct,
                        'calculations': [],
                        'results': [],
                        'weight': 1,
                        },
                'grad':  {'file': grad_struct,
                        'calculations': [],
                        'results': [],
                        'weight': 1,
                        },
                'hess':  {'file': hess_struct,
                        'calculations': [],
                        'results': [],
                        'weight': 1,
                        },
        }


    @classmethod
    def from_config(cls, config):
        if config['en_struct'] is None and config['grad_struct'] is None and config['hess_struct'] is None:
            return None
        return cls(config['en_struct'], config['grad_struct'], config['hess_struct'])

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
        for calculation in (en['calculations'] + grad['calculations'] + hess['calculations']):
            try:
                hessian_files = calculation.check()
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


class AdditionalStructures(Colt):

    def __init__(self, creators):
        self.creators = creators
    
    @classmethod
    def from_config(cls, config):
        creators = {}
        for name, clss in AdditionalStructureCreator.classes().items():
            _cls = clss.from_config(getattr(config,name))
            if _cls is not None:
                creators[name] = _cls
        return cls(creators)

    @classmethod
    def _extend_user_input(cls, questions):
        for name, clss in AdditionalStructureCreator.classes().items():
            questions.generate_block(name, clss.colt_user_input)

    def create(self, qm):
        parent = qm.pathways.getdir("addstruct", create=True)

        for name, creator in self.creators.items():
            creator.create(parent, qm)

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
