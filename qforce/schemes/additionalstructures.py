from ase.io import read
#
from .creator import CustomStructureCreator
from .generalcreators import EnergyCalculationIterCreator, GradientCalculationIterCreator
from .generalcreators import HessianCalculationIterCreator
from .helper import coords_ids_iter


class StructuresFromFile(CustomStructureCreator):

    _user_input = """
        weight = 1 :: float
        en_struct = :: existing_file, optional
        grad_struct = :: existing_file, optional
        hess_struct = :: existing_file, optional
    """

    def __init__(self, weight, folder, en_struct, grad_struct, hess_struct):
        super().__init__(weight, folder=folder)
        self._energy = EnergyCalculationIterCreator(folder / 'en_structs', weight,
                                                    coords_ids_iter(en_struct))
        self._grad = GradientCalculationIterCreator(folder / 'grad_structs', weight,
                                                    coords_ids_iter(grad_struct))
        self._hess = HessianCalculationIterCreator(folder / 'hess_struct', weight,
                                                   coords_ids_iter(hess_struct))

    @classmethod
    def from_config(cls, config, folder):
        if not config['en_struct'] and not config['grad_struct'] and not config['hess_struct']:
            return None
        return cls(config['weight'], folder, config['en_struct'], config['grad_struct'],
                   config['hess_struct'])

    @staticmethod
    def _fileiter(filename):
        if filename is None:
            return
        molecules = read(filename, index=':')
        for molecule in molecules:
            yield molecule.get_positions(), molecule.get_atomic_numbers()

    def setup_main(self, qm):
        self._energy.setup_main(qm)
        self._grad.setup_main(qm)
        self._hess.setup_main(qm)

    def check_main(self):
        for obj in (self._energy, self._grad, self._hess):
            calc = obj.check_main()
            if calc is not None:
                return calc
        return None

    def parse_main(self, qm):
        self._energy.parse_main(qm)
        self._grad.parse_main(qm)
        self._hess.parse_main(qm)

    def enouts(self):
        return self._energy.enouts()

    def gradouts(self):
        return self._grad.gradouts()

    def hessouts(self):
        return self._hess.hessouts()
