from copy import deepcopy
from calkeeper import CalculationIncompleteError
#
from .creator import CustomStructureCreator, CalculationStorage


class OptimizationOutput:

    def __init__(self, mol, coords):
        self._molecule = mol
        self.coords = coords[0]
        self._molecule.set_positions(self.coords)
        self._all = coords

    @property
    def molecule(self):
        return self._molecule

    def all_molecules(self):
        molecules = []
        for coords in self._all_coords:
            mol = deepcopy(self._molecule)
            mol.set_positions(coords)
            molecules.append(mol)
        return molecules


class PreoptCreator(CustomStructureCreator):

    def __init__(self, molecule):
        super().__init__(0)
        self._init_molecule = molecule
        self.atnums = molecule.get_atomic_numbers()
        self._init_coords = molecule.get_positions()
        self._preopt = CalculationStorage()

    def coords(self):
        """get current coordinates"""
        if len(self._preopt.results) == 0:
            return self._init_coords
        return self._preopt.results[0].coords

    def molecule(self, get_all=False):
        if len(self._preopt.results) == 0:
            return self._init_molecule
        #
        if get_all is False:
            return self._preopt.results[0].molecule
        return self._preopt.results[0].all_molecules()

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return []

    def setup_main(self, qm):
        if qm.softwares[self.software] is None:
            return
        folder = qm.pathways.getdir("preopt", create=True)
        calc = qm.setup_opt(folder, self._init_coords, self.atnums, preopt=True)
        self._preopt.calculations.append(calc)

    def check_main(self):
        for calc in self._preopt.calculations:
            try:
                _ = calc.check()
            except CalculationIncompleteError:
                return calc
        return None

    def parse_main(self, qm):
        results = []
        for calculation in self._preopt.calculations:
            files = calculation.check()
            results.append(OptimizationOutput(self._init_molecule,
                                              qm.read_opt(files, software='preopt')))
        self._preopt.results = results
