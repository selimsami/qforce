from .creator import CustomStructureCreator, CalculationStorage


class CrestCreator(CustomStructureCreator):

    def __init__(self, folder, mol):
        super().__init__(0, folder=folder)
        self.init_coords = mol.coords
        self.atomids = mol.atomids
        self._calc = CalculationStorage()
        self.software = 'crest'

    def setup_main(self, qm):
        self._calc.calculations = [qm.setup_crest(self.folder, self.init_coords, self.atomids)]

    def check_main(self):
        return self._check(self._calc.calculations)

    def parse_main(self, qm):
        files = self._calc.calculations[0].check()
        self._calc.results = qm.read_opt(files, software='crest')

    def get_structures(self):
        return [(coord, self.atomids) for coord in self._calc.results[0]]

    def get_bond_orders(self):
        return self._calc.results[1]

    def get_most_stable(self):
        if len(self._calc.results[0]) == 0:
            raise ValueError("Result does not contain a single structure!")
        return self._calc.results[0][0]
