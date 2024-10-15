from .creator import CustomStructureCreator, CalculationStorage


class CrestCreator(CustomStructureCreator):

    def __init__(self, folder, molecule):
        super().__init__(0, folder=folder)
        self.init_coords = molecule.get_positions()
        self.atnums = molecule.get_atomic_numbers()
        self._calc = CalculationStorage()
        self.software = 'crest'

    def setup_main(self, qm):
        self._calc.calculations = [qm.setup_crest(self.folder, self.init_coords, self.atnums)]

    def check_main(self):
        return self._check(self._calc.calculations)

    def parse_main(self, qm):
        files = self._calc.calculations[0].check()
        self._calc.results = qm.read_opt(files, software='crest')

    def structures(self):
        return [(coord, self.atnums) for coord in self._calc.results]

    def most_stable(self):
        if len(self._calc.results) == 0:
            raise ValueError("Result does not contain a single structure!")
        return self._calc.results[0]
