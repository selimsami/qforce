import os
#
from .creator import CalculationStorage, CustomStructureCreator


class MultipleStructureCalculationCreator(CustomStructureCreator):

    def __init__(self, folder, weight, atomids, coords):
        super().__init__(weight, folder=folder)
        self._atomids = atomids
        self._coords = coords
        self._calcs = CalculationStorage()

    def _structure_iter(self):
        ids = self._atomids
        for i, coord in enumerate(self._coords):
            yield i, (coord, ids)

    def check_main(self):
        return self._check(self._calcs.calculations)


class EnergyCalculationCreator(MultipleStructureCalculationCreator):

    def setup_main(self, qm):
        folder = self.folder
        #
        os.makedirs(folder, exist_ok=True)
        self._calcs.calculations = qm.setup_energy_calculations(folder, self._structure_iter())

    def parse_main(self, qm):
        results = []
        for calculation in self._calcs.calculations:
            files = calculation.check()
            results.append(qm.read_energy(files))
        self._calcs.results = results

    def enouts(self):
        return self._calcs.results

    def gradouts(self):
        return []

    def hessouts(self):
        return []


class GradientCalculationCreator(MultipleStructureCalculationCreator):

    def setup_main(self, qm):
        folder = self.folder

        os.makedirs(folder, exist_ok=True)
        self._calcs.calculations = qm.setup_grad_calculations(folder, self._structure_iter())

    def parse_main(self, qm):
        results = []
        for calculation in self._calcs.calculations:
            files = calculation.check()
            results.append(qm.read_gradient(files))
        self._calcs.results = results

    def enouts(self):
        return []

    def gradouts(self):
        return self._calcs.results

    def hessouts(self):
        return []


class HessianCalculationCreator(MultipleStructureCalculationCreator):

    def setup_main(self, qm):
        folder = self.folder
        os.makedirs(folder, exist_ok=True)
        #
        self._calcs.calculations = qm.setup_hessian_calculations(folder, self._structure_iter())

    def parse_main(self, qm):
        results = []
        for calculation in self._hess.calculations:
            hessian_files = calculation.check()
            results.append(qm.read_hessian(hessian_files))
        self._calcs.results = results

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return self._calcs.results


class MultipleStructureCalculationIterCreator(CustomStructureCreator):

    def __init__(self, folder, weight, itr):
        super().__init__(weight, folder=folder)
        self._struct_itr = itr
        self._calcs = CalculationStorage()

    def _structure_iter(self):
        for i, (coords, ids) in enumerate(self._struct_itr):
            yield i, (coords, ids)

    def check_main(self):
        return self._check(self._calcs.calculations)


class EnergyCalculationIterCreator(MultipleStructureCalculationIterCreator):

    def setup_main(self, qm):
        folder = self.folder
        #
        os.makedirs(folder, exist_ok=True)
        self._calcs.calculations = qm.setup_energy_calculations(folder, self._structure_iter())

    def parse_main(self, qm):
        results = []
        for calculation in self._calcs.calculations:
            files = calculation.check()
            results.append(qm.read_energy(files))
        self._calcs.results = results

    def enouts(self):
        return self._calcs.results

    def gradouts(self):
        return []

    def hessouts(self):
        return []


class GradientCalculationIterCreator(MultipleStructureCalculationIterCreator):

    def setup_main(self, qm):
        folder = self.folder

        os.makedirs(folder, exist_ok=True)
        self._calcs.calculations = qm.setup_grad_calculations(folder, self._structure_iter())

    def parse_main(self, qm):
        results = []
        for calculation in self._calcs.calculations:
            files = calculation.check()
            results.append(qm.read_gradient(files))
        self._calcs.results = results

    def enouts(self):
        return []

    def gradouts(self):
        return self._calcs.results

    def hessouts(self):
        return []


class HessianCalculationIterCreator(MultipleStructureCalculationIterCreator):

    def setup_main(self, qm):
        folder = self.folder
        os.makedirs(folder, exist_ok=True)
        #
        self._calcs.calculations = qm.setup_hessian_calculations(folder, self._structure_iter())

    def parse_main(self, qm):
        results = []
        for calculation in self._hess.calculations:
            hessian_files = calculation.check()
            results.append(qm.read_hessian(hessian_files))
        self._calcs.results = results

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return self._calcs.results
