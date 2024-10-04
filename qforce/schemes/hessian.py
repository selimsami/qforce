from ase.io import read
from calkeeper import CalculationIncompleteError
#
from .creator import CostumStructureCreator, CalculationStorage


class HessianCreator(CostumStructureCreator):

    name = 'hessian'

    def __init__(self, molecule):
        self.init_coords = molecule.get_positions()
        self.atnums = molecule.get_atomic_numbers()
        #
        self._hessian = CalculationStorage()
        self._charges = CalculationStorage()

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return self._hessian.results

    def main_hessian(self):
        if len(self._hessian.results) == 0:
            raise ValueError("At least one hessian needs to be provided")
        return self._hessian.results[0]

    def setup_pre(self, qm):
        #
        folder = qm.pathways.getdir("hessian_new", create=True)
        self._hessian.calculations.append(qm.setup_hessian(folder, self.init_coords, self.atnums))

    def check_pre(self):
        for calc in self._hessian.calculations:
            try:
                _ = calc.check()
            except CalculationIncompleteError:
                return calc
        return None

    def parse_pre(self, qm):
        results = []
        for calculation in self._hessian.calculations:
            hessian_files = calculation.check()
            results.append(qm.read_hessian(hessian_files))
        self._hessian.results = results

    def setup_main(self, qm):
        #
        if qm.softwares['charge_software'] is None:
            return
        folder = self.pathways.getdir("hessian_charge", create=True)
        qm_out = self._hessian.results[0]
        
        self._charges.calculations.append(qm.setup_charge_calculation(folder, qm_out.coords, qm_out.atomids))

    def check_main(self):
        for calc in self._charges.calculations:
            try:
                _ = calc.check()
            except CalculationIncompleteError:
                return calc
        return None

    def parse_main(self, qm):
        # adjust hessian out
        results = []
        for i, calculation in enumerate(self._charges.calculations):
            files = calculation.check()
            point_charges = qm.read_charges(files)
            output = self._hessian.results[i]
            output.point_charges = output.check_type_and_shape(
                    point_charges, 'point_charges', float, (output.n_atoms,))
