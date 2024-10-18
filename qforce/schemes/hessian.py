from .creator import CustomStructureCreator, CalculationStorage


class HessianCreator(CustomStructureCreator):

    def __init__(self, mol, weight=0):
        super().__init__(weight)
        #
        self.init_coords = mol.coords
        self.atnums = mol.atomids
        #
        self._hessian = CalculationStorage()
        self._charges = CalculationStorage()

    def enouts(self, select='all'):
        return []

    def gradouts(self, select='all'):
        return []

    def hessouts(self, select='all'):
        return self._hessian.results

    def main_hessian(self):
        if len(self._hessian.results) == 0:
            raise ValueError("At least one hessian needs to be provided")
        return self._hessian.results[0]

    def setup_main(self, qm):
        folder = qm.pathways.getdir("hessian", create=True)
        self._hessian.calculations.append(qm.setup_hessian_calculation(folder, self.init_coords, self.atnums))

    def check_main(self):
        return self._check(self._hessian.calculations)

    def parse_main(self, qm):
        results = []
        for calculation in self._hessian.calculations:
            hessian_files = calculation.check()
            results.append(qm.read_hessian(hessian_files))
        self._hessian.results = results

    def setup_post(self, qm):
        if qm.charge_software_is_defined is False:
            return
        folder = self.pathways.getdir("hessian_charge", create=True)
        for qm_out in self._hessian.results:
            self._charges.calculations.append(
                    qm.setup_charge_calculation(folder, qm_out.coords, qm_out.atomids))

    def check_post(self):
        return self._check(self._charges.calculations)

    def parse_post(self, qm):
        # adjust hessian out
        results = []
        for i, calculation in enumerate(self._charges.calculations):
            files = calculation.check()
            point_charges = qm.read_charges(files)
            output = self._hessian.results[i]
            output.point_charges = output.check_type_and_shape(
                    point_charges, 'point_charges', float, (output.n_atoms,))
