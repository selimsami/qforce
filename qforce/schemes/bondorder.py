from .creator import CustomStructureCreator, CalculationStorage


class BondOrderCreator(CustomStructureCreator):

    def __init__(self, molecule, weight=0, preopt=True):
        super().__init__(weight)
        #
        self.init_coords = molecule.get_positions()
        self.atnums = molecule.get_atomic_numbers()
        #
        self._bondorder = CalculationStorage()
        self._charges = CalculationStorage()
        # use preopt or not
        self._preopt = preopt

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return self._hessian.results

    def bondorder(self):
        if len(self._bondorder.results) == 0:
            raise ValueError("At least one bond order needs to be provided")
        return self._bondorder.results[0]

    def setup_main(self, qm):
        folder = qm.pathways.getdir("hessian", create=True)
        self._bondorder.calculations.append(qm.setup_hessian_calculation(
            folder, self.init_coords, self.atnums, preopt=self._preopt))

    def check_main(self):
        return self._check(self._bondorder.calculations)

    def parse_main(self, qm):
        results = []
        for calculation in self._bondorder.calculations:
            hessian_files = calculation.check()
            results.append(qm.read_hessian(hessian_files, preopt=self._preopt))
        self._bondorder.results = results

    def setup_post(self, qm):
        if self._preopt is False and qm.charge_software_is_defined is False:
            return
        folder = qm.pathways.getdir("hessian_charge", create=True)
        for qm_out in self._bondorder.results:
            self._charges.calculations.append(
                    qm.setup_charge_calculation(folder, qm_out.coords, qm_out.atomids))

    def check_post(self):
        return self._check(self._charges.calculations)

    def parse_post(self, qm):
        # adjust hessian out
        for i, calculation in enumerate(self._charges.calculations):
            files = calculation.check()
            point_charges = qm.read_charges(files)
            output = self._bondorder.results[i]
            output.point_charges = output.check_type_and_shape(
                    point_charges, 'point_charges', float, (output.n_atoms,))
