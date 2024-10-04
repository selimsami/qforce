from calkeeper import CalculationIncompleteError
#
from .creator import CostumStructureCreator, CalculationStorage


class DihedralCreator(CostumStructureCreator):

    name = 'no_frag_dihedrals'

    def __init__(self, mol, job, config):
        # set weight to 0, it has to be set later one
        super().__init__(0)
        self.atomids = mol.atomids
        self.coords = mol.coords
        #
        self._unique_dihedrals = get_unique_dihedrals(mol, config.scan.do_scan)
        self._scans = CalculationStorage(as_dict=True)
        self._dihedrals = {name: CalculationStorage() for name in self._unique_dihedrals}
        #
        self.charge = config.qm.charge
        self.multiplicity = config.qm.multiplicity

    def _scan(self, qm, software, scanned_atomids):
        scan_hash = '_'.join(tuple(('_'.join((scanned_atomids+1).astype(dtype=str)),
                                    software.hash(self.charge, self.multiplicity))))

        folder = qm.pathways.getdir('frag', scan_hash, create=True)

        calc = qm.setup_scan_calculation(folder, scan_hash, scanned_atomids,
                                         self.coords, self.atomids)
        calc.scan_hash = scan_hash
        return calc

    def enouts(self):
        return []

    def gradouts(self):
        results = []
        for dihedral in self._dihedrals.values():
            for res in dihedral.results:
                results.append(res)
        return results

    def hessouts(self):
        return []

    def setup_pre(self, qm):
        """setup scans"""
        software = qm.get_scan_software()
        scans = {}
        for name, scanned_atomids in self._unique_dihedrals.items():
            scans[name] = self._scan(qm, software, scanned_atomids)
        self._scans.calculations = scans

    def check_pre(self):
        for calc in self._scans.calculations.values():
            try:
                _ = calc.check()
            except CalculationIncompleteError:
                return calc
        return None

    def parse_pre(self, qm):
        results = {}
        for name, calculation in self._scans.calculations.items():
            files = calculation.check()
            results[name] = qm.read_scan_data(files)
        self._scans.results = results

    def setup_main(self, qm):
        #
        for name, calc in self._scans.calculations.items():
            qm_out = self._scans.results[name]
            calcs = qm.setup_scan_sp_calculations(calc.folder, qm_out, self.atomids)
            self._dihedrals[name].calculations = calcs

    def check_main(self):
        for scan in self._dihedrals.values():
            for calc in scan.calculations:
                try:
                    _ = calc.check()
                except CalculationIncompleteError:
                    return calc
        return None

    def parse_main(self, qm):
        for dihedral in self._dihedrals.values():
            results = []
            for calculation in dihedral.calculations:
                files = calculation.check()
                results.append(qm.read_gradient(files))
            dihedral.results = results


def get_unique_dihedrals(mol, do_scan):
    scans = []
    unique_dihedrals = {}

    print(mol.terms['dihedral/flexible'])

    if 'dihedral/flexible' not in mol.terms or len(mol.terms['dihedral/flexible']) == 0 or not do_scan:
        return {}

    for term in mol.terms['dihedral/flexible']:
        dih_type = mol.topo.edge(term.atomids[1], term.atomids[2])['vers']
        if dih_type not in unique_dihedrals:
            unique_dihedrals[dih_type] = term.atomids

    return unique_dihedrals
