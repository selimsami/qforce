import os
from calkeeper import CalculationKeeper, CalculationIncompleteError
from .fragment import check_and_notify
from .logger import LoggerExit

def do_no_frag_scanning(mol, qm, job, config):
    scans = []
    unique_dihedrals = {}

    os.makedirs(config.scan.frag_lib, exist_ok=True)
    scan_dir = job.pathways.getdir('fragments', create=True)

    for term in mol.terms['dihedral/flexible']:
        name = term.typename.partition('_')[0]

        if name not in unique_dihedrals:
            unique_dihedrals[name] = term.atomids

    generated = []  # Number of fragments generated but not computed
    error = ''
    for name, atomids in unique_dihedrals.items():
        try:
            scan = Scan(job, config, mol, qm, atomids, name)
            if scan.has_data:
                scans.append(scan)
            elif config.scan.batch_run and scan.has_inp:
                generated.append(scan)
        except (CalculationIncompleteError, LoggerExit):
            # ignore these errors, checking is done in check_and_notify!
            pass

    check_and_notify(job, config.scan, len(unique_dihedrals), len(scans), len(generated)

    return scans

class Scan():
    def __init__(self, job, config, mol, qm, scanned_atomids, name):
        self.central_atoms = tuple(scanned_atomids[1:3])
        self.scanned_atomids = scanned_atomids
        self.atomids = mol.atomids
        self.name = name
        self.n_atoms = len(self.atomids)

        self.hash = ''
        self.hash_idx = 0
        self.id = ''
        self.has_data = False
        self.has_inp = False

        self.elements = []
        self.terms = None
        self.non_bonded = None
        self.remove_non_bonded = []
        self.qm_energies = []
        self.qm_coords = []
        self.qm_angles = []
        self.fit_terms = []
        self.coords = []
        self.frag_charges = []
        self.charge_scaling = config.ff.charge_scaling
        self.ext_charges = config.ff.ext_charges
        self.use_ext_charges_for_frags = config.ff.use_ext_charges_for_frags
        # set charge_method
        self.charge_method = qm.get_scan_software().config.charge_method

        self.check_fragment(job, config.scan, mol, qm, config.qm.dihedral_scanner)
