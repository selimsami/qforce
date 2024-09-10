from calkeeper import CalculationKeeper, CalculationIncompleteError
import os
import numpy as np
#
from .fragment import check_and_notify
from .logger import LoggerExit
from .forces import get_dihed
from .qm.qm_base import GradientOutput


def do_nofrag_scanning(mol, qm, job, config):
    scans = []
    unique_dihedrals = {}

    if len(mol.terms['dihedral/chosen']) == 0 or not config.scan.do_scan:
        return scans

    os.makedirs(config.scan.frag_lib, exist_ok=True)
    scan_dir = job.pathways.getdir('fragments', create=True)

    for term in mol.terms['dihedral/chosen']:
        name = term.type.split('_')[1]
        if name not in unique_dihedrals:
            unique_dihedrals[name] = term.atomids

    generated = []  # Number of fragments generated but not computed
    error = ''
    for name, atomids in unique_dihedrals.items():
        try:
            scan = Scan(job, config, mol, qm, atomids, name)
            if scan.has_data:
                scans.extend(convert_to_grad_out(scan))
            elif config.scan.batch_run and scan.has_inp:
                generated.extend(convert_to_grad_out(scan))
        except (CalculationIncompleteError, LoggerExit):
            # ignore these errors, checking is done in check_and_notify!
            pass

    n_scan_steps = int(np.ceil(360/config.qm.scan_step_size))

    check_and_notify(job, config.scan, len(unique_dihedrals), len(scans)//n_scan_steps, len(generated)//n_scan_steps)

    return scans


def convert_to_grad_out(scan):
    grad_outs = []
    for i in range(len(scan.coords)):
        grad_out = GradientOutput(scan.energies[i], scan.forces[i], scan.dipoles[i], scan.atomids, scan.coords[i])
        grad_outs.append(grad_out)
    return grad_outs


class Scan():
    def __init__(self, job, config, mol, qm, scanned_atomids, name):
        self.central_atoms = tuple(scanned_atomids[1:3])
        self.scanned_atomids = scanned_atomids
        self.atomids = mol.atomids
        self.opt_coords = mol.coords
        self.charge = mol.charge
        self.multiplicity = mol.multiplicity
        self.equil_angle = np.degrees(get_dihed(self.opt_coords[self.scanned_atomids])[0])
        self.name = name
        self.n_atoms = len(self.atomids)
        self.has_data = False
        self.has_inp = False

        self.software = qm.get_scan_software()
        self.hash = self.make_hash()
        self.folder = job.pathways.getdir('frag', self.hash, create=True)
        self.calc = self.set_calc(config, job)

        self.energies = []
        self.qcoords = []
        self.forces = []
        self.dipoles = []

        self.check_for_qm_data(job, config, mol, qm)

    def make_hash(self):
        atomids = '_'.join((self.scanned_atomids+1).astype(dtype=str)) + '_'
        hash = self.software.hash(self.charge, self.multiplicity)
        return atomids + hash

    def set_calc(self, config, job):
        if config.qm.dihedral_scanner == 'torsiondrive':
            calc = job.Calculation(f'{self.hash}_torsiondrive.inp',
                                        self.software.required_scan_torsiondrive_files,
                                        folder=self.folder, software='torsiondrive')
        elif config.qm.dihedral_scanner == 'relaxed_scan':
            calc = job.Calculation(f'{self.hash}.inp', self.software.required_scan_files,
                                        folder=self.folder, software=self.software.name)
        else:
            raise ValueError("scanner can only be 'torsiondrive' or 'relaxed_scan'")

        return calc

    def check_for_qm_data(self, job, config, mol, qm):
        files = [f for f in os.listdir(self.folder) if self.hash in f and f.endswith(('log', 'out'))]
        if files:
            self.has_data = True
            qm_out = qm.read_scan(self.folder, files)
            qm_out = qm.do_scan_sp_calculations_v2(self.folder, self.hash, qm_out, mol.atomids)

            self.energies = qm_out.energies
            self.forces = qm_out.forces
            self.coords = qm_out.coords
            self.dipoles = qm_out.dipoles

            if qm_out.mismatch:
                if config.scan.avail_only:
                    job.logger.info('"\navail_only" requested, attempting to continue with '
                                    'the missing points...\n\n')
                else:
                    job.logger.exit('Exiting...\n\n')

        if not self.has_data:
                self.make_qm_input(qm)

    def make_qm_input(self, qm):
        with open(self.calc.inputfile, 'w') as file:
            qm.write_scan(file, self.hash, self.opt_coords, self.atomids, self.scanned_atomids+1,
                          self.equil_angle, self.charge, self.multiplicity)
