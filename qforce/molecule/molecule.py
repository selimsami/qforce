from ase import Atoms
from ase.io import read, write
import numpy as np
#
from .topology import Topology
from .terms import Terms
from .non_bonded import NonBonded


class Molecule(object):

    def __init__(self, job, config):
        self.name = job.name
        self.job_dir = job.dir
        self.charge = config.qm.charge
        self.multiplicity = config.qm.multiplicity

        self.coords = None
        self.bond_orders = None
        self.point_charges = None

        coords, self.atomids = self._read_input_coords(job.coord_file)
        self.update_coords(coords, 'Input Coordinates')
        self.all_coords = [self.coords]
        self.n_atoms = len(self.atomids)

        self.topo = None
        self.non_bonded = None
        self.terms = None

        self.qm_minimum_energy = None
        self.qm_minimum_coords = None

    def _read_input_coords(self, file):
        ase_molecule = read(file)
        init_coords = ase_molecule.get_positions()
        atomids = ase_molecule.get_atomic_numbers()
        return init_coords, atomids

    def update_coords(self, coords, comment=''):
        self.coords = np.array(coords)
        atoms = Atoms(positions=coords, numbers=self.atomids)
        write(self.job_dir+'/coords.xyz', atoms, plain=True, comment=comment)

    def setup(self, config, job, ff_interface, hessian_out, ext_q, ext_lj):
        self.topo = Topology(config.ff, self)
        self.non_bonded = NonBonded.from_topology(config.ff, job, hessian_out, self.topo, ext_q, ext_lj)
        self.terms = Terms.from_topology(config.terms, self.topo, self.non_bonded, ff_interface)
