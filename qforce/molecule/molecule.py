from .topology import Topology
from .terms import Terms
from .non_bonded import NonBonded


class Molecule(object):

    def __init__(self, config, job, qm_out, ff, ext_q=None, ext_lj=None):
        self.name = job.name
        self.atomids = qm_out.atomids
        self.coords = qm_out.coords
        self.charge = qm_out.charge
        self.multiplicity = qm_out.multiplicity
        self.n_atoms = len(self.atomids)
        self.topo = Topology(config.ff, qm_out)
        self.non_bonded = NonBonded.from_topology(config.ff, job, qm_out, self.topo, ext_q, ext_lj)
        self.terms = Terms.from_topology(config.terms, self.topo, self.non_bonded, ff)
        self.qm_minimum_energy = None
        self.qm_minimum_coords = None
