from .topology import Topology
from .terms import Terms
from .non_bonded import NonBonded


class Molecule(object):
    """

    To do:
    -----
    - Improper scan requires removal of relevant proper dihedrals from
      Redundant Coordinates in Gaussian

    - Think about cis/trans and enantiomers

    """

    def __init__(self, config, job, qm_out):
        self.name = job.name
        self.elements = qm_out.elements
        self.n_atoms = len(self.elements)
        self.topo = Topology(config.ff, qm_out)
        self.non_bonded = NonBonded.from_topology(config.ff, job, qm_out, self.topo)
        self.terms = Terms.from_topology(config.terms, self.topo, self.non_bonded)
