from .topology import Topology
from .terms import Terms
from .non_bonded import NonBonded


class Molecule(object):
    """

    To do:
    -----
    - 180 degree angle = NO DIHEDRAL! (can/should avoid this in impropers?)

    - Since equivalent rigid dihedrals not necessarily have the same angle,
      can't average angles? Like in C60. But is it a bug or a feature? :)

    - Improper scan requires removal of relevant proper dihedrals from
      Redundant Coordinates in Gaussian

    - Think about cis/trans and enantiomers

    """

    def __init__(self, inp, qm):
        self.elements = qm.elements
        self.n_atoms = len(self.elements)
        self.topo = Topology(inp, qm)
        self.non_bonded = NonBonded.from_topology(inp, qm, self.topo)
        self.terms = Terms.from_topology(self.topo, self.non_bonded, ignore=inp.ignored_terms)
