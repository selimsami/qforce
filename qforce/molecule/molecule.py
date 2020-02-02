from .topology import Topology
from .terms import Terms


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
        self.topo = Topology(inp, qm)
        self.terms = Terms(self.topo, ignore=inp.ignored_terms)#+['non_bonded']
