from .topology import Topology
from .terms import Terms


class Molecule(object):
    """
    Scope:
    ------
    self.list : atom numbers of unique atoms grouped together
    self.atoms : unique atom numbers of each atom
    self.types : atom types of each atom
    self.neighbors : First 3 neighbors of each atom


    To do:
    -----
    - 180 degree angle = NO DIHEDRAL! (can/should avoid this in impropers?)

    - Since equivalent rigid dihedrals not necessarily have the same angle,
      can't average angles? Like in C60. But is it a bug or a feature? :)

    - Improper scan requires removal of relevant proper dihedrals from
      Redundant Coordinates in Gaussian

    - Think about cis/trans and enantiomers

    """

    def __init__(self, coords, atomids, inp, qm=None):
        self.topo = Topology(atomids, coords, qm, inp.n_equiv)
        self.terms = Terms(self.topo)
