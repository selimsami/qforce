from .dihedral_terms import DihedralTerms
from .non_dihedral_terms import BondTerm, AngleTerm, UrayAngleTerm
#
from .base import MappingIterator
from .baseterms import TermFactory
#
from .topology import QM, Topology


class Terms(MappingIterator):

    _term_factories = {
            'bond': BondTerm,
            'angle': AngleTerm,
            'uray': UrayAngleTerm,
            'dihedral': DihedralTerms,
    }

    def __init__(self, topo, ignore=[]):
        _terms = {name: factory.get_terms(topo)
                  for name, factory in self._term_factories.items()}
        # enable iteration
        MappingIterator.__init__(self, _terms, ignore)

    @classmethod
    def add_term(cls, name, term):
        if not isinstance(term, TermFactory):
            raise ValueError('New term needs to be a TermFactory!')
        cls._term_factories[name] = term
