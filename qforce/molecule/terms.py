from contextlib import contextmanager
#
from .dihedral_terms import DihedralTerms
from .non_dihedral_terms import (BondTerm, AngleTerm, UreyAngleTerm,
                                 CrossBondAngleTerm)
from .non_bonded_terms import NonBondedTerms
#
from .base import MappingIterator
from .baseterms import TermFactory


class Terms(MappingIterator):

    _term_factories = {
            'bond': BondTerm,
            'angle': AngleTerm,
            'urey': UreyAngleTerm,
            'cross_bond_angle': CrossBondAngleTerm,
            'dihedral': DihedralTerms,
            'non_bonded': NonBondedTerms,
    }

    def __init__(self, topo, ignore=[]):
        not_fit_terms = [term for term in ['dihedral/flexible', 'dihedral/constr', 'non_bonded']
                         if term not in ignore]
        _terms = {name: factory.get_terms(topo)
                  for name, factory in self._term_factories.items() if name not in ignore}
        # enable iteration
        MappingIterator.__init__(self, _terms, ignore)
        self.n_fitted_terms = self._set_fitting_term_idx(not_fit_terms)
        self.term_names = [name for name in self._term_factories.keys() if name not in ignore]

    def _set_fitting_term_idx(self, not_fit_terms):

        with self.add_ignore(not_fit_terms):
            names = list(set(str(term) for term in self))
            for term in self:
                term.set_idx(names.index(str(term)))

        n_fitted_terms = len(names)

        for key in not_fit_terms:
            for term in self[key]:
                term.set_idx(n_fitted_terms)

        return n_fitted_terms

    @classmethod
    def add_term(cls, name, term):
        if not isinstance(term, TermFactory):
            raise ValueError('New term needs to be a TermFactory!')
        cls._term_factories[name] = term

    @contextmanager
    def add_ignore(self, ignore_terms):
        self.add_ignore_keys(ignore_terms)
        yield
        self.remove_ignore_keys(ignore_terms)
