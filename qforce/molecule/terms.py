from .dihedral_terms import DihedralTerms
from .non_dihedral_terms import BondTerm, AngleTerm, UreyAngleTerm
#
from .base import MappingIterator
from .baseterms import TermFactory


class Terms(MappingIterator):

    _term_factories = {
            'bond': BondTerm,
            'angle': AngleTerm,
            'urey': UreyAngleTerm,
            'dihedral': DihedralTerms,
    }

    def __init__(self, topo, ignore=['dihedral/flexible', 'dihedral/constr'],
                 not_fit_terms=[]):
        _terms = {name: factory.get_terms(topo)
                  for name, factory in self._term_factories.items()}
        # enable iteration
        MappingIterator.__init__(self, _terms, ignore)
        self.n_fitted_terms = self._set_fitting_term_idx(not_fit_terms)

    def _set_fitting_term_idx(self, not_fit_terms):
        self.add_ignore_keys(not_fit_terms)

        names = list(set(str(term) for term in self))
        for term in self:
            term.set_idx(names.index(str(term)))

        self.remove_ignore_keys(not_fit_terms)

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
