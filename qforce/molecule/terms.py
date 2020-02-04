from contextlib import contextmanager
from copy import deepcopy
import numpy as np
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

    def __init__(self, terms, ignore, not_fit_terms):
        MappingIterator.__init__(self, terms, ignore)
        self.n_fitted_terms = self._set_fitting_term_idx(not_fit_terms)
        self.term_names = [name for name in self._term_factories.keys() if name not in ignore]

    @classmethod
    def from_topology(cls, topo, ignore=[]):
        not_fit_terms = [term for term in ['dihedral/flexible', 'dihedral/constr', 'non_bonded']
                         if term not in ignore]
        terms = {name: factory.get_terms(topo)
                 for name, factory in cls._term_factories.items() if name not in ignore}
        return cls(terms, ignore, not_fit_terms)

    @classmethod
    def as_subset(cls, terms, fragment, mapping, ignore=[], not_fit_terms=[]):
        subterms = {}
        for key, termlist in terms.items():
            if key == 'dihedral/flexible':
                continue
            subterms[key] = []
            for term in termlist:
                if set(term.atomids).issubset(fragment):
                    term = deepcopy(term)
                    term.atomids = np.array([mapping[i] for i in term.atomids])
                    subterms[key].append(term)
        return cls(subterms, ignore, not_fit_terms)

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

    def subset(self, fragment, mapping):
        return self.as_subset(self, fragment, mapping)

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
