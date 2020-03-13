from contextlib import contextmanager
from copy import deepcopy
import numpy as np
#
from .storage import MultipleTermStorge, TermStorage
from .dihedral_terms import DihedralTerms
from .non_dihedral_terms import (BondTerm, AngleTerm, UreyAngleTerm, CrossBondAngleTerm)
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
        self.n_fitted_terms = self._set_fit_term_idx(not_fit_terms)
        self.term_names = [name for name in self._term_factories.keys() if name not in ignore]

    @classmethod
    def from_topology(cls, topo, ignore=[]):
        not_fit_terms = [term for term in ['dihedral/flexible', 'dihedral/constr', 'non_bonded']
                         if term not in ignore]
        terms = {name: factory.get_terms(topo)
                 for name, factory in cls._term_factories.items() if name not in ignore}
        print(terms)
        return cls(terms, ignore, not_fit_terms)

    @classmethod
    def from_terms(cls, terms, ignore, not_fit_terms):
        return cls(terms, ignore, not_fit_terms)

    def subset(self, fragment, mapping, ignore=[], not_fit_terms=[]):

        subterms = {}
        for key, term in self.ho_items():
            if key in ignore:
                continue
            if isinstance(term, MultipleTermStorge):
                key_ignore = [term.get_key_subkey(ignore_key)[1] for ignore_key in ignore if ignore_key.startswith(key)]
                subterms[key] = term.get_subset(fragment, mapping, key_ignore)
            elif isinstance(term, TermStorage):
                subterms[key] = term.get_subset(fragment, mapping)
            else:
                raise ValueError("Term can only be TermStorage or MultipleTermStorage")

        return self.from_terms(subterms, ignore, not_fit_terms)

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

    def _set_fit_term_idx(self, not_fit_terms):

        with self.add_ignore(not_fit_terms):
            names = list(set(str(term) for term in self))
            for term in self:
                term.set_idx(names.index(str(term)))

        n_fitted_terms = len(names)

        for key in not_fit_terms:
            for term in self[key]:
                term.set_idx(n_fitted_terms)

        return n_fitted_terms
