from contextlib import contextmanager
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
        self._term_paths = self._get_term_paths(terms)

    @classmethod
    def from_topology(cls, topo, ignore=[]):
        not_fit_terms = [term for term in ['dihedral/flexible', 'dihedral/constr', 'non_bonded']
                         if term not in ignore]
        terms = {name: factory.get_terms(topo)
                 for name, factory in cls._term_factories.items() if name not in ignore}
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

    def get_terms_from_name(self, name):
        termtyp = name.partition('(')[0]
        names = self._term_paths.get(termtyp, None)
        if names is None:
            raise ValueError(f"Term name {termtyp} not known!")
        terms = get_entry(self._data, names)
        return (term for term in terms if term == name)

    def remove_terms_by_name(self, name):
        termtyp = name.partition('(')[0]
        names = self._term_paths.get(termtyp, None)
        if names is None:
            raise ValueError(f"Term name {termtyp} not known!")
        terms = get_entry(self._data, names)
        terms.remove_term(name)

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

    def _get_term_paths(self, terms):
        paths = {}
        for name, term in terms.ho_items():
            if isinstance(term, TermStorage):
                paths[term.name] = name
            elif isinstance(term, MultipleTermStorge):
                for key, termstorage in term.ho_items():
                    self._get_storage_paths(paths, termstorage, key, [name])
            else:
                raise ValueError("Terms can only be stored in TermStorage or MultipleTermStorge")
        return paths

    def _get_storage_paths(self, result, term, name, names=[]):
        names = deepcopy(names)
        if isinstance(term, TermStorage):
            names.append(name)
            result[term.name] = names
        elif isinstance(term, MultipleTermStorge):
            names.append(name)
            for key, termstorage in term.ho_items():
                self._get_storage_paths(result, termstorage, key, names)
        else:
            raise ValueError("Terms can only be stored in TermStorage or MultipleTermStorge")


def get_entry(dct, names):
    """get the entry of a folded dct by the names of the entries"""
    if len(names) == 0:
        return dct
    return get_entry(dct[names[0]], names[1:])
