from collections import UserDict
from contextlib import contextmanager
from copy import deepcopy
#
from .storage import MultipleTermStorge, TermStorage
from .dihedral_terms import DihedralTerms
from .non_dihedral_terms import (HarmonicBondTerm, MorseBondTerm, HarmonicAngleTerm, CosineAngleTerm, UreyAngleTerm,
                                 CrossBondAngleTerm, CrossBondBondTerm, CrossAngleAngleTerm,
                                 CrossBondCosineAngleTerm, CrossCosineAngleAngleTerm,
                                 CrossDihedAngleTerm, CrossDihedBondTerm)
from .non_bonded_terms import NonBondedTerms
from .charge_flux_terms import ChargeFluxTerms
from .local_frame import LocalFrameTerms
#
from .base import MappingIterator
from .baseterms import TermFactory, EmptyTerm


class DefaultFalseDict(UserDict):

    def get(self, key, default=False):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data.get(key, False)



class TermSelector:

    def __init__(self, name, options, default=None):
        self.name = name
        self._options = options
        self._default = default

    def _select(self, config):
        typ = config.__dict__.get(f'{self.name}_type', self._default)
        if typ not in self._options:
            raise ValueError(f"Unknown term type '{typ}' for term '{self.name}'")
        return self._options[typ]

    def get_terms_from_config(self, config, topo, non_bonded, settings):
        factory = self._select(config)
        return factory.get_terms(topo, non_bonded, settings)


def split_name(name):
    maintype, subtype = name.split('/')
    maintype = maintype.strip()
    subtype = subtype.strip()
    return maintype, subtype


class Terms(MappingIterator):

    _term_factories = {
        'bond': TermSelector('bond', {
            'harmonic': HarmonicBondTerm,
            'morse': MorseBondTerm,
            }, 'harmonic'),
        'angle': TermSelector('angle', {
            'harmonic': HarmonicAngleTerm,
            'cosine': CosineAngleTerm,
            }, 'harmonic'),
        'cross_bond_bond': CrossBondBondTerm,
        #
        'cross_bond_angle': TermSelector('cross_bond_angle', {
            'bond_angle': CrossBondAngleTerm,
            'bond_cos_angle': CrossBondCosineAngleTerm,
            'false': EmptyTerm,
        }, 'false'),
        #
        'cross_angle_angle': TermSelector('cross_angle_angle', {
            'harmonic': CrossAngleAngleTerm,
            'cosine': CrossCosineAngleAngleTerm,
        }, 'harmonic'),
        #
        '_cross_dihed_angle': CrossDihedAngleTerm,
        '_cross_dihed_bond': CrossDihedBondTerm,
        #
        'dihedral': DihedralTerms,
        #
        'non_bonded': NonBondedTerms,
        #
        'charge_flux': ChargeFluxTerms,
        #
        'local_frame': LocalFrameTerms,
    }

    def __init__(self, terms, ignore, not_fit_terms, fit_flexible=False):
        MappingIterator.__init__(self, terms, ignore)
        self.n_fitted_terms, self.n_fitted_flux_terms = self._set_fit_term_idx(not_fit_terms, fit_flexible=fit_flexible)
        self.term_names = [name for name in self._term_factories.keys() if name not in ignore]
        self._term_paths = self._get_term_paths(terms)

    @classmethod
    def add_terms(cls, terms, name, config, topo, non_bonded, settings=None):
        _terms = cls._term_factories[name].get_terms_from_config(config, topo, non_bonded, settings)
        if _terms is not None:
            terms[name] = _terms

    @classmethod
    def from_topology(cls, config, topo, non_bonded, ff, *,
                      not_fit=['dihedral/flexible', 'non_bonded', 'charge_flux', 'local_frame'],
                      fit_flexible=False):
        terms = {}
        # handle always on terms
        for term in ff.always_on_terms:
            if '/' not in term:
                cls.add_terms(terms, term, config, topo, non_bonded)
            else:
                maintype, subtype = split_name(term)
                if maintype not in factories:
                    factories[maintype] = DefaultFalseDict()
                factories[maintype][subtype] = True
        # get off terms
        ignore = []
        factories = {}
        for term, enabled in config.__dict__.items():
            if term in ff.always_on_terms:
                continue
            if '/' in term:
                maintype, subtype = split_name(term)
                if maintype not in factories:
                    factories[maintype] = DefaultFalseDict()
                factories[maintype][subtype] = enabled
            else:
                if enabled is True:
                    cls.add_terms(terms, term, config, topo, non_bonded)
                else:
                    ignore.append(term)

        for term, settings in factories.items():
            cls.add_terms(terms, term, config, topo, non_bonded, settings)

        not_fit_terms = [term for term in not_fit if term not in ignore and term in config.__dict__.keys()]
        return cls(terms, ignore, not_fit_terms, fit_flexible=fit_flexible)

    @classmethod
    def from_terms(cls, terms, ignore, not_fit_terms):
        return cls(terms, ignore, not_fit_terms)

    def subset(self, fragment, mapping, remove_non_bonded=[], ignore=[], not_fit_terms=[]):
        subterms = {}
        for key, term in self.ho_items():
            if key in ignore:
                continue
            if isinstance(term, MultipleTermStorge):
                key_ignore = [term.get_key_subkey(ignore_key)[1] for ignore_key in ignore
                              if ignore_key.startswith(key)]
                subterms[key] = term.get_subset(fragment, mapping, key_ignore)
            elif isinstance(term, TermStorage):
                subterms[key] = term.get_subset(fragment, mapping, remove_non_bonded)
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

    def get_terms_from_name(self, name, atomids=None):
        termtyp = name.partition('(')[0]
        terms = self._get_terms(termtyp)
        return terms.fulfill(name, atomids)

    def remove_terms_by_name(self, name, atomids=None):
        termtyp = name.partition('(')[0]
        terms = self._get_terms(termtyp)
        terms.remove_term(name, atomids)

    def _set_fit_term_idx(self, not_fit_terms, fit_flexible=False):

        with self.add_ignore(not_fit_terms):
            names = list(set(str(term) for term in self))
            for term in self:
                term.set_idx(names.index(str(term)))
        n_fitted_terms = len(names)

        if 'charge_flux' not in self:
            names = []
        else:
            names = list(set(str(term) for term in self['charge_flux']))
            for term in self['charge_flux']:
                term.set_flux_idx(names.index(str(term)))
        n_fitted_flux_terms = len(names)

        if fit_flexible is True:
            names = list(set(str(term) for term in self['dihedral/flexible']))
            if len(names) != 0:
                for term in self['dihedral/flexible']:
                    term.set_idx(n_fitted_terms + term.idx_buffer*names.index(str(term)))
                n_fitted_terms += len(names)*term.idx_buffer
                if 'dihedral/flexible' in not_fit_terms:
                    not_fit_terms.remove('dihedral/flexible')

        for key in not_fit_terms:
            for term in self[key]:
                term.set_idx(n_fitted_terms)

        return n_fitted_terms, n_fitted_flux_terms

    def _get_term_paths(self, terms):
        paths = {}
        for name, term in terms.items():
            result = [name]
            if isinstance(term, TermStorage):
                paths[term.name] = result
            elif isinstance(term, MultipleTermStorge):
                for key, termstorage in term.ho_items():
                    self._get_storage_paths(paths, termstorage, key, result)
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

    def _get_terms(self, termname):
        names = self._term_paths.get(termname, None)
        if names is None:
            raise ValueError(f"Term name {termname} not known!")
        return get_entry(self._data, names)


def get_entry(dct, names):
    """get the entry of a folded dct by the names of the entries"""
    if len(names) == 0:
        return dct
    return get_entry(dct[names[0]], names[1:])
