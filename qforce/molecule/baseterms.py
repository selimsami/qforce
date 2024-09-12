from collections import UserDict
from abc import ABC, abstractmethod
#
import numpy as np
#
from .storage import TermStorage, MultipleTermStorge
from .selectors import to_selector


class TermABC(ABC):

    __slots__ = ('atomids', 'equ', 'idx', 'fconst', 'type', '_name')

    name = 'NOT_NAMED'

    def __init__(self, atomids, equ, term_type, fconst=None):
        """Initialization of a term"""
        self.atomids = np.array(atomids)
        self.equ = equ
        self.idx = 0
        self.flux_idx = 0
        self.fconst = fconst
        self.type = term_type
        self._name = f"{self.name}({term_type})"

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    def set_idx(self, idx):
        self.idx = idx

    def set_flux_idx(self, idx):
        self.flux_idx = idx

    def do_force(self, crd, force):
        """force calculation with given geometry"""
        return self._calc_forces(crd, force, self.fconst)

    def do_fitting(self, crd, energies, forces):
        """compute fitting contributions"""
        energies[self.idx] += self._calc_forces(crd, forces[self.idx], 1.0)

    def set_fitparameters(self, parameters):
        """set the parameters after fitting"""
        self.fconst = parameters[self.idx]

    @abstractmethod
    def _calc_forces(self, crd, force, fconst):
        """Perform actual force computation"""

    @classmethod
    def get_terms_container(cls):
        return TermStorage(cls.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self._name
        if isinstance(other, TermABC):
            return str(other) == self._name
        else:
            raise Exception("Cannot compare Term with")

    def __ne__(self, other):
        if isinstance(other, str):
            return other != self._name
        if isinstance(other, TermABC):
            return str(other) != self._name
        else:
            raise Exception("Cannot compare Term with")


class TermBase(TermABC):
    """Base class for terms that are TermFactories for themselves as well"""
    _multiple_terms = False

    @classmethod
    def get_terms_container(cls):
        return TermStorage(cls.name)

    @classmethod
    def get_terms(cls, topo, non_bonded, settings):
        """
        Parameters
        ----------
        topo: Topology object, const
            Stores all topology information
        non_bonded: NonBonded object, const
            Stores all non bonded interaction information

        Return:
            list of term objects
        """
        return cls._get_terms(topo, non_bonded)

    @classmethod
    @abstractmethod
    def _get_terms(cls, topo, non_bonded):
        """
        Parameters
        ----------
        topo: Topology object, const
            Stores all topology information
        non_bonded: NonBonded object, const
            Stores all non bonded interaction information
        settings : dict
            should be off

        Return:
            list of term objects
        """


class EmptyTerm(TermBase):
    """Return an empty term container"""

    def _calc_forces(self, crd, force, fconst):
        raise NotImplementedError("should never be created")

    @classmethod
    def get_terms(cls, topo, non_bonded, settings):
        return None

    @classmethod
    def get_term(cls, *args, **kwargs):
        return None


class DefaultEmptyDict(UserDict):
    """Dictory that return False in case a key is not defined"""

    def is_empty(self):
        return all(val is EmptyTerm for val in self.data.values())

    def is_on(self, key):
        return not (self[key] is EmptyTerm)

    def get(self, key, default=EmptyTerm):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data.get(key, EmptyTerm)


class TermFactory:
    """Base class for terms that are TermFactories for themselves as well"""
    _multiple_terms = True
    _term_types = None

    @classmethod
    def factories(cls):
        if cls._term_types is None:
            return {}
        return to_selector(cls._term_types, EmptyTerm)

    @classmethod
    def get_terms_container(cls, termtypes=None):
        if termtypes is None:
            return MultipleTermStorge(cls.name, {key: value.get_terms_container()
                                                 for key, value in cls._term_types.items()})
        return MultipleTermStorge(cls.name, {key: value.get_terms_container()
                                             for key, value in termtypes.items()})

    @classmethod
    def get_terms(cls, topo, non_bonded, settings):
        """
        Parameters
        ----------
        topo: Topology object, const
            Stores all topology information
        non_bonded: NonBonded object, const
            Stores all non bonded interaction information
        settings : dict
            selections for termtypes

        Return:
            list of term objects
        """
        factories = cls.factories()
        termtypes = DefaultEmptyDict()
        for name, prop in settings.items():
            termtypes[name] = factories[name].get_factory(prop)

        return cls._get_terms(topo, non_bonded, termtypes)

    @classmethod
    @abstractmethod
    def _get_terms(cls, topo, non_bonded, termtypes):
        """
        Parameters
        ----------
        topo: Topology object, const
            Stores all topology information
        non_bonded: NonBonded object, const
            Stores all non bonded interaction information
        termtypes : SimpleNamespace
            selections for termtypes

        Return:
            list of term objects
        """
