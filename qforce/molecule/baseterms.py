from abc import ABC, abstractmethod
#
import numpy as np
#
from .storage import TermStorage, MultipleTermStorge


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


class TermFactory(ABC):
    """Factory class to create ForceField Terms of one ore multiple TermABC classes"""

    _term_types = None
    _multiple_terms = True
    name = "NAME_NOT_DEFINED"

    @classmethod
    def get_terms_container(cls):
        if cls._multiple_terms is False:
            return TermStorage(cls.name)
        return MultipleTermStorge(cls.name, {key: value.get_terms_container()
                                             for key, value in cls._term_types.items()})

    @classmethod
    @abstractmethod
    def get_terms(cls, topo, non_bonded, settings):
        """
            Args:
                topo: Topology object, const
                    Stores all topology information
                non_bonded: NonBonded object, const
                    Stores all non bonded interaction information

            Return:
                list of cls objects
        """
        ...


class TermBase(TermFactory, TermABC):
    """Base class for terms that are TermFactories for themselves as well"""
    _multiple_terms = False


class EmptyTerm(TermBase):
    """Return an empty term container"""

    def _calc_forces(self, crd, force, fconst):
        raise NotImplementedError("should never be created")

    @classmethod
    def get_terms(cls, topo, non_bonded, settings):
        return None
