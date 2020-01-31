from abc import ABC, abstractmethod
#
import numpy as np
#
from .storage import TermStorage, MultipleTermStorge


class TermABC(ABC):

    __slots__ = ('atomids', 'equ', 'idx', 'fconst', '_typename')

    name = 'NOT_NAMED'

    def __init__(self, atomids, equ, typename, fconst=None):
        """Initialization of a term"""
        self.atomids = np.array(atomids)
        self.equ = equ
        self.idx = 0
        self.fconst = fconst
        self._typename = typename

    def __repr__(self):
        return f"{self.name}({self._typename})"

    def __str__(self):
        return f"{self.name}({self._typename})"

    def set_idx(self, idx):
        self.idx = idx

    def do_force(self, crd, force):
        """force calculation with given geometry"""
        self._calc_forces(crd, force, self.fconst)

    def do_fitting(self, crd, forces):
        """compute fitting contributions"""
        self._calc_forces(crd, forces[self.idx], 1.0)

    @abstractmethod
    def _calc_forces(self, crd, force, fconst):
        """Perform actuall force computation"""

    @classmethod
    def get_terms_container(cls):
        return TermStorage(cls.name)


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
    def get_terms(cls, topo):
        """
            Args:
                topo: Topology object, const
                    Stores all topology information

            Return:
                list of cls objects
        """
        ...


class TermBase(TermFactory, TermABC):
    """Base class for terms that are TermFactories for themselves as well"""
    _multiple_terms = False
