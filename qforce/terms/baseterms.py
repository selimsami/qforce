from abc import ABC, abstractmethod
#
import numpy as np
# smart iterator
from collections.abc import Mapping


class TermABC(ABC):

    def __init__(self, atomids, equ, typename, fconst=None):
        """Initialization of a term"""
        self.atomids = np.array(atomids)
        self.equ = equ
        self.idx = 0
        self.fconst = fconst
        self._typename = typename

    def __repr__(self):
        return f"{self.__class__}({self._typename})"

    def __str__(self):
        return f"{self.__class__}({self._typename})"

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


class TermFactory(ABC):
    """Factory class to create ForceField Terms of one ore multiple TermABC classes"""

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


class TermBase(TermABC, TermFactory):
    """Base class for terms that are TermFactories for themselves as well"""
    pass


class MappingIterator(Mapping):

    def __init__(self, dct, ignore=[]):
        self._dct_data = dct
        self._ignore = set(ignore)

    @property
    def ignore(self):
        return self._ignore

    @ignore.setter
    def ignore(self, value):
        self._ignore = set(value)

    def add_ignore_key(self, value):
        self._ignore.add(value)

    def remove_ignore_key(self, value):
        self._ignore.remove(value)

    def __getitem__(self, key):
        return self._dct_data[key]

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        for key, value in self._dct_data.items():
            if key not in self.ignore:
                for rval in value:
                    yield rval
