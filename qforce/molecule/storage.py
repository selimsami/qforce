from copy import deepcopy
from collections import UserList
#
import numpy as np
#
from .base import MappingIterator


class MultipleTermStorge(MappingIterator):

    def __init__(self, name, dct):
        self.name = name
        MappingIterator.__init__(self, dct)

    @classmethod
    def new_storage(cls, name, dct):
        return cls(name, dct)

    def get_subset(self, fragment, mapping, ignore=[]):

        out = {}
        for name, term in self.ho_items():
            if name in ignore:
                continue
            if isinstance(term, MultipleTermStorge):
                key_ignore = [self.get_key_subkey(ignore_key)[1] for ignore_key in ignore
                              if ignore.startswith(name)]
                out[name] = term.get_subset(fragment, mapping, key_ignore)
            elif isinstance(term, TermStorage):
                out[name] = term.get_subset(fragment, mapping)
            else:
                raise ValueError("Term can only be TermStorage or MultipleTermStorage")

        return self.new_storage(self.name, out)

    def __str__(self):
        names = ", ".join(self.keys())
        return f"MultipleTermStorge({self.name}, [{names}])"

    def __repr__(self):
        names = ", ".join(self.keys())
        return f"MultipleTermStorge({self.name}, [{names}])"


class TermStorage(UserList):

    def __init__(self, name, data=None):
        self.name = name
        if data is None:
            data = []
        UserList.__init__(self, data)

    def __str__(self):
        return f"TermStorage({self.name})"

    def __repr__(self):
        return f"TermStorage({self.name})"

    def remove_term(self, name, atomids=None):
        self.data = list(self.fulfill(name, atomids, notthis=True))

    def fulfill(self, name, atomids=None, notthis=False):

        if atomids is None:
            condition = lambda term: term == name
        else:
            condition = lambda term: (term == name and all(termid == idx
                                      for termid, idx in zip(term.atomids, atomids)))
        if notthis is True:
            return filter(lambda term: not condition(term), self.data)

        return filter(condition, self.data)

    @classmethod
    def new_storage(cls, name, data=None):
        return cls(name, data)

    def get_subset(self, fragment, mapping, remove_non_bonded=[]):

        newstorage = self.new_storage(self.name)
        for term in self:
            if set(term.atomids).issubset(fragment) and (term.name != 'NonBondedTerm' or
                                                         all([idx not in remove_non_bonded for
                                                              idx in term.atomids])):
                newterm = deepcopy(term)
                newterm.atomids = np.array([mapping[i] for i in term.atomids])
                newstorage.append(newterm)
        return newstorage
