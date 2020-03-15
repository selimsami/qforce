from collections import UserList
from copy import deepcopy
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

    @classmethod
    def new_storage(cls, name, data=None):
        return cls(name, data)

    def get_subset(self, fragment, mapping):
        newstorage = self.new_storage(self.name)
        for term in self:
            if set(term.atomids).issubset(fragment):
                newterm = deepcopy(term)
                newterm.atomids = np.array([mapping[i] for i in term.atomids])
                newstorage.append(newterm)
        return newstorage
