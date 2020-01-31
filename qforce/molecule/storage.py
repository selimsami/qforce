from collections.abc import MutableSequence
from .base import MappingIterator


class MultipleTermStorge(MappingIterator):

    def __init__(self, name, dct):
        self.name = name
        MappingIterator.__init__(self, dct)


class TermStorage(MutableSequence):

    def __init__(self, name, data=None):
        self.name = name
        if data is None:
            self._data = []
        else:
            self._data = data

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value

    def __delitem__(self, idx):
        del self._data[idx]

    def __len__(self):
        return len(self._data)

    def insert(self, idx, ele):
        self._data.insert(idx, ele)
