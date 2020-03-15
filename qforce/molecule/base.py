from collections.abc import Mapping
import re


class MappingIterator(Mapping):
    """Base class to store recursively data"""

#    _regex_substring = re.compile(r"(?P<key>\w+)\((?P<subkey>.*)\)")
#    _key_subkey = '%s(%s)'
    _regex_substring = re.compile(r"(?P<key>\w+)\/(?P<subkey>.*)")
    _key_subkey = '%s/%s'

    def __init__(self, dct, ignore=tuple()):
        self._data = dct
        self.ignore = set(self._set_ignore(dct, ignore))

    @classmethod
    def get_key_subkey(cls, string):
        match = cls._regex_substring.match(string)
        if match is None:
            return string, None
        return match.group('key'), match.group('subkey')

    def ho_keys(self):
        return self._data.keys()

    def ho_items(self):
        return self._data.items()

    def ho_values(self):
        return self._data.values()

    def add_ignore_key(self, value):
        regular = self._set_ignore(self._data, [value])
        for key in regular:
            self._add_key(key)

    def remove_ignore_key(self, value):
        regular = self._remove_ignore(self._data, [value])
        for key in regular:
            self._remove_key(key)

    def remove_ignore_keys(self, value):
        regular = self._remove_ignore(self._data, value)
        for key in regular:
            self._remove_key(key)

    def _add_key(self, value):
        self.ignore.add(value)

    def _remove_key(self, value):
        self.ignore.remove(value)

    def add_ignore_keys(self, ignore_keys):
        regular = self._set_ignore(self._data, ignore_keys)
        for key in regular:
            self._add_key(key)

    def _goto_key(self, subkey):
        data = self._data
        while True:
            key, subkey = self.get_key_subkey(subkey)
            if subkey is None:
                break
            data = data[key]
        return key, data

    def _perform_ignore_action(self, dct, ignore, action=lambda *args: None):

        ignore_keys = {'__REGULAR_KEY__': []}
        for ign in ignore:
            key, subkey = self.get_key_subkey(ign)
            if subkey is None:
                ignore_keys['__REGULAR_KEY__'].append(key)
                continue
            if key not in dct:
                print(f"WARNING: '{key}' not known, therefore ignored!")
                continue
            iterm = dct[key]
            if not isinstance(iterm, MappingIterator):
                print(f"WARNING: '{key}({subkey})' cannot be modified therefore ignored!")
                continue
            if key not in ignore_keys:
                ignore_keys[key] = []
            ignore_keys[key].append(subkey)

        for name, keys in ignore_keys.items():
            if name == '__REGULAR_KEY__':
                continue
            action(dct, name, keys)

        return ignore_keys['__REGULAR_KEY__']

    def _set_ignore(self, dct, ignore):
        return self._perform_ignore_action(dct, ignore, action=lambda dct,
                                           name, keys: dct[name].add_ignore_keys(keys))

    def _remove_ignore(self, dct, ignore):
        return self._perform_ignore_action(dct, ignore, action=lambda dct,
                                           name, keys: dct[name].remove_ignore_keys(keys))

    def __getitem__(self, key):
        key, data = self._goto_key(key)
        if isinstance(data, MappingIterator):
            if key in data.ignore:
                raise KeyError
        return data[key]

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def keys(self):
        keys = []
        for key, value in self._data.items():
            if key in self.ignore:
                continue
            if isinstance(value, Mapping):
                for subkey in value.keys():
                    keys.append(self._key_subkey % (key, subkey))
            else:
                keys.append(key)
        return keys

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield (key, self[key])

    def allkeys(self):
        keys = []
        for key, value in self._data.items():
            if isinstance(value, Mapping):
                for subkey in value.keys():
                    keys.append(self._key_subkey % (key, subkey))
            keys.append(key)
        return keys

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        for key, value in self._data.items():
            if key not in self.ignore:
                for rval in value:
                    yield rval
