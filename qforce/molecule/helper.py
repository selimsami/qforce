from collections import UserDict


class DefaultFalseDict(UserDict):
    """Dictory that return False in case a key is not defined"""

    def get(self, key, default=False):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data.get(key, False)
