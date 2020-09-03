class Type:
    """container class for all existing types"""

    types = {}

    def __new__(cls, name, alias=None):
        typ = cls.types.get(name, None)
        if typ is not None:
            return typ
        typ = _TypeImpl(name)
        cls.types[name] = typ
        #
        if alias is None:
            return typ
        #
        for _name in alias:
            cls.types[_name] = typ
        return typ

    @classmethod
    def add_subtypes(cls, parent, subtypes):
        typ = cls.types[parent]
        if isinstance(subtypes, str):
            typ.add_subtype(cls.types[subtypes])
        else:
            for subtype in subtypes:
                typ.add_subtype(cls.types[subtype])


class _TypeImpl:
    """Actual type definition"""

    def __init__(self, typ):
        self.typ = typ
        self._ele = []

    def __str__(self):
        return f"Type({self.typ})"

    def __repr__(self):
        return f"Type({self.typ})"

    def __len__(self):
        return sum(len(ele) for ele in self._ele)

    def add_subtype(self, other):
        if not isinstance(other, _TypeImpl):
            raise Exception("can only add subtypes")
        if other not in self._ele:
            self._ele.append(other)

    def __iter__(self):
        for ele in self._ele:
            yield ele

    def __contains__(self, other):
        return any(other.is_type(ele) for ele in self._ele)

    def is_type(self, other):
        if not isinstance(other, _TypeImpl):
            raise Exception("can only add subtypes")
        #
        if other.typ == 'anything':
            return True
        #
        if other is self:
            return True
        #
        return self in other


# general typ that could be anything ;)
Type('anything')
