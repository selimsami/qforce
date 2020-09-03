""" Create slottedcls in analogue to a namedtuple"""
from collections.abc import Mapping
from .validator import NOT_DEFINED


class Snone:
    """empty object used as own None"""
    __slots__ = ()


SNONE = Snone()


def slottedcls(name, entries):
    """create a slotted class in analogue to a namedtuple """

    if isinstance(entries, str):
        entries = (entries, )

    if isinstance(entries, Mapping):
        _slots = tuple(entries.keys())
        _args = tuple(name for name, value in entries.items()
                      if value is SNONE)
        _kwargs = []
        for _name, value in entries.items():
            if value is SNONE:
                continue
            if isinstance(value, str):
                string = f"{_name}='{value}'"
            elif value is NOT_DEFINED:
                string = f"{_name}=NOT_DEFINED"
            else:
                string = f"{_name}={value}"
            _kwargs.append(string)
    else:
        _slots = tuple(list(entries))
        _args = _slots
        _kwargs = {}

    if len(_args) == 0:
        args_str = ""
    else:
        args_str = f'{", ".join(arg for arg in _args)}'
    if len(_kwargs) == 0:
        kwargs_str = ""
    else:
        kwargs_str = f'{", ".join(kwarg for kwarg in _kwargs)}'

    args_str = ", ".join(arg for arg in (args_str, kwargs_str) if arg != "")

    attributes = tuple(f'    setattr(self, "{name}", {name})\n' for name in _slots)

    _init = f"def __init__(self, {args_str}):\n{''.join(setattr for setattr in attributes)}"

    def __repr__(self):
        return (f"{self._name}("
                + (", ".join(f"{slot}={getattr(self, slot)}" for slot in self.__slots__))
                + ")")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.__slots__ != other.__slots__:
            return False
        return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return True
        if self.__slots__ != other.__slots__:
            return True
        return any(getattr(self, slot) != getattr(other, slot) for slot in self.__slots__)

    functions = {'name': name, 'NOT_DEFINED': NOT_DEFINED}

    exec(_init, functions)

    return type(name, (),
                {'__slots__': _slots,
                 '_name': name,
                 '__init__': functions['__init__'],
                 '__str__': __repr__,
                 '__repr__': __repr__,
                 '__eq__': __eq__,
                 '__ne__': __ne__,
                 })
