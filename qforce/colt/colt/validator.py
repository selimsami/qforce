"""Basic Validator to convert user input into python objects
while automatically checking the type and doing error handling"""
import os
import ast
from collections.abc import KeysView
#
import numpy as np


__all__ = ["NOT_DEFINED", "Validator", "ValidatorErrorNotInChoices"]


class NoChoice:
    """Empty object that indicates that no choice are used"""

    __slots__ = ()

    @staticmethod
    def is_subset(rhs):
        """Is subset of itself"""
        if isinstance(rhs, NoChoice):
            return True
        return False

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    @staticmethod
    def validate(value):
        """Always True"""
        return True


NO_CHOICE = NoChoice()


class Choices:
    """Store possible choices"""

    __slots__ = ('choices',)

    def __init__(self, choices):
        self.choices = choices

    def as_str(self):
        """return choices as a string"""
        txt = ", ".join(str(choice) for choice in self.choices)
        return f"Choices({txt})"

    def __len__(self):
        return len(self.choices)

    def __str__(self):
        return self.as_str()

    def __repr__(self):
        return self.as_str()

    def __getitem__(self, idx):
        return self.choices[idx]

    def __iter__(self):
        return iter(self.choices)

    def as_list(self):
        """return choices as a python list"""
        return list(self.choices)

    def validate(self, value):
        """check if value in choices"""
        return value in self.choices

    def is_subset(self, rhs):
        """check if `rhs` is a subset of `self`"""
        if rhs in (None, NO_CHOICE):
            return True
        if not isinstance(rhs, Choices):
            return False
        return all(choice in rhs.choices for choice in self.choices)


class RangeExpression:
    """Simple class to handle mathematical ranges"""

    __slots__ = ('lower', 'upper', 'expr')

    def __init__(self, expr):
        self.expr = expr
        self.lower, self.upper = self._parse(expr)

    def validate(self, value):
        """check that value fullfills the expression"""
        if self.lower is not None:
            if value < self.lower:
                return False
        if self.upper is not None:
            if value > self.upper:
                return False
        return True

    def __str__(self):
        return self.as_str()

    def __repr__(self):
        return self.as_str()

    def as_str(self):
        """Return str of range expression"""
        return f"Range({self.expr})"

    def is_subset(self, rhs):
        """Check if `rhs` is a subset of `self`"""
        if rhs is None:
            return True
        if not isinstance(rhs, RangeExpression):
            return False

        if self._smaller(self.lower, rhs.lower):
            return False
        if self._larger(self.upper, rhs.upper):
            return False
        return True

    def _parse(self, expr):
        if '>' in expr:
            upper, lower = (ele.strip() for ele in expr.split('>'))
        elif '<' in expr:
            lower, upper = (ele.strip() for ele in expr.split('<'))
        else:
            raise ValueError("Could not parse Expr")
        upper = self._parse_value(upper)
        lower = self._parse_value(lower)
        if self._larger(lower, upper):
            raise ValueError("Range is None")
        return lower, upper

    @staticmethod
    def _larger(value1, value2):
        """value1 > value2"""
        if value1 is None:
            return True
        if value2 is None:
            return False
        return value1 > value2

    @staticmethod
    def _smaller(value1, value2):
        """value1 < value2"""
        if value1 is None:
            return False
        if value2 is None:
            return True
        return value1 < value2

    @staticmethod
    def _parse_value(value):
        """convert value to float"""
        if value == "":
            return None
        return float(value)


def abspath(answer):
    """abspath including expanduser"""
    return os.path.abspath(os.path.expanduser(answer))


def file_exists(path):
    """check if file exists"""
    path = abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"File does not exisit '{path}'")
    return path


def folder_exists(path):
    """check if folder exists"""
    path = abspath(path)
    if not os.path.isdir(path):
        raise ValueError(f"Folder does not exisit '{path}'")
    return path


def non_existing_path(path):
    """check that there is nothing at the given path"""
    path = abspath(path)
    if os.path.exists(path):
        raise ValueError(f"File/Folder does already exisit '{path}'")
    return path


def bool_parser(answer):
    """convert string into a bool"""

    _positive = ('y', 'yes', 'True', 'true')
    _negative = ('n', 'no', 'False', 'false')
    #
    if answer in _positive:
        return True
    if answer in _negative:
        return False

    raise ValueError("Answer can only be [%s] or [%s]"
                     % (", ".join(_positive), ", ".join(_negative)))


def list_parser(answer):
    """convert string to list of strings"""
    split_char, answer = _prepare_list_parsing(answer)
    return [ele.strip() for ele in answer.split(split_char) if ele.strip() != ""]


def flist_parser(answer):
    """convert string to list of floats"""
    return [float(ele) for ele in list_parser(answer)]


def flist_np_parser(answer):
    """Transfers a string into a numpy array of floats"""
    return np.array(flist_parser(answer))


def ilist_parser(answer):
    """convert string to list of integers"""
    if '~' not in answer:
        return [int(ele) for ele in list_parser(answer)]
    #
    split_char, answer = _prepare_list_parsing(answer)
    numbers = (parse_integer_numbers(number) for number in answer.split(split_char))
    return sum(numbers, [])


def ilist_np_parser(answer):
    """convert string to numpy array of integers"""
    return np.array(ilist_parser(answer))


def get_upper_bounds(start, stop):
    """get upper bound for range"""
    start, stop = int(start), int(stop)
    if start > stop:
        start, stop = stop, start
    return start, stop+1


def parse_integer_numbers(string):
    """convert integer numbers into ilist liste"""
    # check if tilde in line
    if '~' in string:
        start, _, stop = string.partition('~')
        start, stop = get_upper_bounds(start, stop)
        return list(range(start, stop))
    if string.strip() == "":
        return []
    return [int(string)]


def _prepare_list_parsing(answer):
    """setup string for list parsing"""
    split_char = choose_split_char(answer)
    answer = remove_brackets_and_quotes(answer)
    return split_char, answer


def choose_split_char(answer):
    """get a suitable split character"""
    if ',' in answer:
        split_char = ","
    else:
        split_char = None
    return split_char


def remove_brackets_and_quotes(string):
    """remove brackets from string"""
    return string.replace("[", "").replace("]", "").replace("'", "").replace('"', "")


def _as_python_object(string, obj, name):
    """convert the string to an python object using ast.literal_eval"""
    result = ast.literal_eval(string)
    if not isinstance(result, obj):
        raise ValueError(f"Value is not a python {name}!")
    return result


def as_python_tuple(string):
    """try to convert string to python tuple"""
    return _as_python_object(string, tuple, "tuple")


def as_python_list(string):
    """try to convert string to python list"""
    return _as_python_object(string, list, "list")


def as_python_dict(string):
    """try to convert string to python list"""
    return _as_python_object(string, dict, "dict")


def as_python_numpy_array(string):
    """try to convert string to python list"""
    return np.array(as_python_list(string))


# empty class
class NotDefined:
    """Empty class to use something analogue to None"""
    __slots__ = ()

    def __repr__(self):
        return "<NOT_DEFINED>"

    def __str__(self):
        return "<NOT_DEFINED>"


NOT_DEFINED = NotDefined()


class ValidatorErrorNotInChoices(Exception):
    """Exception in case value is not in choices"""


class ValidatorErrorNotChoicesSubset(Exception):
    """Exception in case the new choices are not an subset of the old ones"""


class ValidatorBase:
    """Base class to validator"""

    __slots__ = ('_choices', '_value', '_string')
    # overwrite this method

    def __init__(self, default=NOT_DEFINED, choices=None):
        self._string = NOT_DEFINED
        self._choices = self._set_choices(choices)
        self._value = self._set_value(default)

    @staticmethod
    def _parse(string):
        """Abstract method to be overwritten"""

    def validate(self, value):
        """Parse a string and return its value,
        raises ValueError on failure

        Parameters
        ----------
        value: str
            user input to be validated

        Returns
        -------
        Parsed Value

        Raises
        ------
        ValueError
            if value does not fullfill condition
        """
        value = self._parse(str(value))
        if not self._choices.validate(value):
            raise ValidatorErrorNotInChoices("Answer is not in {self._choices}")
        return value

    def answer_as_string(self):
        """return the answer as a string"""
        if self._string is NOT_DEFINED:
            return ''
        return self._string

    def get(self):
        """Return self._value if its set or not!"""
        return self._value

    def set(self, value):
        """set the value"""
        self._value = self._get_value(value)

    @property
    def choices(self):
        """Return choices"""
        if self._choices is NO_CHOICE:
            return None
        return self._choices

    @choices.setter
    def choices(self, choices):
        """  """
        # will raise an value error if choices wrong!
        choices = self._set_choices(choices)
        # check that the new choices are a subset of the old ones
        if not choices.is_subset(self._choices):
            raise ValidatorErrorNotChoicesSubset(("cannot update choices,",
                                                  " needs to be subset of the original ones"))
        # overwrite existing ones
        self._choices = choices
        # validate choice, if existing default is not in choices, reset
        if not self._choices.validate(self._value):
            self._value = NOT_DEFINED

    def _set_value(self, value):
        if value is not NOT_DEFINED:
            value = self._get_value(value)
        return value

    def _set_choices(self, in_choices):
        """set choices"""
        if in_choices is None:
            return NO_CHOICE
        return self.set_choices(in_choices)

    def set_choices(self, in_choices):
        """set choices"""
        if isinstance(in_choices, KeysView):
            return Choices([self._parse(choice) for choice in in_choices])
        try:
            choices = [self._parse(choice) for choice in list_parser(in_choices)]
        except ValueError:
            choices = None
        if choices is None:
            raise ValueError(f"Choices '{in_choices}' cannot be parsed")
        return Choices(choices)

    def _get_value(self, string):
        value = self.validate(string)
        if not self._choices.validate(value):
            raise ValidatorErrorNotInChoices("Answer is not in choices")
        self._string = string
        return value


class RangeValidator(ValidatorBase):
    """Validator that allowes both `Choices` and RangeExpression"""

    def set_choices(self, in_choices):
        """set choices"""
        if isinstance(in_choices, KeysView):
            return Choices([self._parse(choice) for choice in in_choices])
        #
        try:
            return RangeExpression(in_choices)
        except ValueError:
            pass
        #
        try:
            choices = [self._parse(choice) for choice in list_parser(in_choices)]
        except ValueError:
            choices = None
        #
        if choices is None:
            raise ValueError(f"Choices '{in_choices}' cannot be parsed")
        return Choices(choices)


def create_validators(base_validators, range_validators):
    """simple helper to generate validators from a dictionary

    Parameters
    ----------
    base_validators: dict
        all types that get the `ValidatorBase`

    range_validators: dict
        all types that get the `RangeValidator`

    Returns
    -------
    dict
        all validators in a single dictionary
    """
    out = {}
    for name, parser in base_validators.items():
        parser_name = name.capitalize() + "Validator"
        out[name] = type(parser_name, (ValidatorBase,), {'_parse': staticmethod(parser)})
    for name, parser in range_validators.items():
        parser_name = name.capitalize() + "Validator"
        out[name] = type(parser_name, (RangeValidator,), {'_parse': staticmethod(parser)})
    return out


class Validator:
    """Factory class"""

    parsers = create_validators(
        {'str': str,
         'bool': bool_parser,
         'list': list_parser,
         'ilist': ilist_parser,
         'ilist_np': ilist_np_parser,
         'flist': flist_parser,
         'flist_np': flist_np_parser,
         'file': abspath,  # return abspath
         'folder': abspath,
         'existing_file': file_exists,
         'existing_folder': folder_exists,
         'non_existing_file': non_existing_path,
         'non_existing_folder': non_existing_path,
         # python objects
         'python(list)': as_python_list,
         'python(dict)': as_python_dict,
         'python(tuple)': as_python_tuple,
         'python(np.array)': as_python_numpy_array,
         }, {'int': int,
             'float': float,
             }
    )

    _validators = {'base': ValidatorBase, 'range': RangeValidator}

    def __new__(cls, typ, default=NOT_DEFINED, choices=None):
        if typ not in cls.parsers:
            raise ValueError(f"Validator '{typ}' not defined")
        return cls.parsers[typ](default=default, choices=choices)

    @classmethod
    def add_validator(cls, name, func, typ='base'):
        """Add a new custom validator.

        Parameters
        ----------
        name: str
            name of the validator typ
        func: function
            validation function, should raise ValueError on fail
        typ: str, optional
            typ of validator, currently: base, range

        Raises
        ------
        ValueError
            In case the typ is unknown
        """
        base_cls = cls._validators.get(typ, None)
        if base_cls is None:
            raise ValueError(f"Validator type '{typ}' not known")
        parser_name = name.capitalize() + "Validator"
        cls.parsers[name] = type(parser_name, (base_cls,), {'_parse': staticmethod(func)})

    @classmethod
    def remove_validator(cls, name):
        """Remove validator """
        if name in cls.parsers:
            del cls.parsers[name]
