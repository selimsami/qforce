from colt import Plugin
#
from .qm.qm_base import Calculation, check


class Option(Plugin):
    """helper options"""
    _is_plugin_factory = True
    _plugins_storage = 'options'

    _user_input = """
    option =
    """

    @classmethod
    def _extend_user_input(cls, questions):
        cls.options = {value.name: value for key, value in cls.options.items()}
        questions.add_branching("option", {option.name: option.colt_user_input
                                           for name, option in cls.options.items()})

    def run(self):
        raise NotImplementedError("Method currently not available")

    def from_config(cls, config, calculations):
        return cls.plugin_from_config(config['option'], calculations)


class Check(Option):

    name = 'check'
    _user_input = ""
    _colt_description = 'Check if all the coords'
    __slots__ = ['calculations']

    def __init__(self, calculations):
        self.calculations = calculations

    @classmethod
    def from_config(cls, config, calculations):
        return cls(calculations)

    def run(self):
        check(self.calculations)


class Bash(Option):

    name = 'bash'
    _colt_description = 'Generate a bash'
    _user_input = """
    # basic bash file that will be written
    filename = :: str
    """
    __slots__ = ['filename']

    def __init__(self, calculations, filename):
        self.filename = filename
        self.calculations = calculations

    @classmethod
    def from_config(cls, config, calculations):
        return cls(calculations, config['filename'])
