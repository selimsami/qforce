"""Public commandline interface for QForce"""
from colt import Plugin
from colt.validator import Validator
#
from calkeeper import CalculationKeeper, CalculationIncompleteError
#
from .initialize import initialize as _initialize
from .main import runjob
from .qm.qm_base import KEEPER
from .misc import check_if_file_exists, LOGO


# define new validator
Validator.overwrite_validator("file", check_if_file_exists)


STANDARD_USER_INPUT = """
# Input coordinate file mol.ext (ext: pdb, xyz, gro, ...)
# or directory (mol or mol_qforce) name.
file = :: file

# File name for the optional options.
options = :: file, optional, alias=o
"""


class Option(Plugin):
    """helper options"""
    _is_plugin_factory = True
    _plugins_storage = 'options'

    _user_input = """
    # what to do
    mode =
    """

    @classmethod
    def _extend_user_input(cls, questions):
        cls.options = {value.name: value for key, value in cls.options.items()}
        questions.add_branching("mode", {option.name: option.colt_user_input
                                         for name, option in cls.options.items()})

    def run(self):
        raise NotImplementedError("Method currently not available")

    @classmethod
    def from_config(cls, _config):
        print(LOGO)
        return cls.plugin_from_config(_config['mode'])


def initialize(config):
    return _initialize(config['file'], config['options'], None)


def save_jobs(job):
    with open(job.pathways['calculations.json'], 'w') as fh:
        fh.write(KEEPER.as_json())


class RunQforce(Option):

    name = 'run'
    _user_input = STANDARD_USER_INPUT

    _colt_description = 'run qforce manually'
    __slots__ = ['config', 'job']

    def __init__(self, config, job):
        self.config = config
        self.job = job

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job)

    def run(self):
        try:
            runjob(self.config, self.job)
            save_jobs(self.job)
        except CalculationIncompleteError:
            save_jobs(self.job)


def load_keeper(job):
    file = job.pathways['calculations.json']
    if file.exists():
        with open(file, 'r') as fh:
            keeper = CalculationKeeper.from_json(fh.read())
        return keeper
    raise SystemExit(f"No calculation for '{job.dir}'")


class Check(Option):

    name = 'check'
    _user_input = STANDARD_USER_INPUT
    _colt_description = 'Check if the calculations are done or not'
    __slots__ = ['keeper']

    def __init__(self, keeper):
        self.keeper = keeper

    @classmethod
    def from_config(cls, config):
        _, job = initialize(config)
        return cls(load_keeper(job))

    def run(self):
        self.keeper.check()
        print("All checks have passed!")


class RunQforceComplete(Option):

    name = 'run_complete'
    _colt_description = 'Run qforce automatically till completion'
    _user_input = STANDARD_USER_INPUT
    __slots__ = ['config', 'job']

    def __init__(self, config, job):
        self.config = config
        self.job = job

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job)

    def run(self):
        running = True
        while running:
            try:
                runjob(self.config, self.job)
                running = False
            except CalculationIncompleteError:
                KEEPER.do_calculations()
            except SystemExit:
                KEEPER.do_calculations()
        save_jobs(self.job)


class Literature(Option):

    name = 'citations'
    _colt_description = 'Print basic citations'
    _user_input = """
    format = txt :: str, alias=f :: txt, tex
    """
    __slots__ = ['format']

    paper = {
            'qforce2021': {
                'txt': (
                    """Q-Force: Quantum Mechanically Augmented Molecular Force Fields
Selim Sami, Maximilian F.S.J Menger, Shirin Faraji, Ria Broer, and Remco W. A. Havenith
Journal of Chemical Theory and Computation 2021 17 (8), 4946-4960
DOI: 10.1021/acs.jctc.1c00195"""),

                'tex': (
                    """@article{QForceJCTC2021,
author = {Sami, Selim and Menger, Maximilian F.S.J and Faraji, Shirin and Broer, Ria and Havenith, Remco W. A.},
title = {Q-Force: Quantum Mechanically Augmented Molecular Force Fields},
journal = {Journal of Chemical Theory and Computation},
volume = {17},
number = {8},
pages = {4946-4960},
year = {2021},
doi = {10.1021/acs.jctc.1c00195},
}"""),
                }
            }

    def __init__(self, _format):
        self._format = _format

    @classmethod
    def from_config(cls, config):
        return cls(config['format'])

    def run(self):
        print(self.paper['qforce2021'][self._format])


class Bash(Option):

    name = 'bash'
    _colt_description = 'Generate a bash'
    _user_input = STANDARD_USER_INPUT + """
    # basic bash file that will be written
    filename = :: str
    """
    __slots__ = ['filename', 'keeper']

    def __init__(self, keeper, filename):
        self.filename = filename
        self.keeper = keeper

    @classmethod
    def from_config(cls, config):
        _, job = initialize(config)
        return cls(load_keeper(job), config['filename'])

    def run(self):
        with open(self.filename, 'w') as fh:
            fh.write("current=$PWD\nfor folder in ")
            fh.write("  ".join(str(calc.folder) for calc in self.keeper.get_incomplete()))
            fh.write(";\ndo\n    cd ${folder}\n\n\n    cd ${current}\ndone\n")


def run():
    """Run qforce calculation from commandline"""
    Option.from_commandline(description={'logo': LOGO,
                                         'alias': 'qforce',
                                         'arg_format': {'name': 18,
                                                        'comment': 62,
                                                        },
                                         'subparser_format': {'name': 18, 'comment': 62},
                                         }
                            ).run()
