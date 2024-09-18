"""Public commandline interface for QForce"""
from colt import Plugin
from colt.validator import Validator
#
from calkeeper import CalculationIncompleteError
#
from .initialize import initialize as _initialize
from .main import runjob, save_jobs, runspjob
from .main import load_keeper, write_bashscript
from .misc import check_if_file_exists, LOGO
from .logger import LoggerExit


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
        return cls.plugin_from_config(_config['mode'])


def initialize(config):
    return _initialize(config['file'], config['options'], None)


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
        runspjob(self.config, self.job)


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
        print(LOGO)
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
        self.job.logger.info(LOGO)
        methods = {name: calculator.run for name, calculator in self.job.calculators.items()}
        ncores = self.config.qm.n_proc

        running = True
        while running:
            try:
                runjob(self.config, self.job)
                running = False
            except CalculationIncompleteError:
                self.job.calkeeper.do_calculations(methods, ncores)
            except LoggerExit:
                self.job.calkeeper.do_calculations(methods, ncores)
        save_jobs(self.config, self.job)


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
        print(LOGO)
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

    def __init__(self, config, job, filename):
        self.config = config
        self.job = job
        self.filename = filename

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job, _config['filename'])

    def run(self):
        print(LOGO)
        print(f"Creating {self.filename}...")
        write_bashscript(self.filename, self.config, self.job)


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
