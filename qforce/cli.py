"""Public commandline interface for QForce"""
import os
from pathlib import Path
from hashlib import md5
from shutil import copy2 as copy

from colt import Plugin
from colt.validator import Validator
#
from calkeeper import CalculationKeeper, CalculationIncompleteError
#
from .initialize import initialize as _initialize
from .main import runjob, save_jobs, runspjob, runjob_v2
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
    _user_input = STANDARD_USER_INPUT + """
    version = 1 :: int, alias=v :: [1, 2]
    """

    _colt_description = 'run qforce manually'
    __slots__ = ['config', 'job']

    def __init__(self, config, job, version):
        self.config = config
        self.job = job
        self.version = version

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job, _config['version'])

    def run(self):
        if self.version == 1:
            runspjob(self.config, self.job)
        else:
            runspjob(self.config, self.job, v2=True)


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
        print(LOGO)
        self.keeper.check()
        print("All checks have passed!")


class RunQforceComplete(Option):

    name = 'run_complete'
    _colt_description = 'Run qforce automatically till completion'
    _user_input = STANDARD_USER_INPUT + """
    version = 1 :: int, alias=v :: [1, 2]
    """
    __slots__ = ['config', 'job']

    def __init__(self, config, job, version):
        self.config = config
        self.job = job
        self.version = version

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job, _config['version'])

    def run(self):
        self.job.logger.info(LOGO)
        methods = {name: calculator.run for name, calculator in self.job.calculators.items()}
        ncores = self.config.qm.n_proc

        running = True
        while running:
            try:
                if self.version == 1:
                    runjob(self.config, self.job)
                else:
                    runjob_v2(self.config, self.job)
                running = False
            except CalculationIncompleteError:
                self.job.calkeeper.do_calculations(methods, ncores)
            except LoggerExit:
                self.job.calkeeper.do_calculations(methods, ncores)
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

    def __init__(self, keeper, filename):
        self.filename = filename
        self.keeper = keeper

    @classmethod
    def from_config(cls, config):
        _, job = initialize(config)
        return cls(load_keeper(job), config['filename'])

    def run(self):
        print(LOGO)
        print(f"Creating {self.filename}...")
        with open(self.filename, 'w') as fh:
            fh.write("current=$PWD\nfor folder in ")
            fh.write("  ".join(str(calc.folder) for calc in self.keeper.get_incomplete()))
            fh.write(";\ndo\n    cd ${folder}\n\n\n    cd ${current}\ndone\n")


class Bash2(Option):

    name = 'bash2'
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
        self.keeper = load_keeper(job)

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job, _config['filename'])

    def run(self):
        print(LOGO)
        print(f"Creating {self.filename}...")
        methods = {name: calculator.as_string for name, calculator in self.job.calculators.items()}
        ncores = self.config.qm.n_proc
        with open(self.filename, 'w') as fh:
            fh.write("current=$PWD\n")
            for calc in self.keeper.get_incomplete():
                call = methods.get(calc.software, None)
                if call is None:
                    raise ValueError("Call unknown!")
                fh.write(call(calc, ncores))


class Bash3In(Option):

    name = 'bash3in'
    _colt_description = 'Generate a bash'
    _user_input = STANDARD_USER_INPUT + """
    # basic bash file that will be written
    folder = :: str
    """
    __slots__ = ['filename', 'keeper']

    def __init__(self, config, job, folder):
        self.config = config
        self.job = job
        self.folder = folder
        self.keeper = load_keeper(job)

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job, _config['folder'])

    def run(self):
        print(LOGO)
        print(f"Creating  {self.folder}")
        methods = {name: calculator.as_minimal_string for name, calculator in self.job.calculators.items()}
        ncores = self.config.qm.n_proc
        main = Path(self.folder)
        with open(main / 'run.sh', 'w') as fh:
            fh.write("current=$PWD\nfor folder in ")
            for calc in self.keeper.get_incomplete():
                call = methods.get(calc.software, None)
                if call is None:
                    raise ValueError("Call unknown!")
                folder = os.path.abspath(calc.folder)
                nfolder = str(md5(folder.encode()).hexdigest())
                fh.write(f" {nfolder} ")
                newfolder = main / nfolder
                os.makedirs(newfolder, exist_ok=True)
                for file in Path(folder).iterdir():
                    if file.is_file():
                        copy(file, newfolder)
                with open(newfolder / 'run.sh', 'w') as nfh:
                    nfh.write(call(calc, ncores))
            fh.write(";\ndo\n    cd ${folder}\n\n    bash run.sh\n\n    cd ${current}\ndone\n")


class Bash3Out(Option):

    name = 'bash3out'
    _colt_description = 'Copy output files back, from a folder'
    _user_input = STANDARD_USER_INPUT + """
    # basic bash file that will be written
    folder = :: str
    """
    __slots__ = ['filename', 'keeper']

    def __init__(self, config, job, folder):
        self.config = config
        self.job = job
        self.folder = folder
        self.keeper = load_keeper(job)

    @classmethod
    def from_config(cls, _config):
        config, job = initialize(_config)
        return cls(config, job, _config['folder'])

    def run(self):
        print(LOGO)
        print(f"Copy from {self.folder}")
        main = Path(self.folder)
        for calc in self.keeper.get_incomplete():
            folder = os.path.abspath(calc.folder)
            nfolder = str(md5(folder.encode()).hexdigest())
            newfolder = main / nfolder
            if newfolder.exists():
                for file in newfolder.iterdir():
                    if file.is_file():
                        copy(file, folder)
            else:
                print(f"Could not find folder for {calc.folder}")


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
