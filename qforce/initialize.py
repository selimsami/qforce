import os
import shutil
from io import StringIO
from types import SimpleNamespace
import pkg_resources
from colt import Colt
from calkeeper import CalculationKeeper
#
from .schemes import Computations_
from .qm.qm import QM, calculators
from .forcefield.forcefield import ForceField
from .dihedral_scan import DihedralScan
from .pathkeeper import Pathways
from .logger import QForceLogger


class Initialize(Colt):
    _user_input = """
[logging]
# if given log qforce to filename
filename = :: str, optional
# write a bash script each step
write_bash = True :: bool
"""

    @classmethod
    def _set_config(cls, config):
        output_software = config['ff']['output_software']
        config.update({'terms': config['ff'][output_software]})
        #
        config.update({key: SimpleNamespace(**val) for key, val in config.items()})
        return SimpleNamespace(**config)

    @classmethod
    def _extend_user_input(cls, questions):
        questions.generate_block("qm", QM.colt_user_input)
        questions.generate_block("ff", ForceField.colt_user_input)
        questions.generate_block("scan", DihedralScan.colt_user_input)
        questions.generate_block("addstructs", Computations_.colt_user_input)
        # ff terms
        for name, ffcls in ForceField.implemented_md_software.items():
            questions.generate_block(name, ffcls.get_questions(), block='ff')
        # calculator block
        questions.generate_block("calculators", "")
        for name, calculator in calculators.items():
            questions.generate_block(name, calculator.colt_user_input, block='calculators')

    @classmethod
    def from_config(cls, config):
        return cls._set_config(config)

    @staticmethod
    def set_basis(value):
        if value.endswith('**'):
            return f'{value[:-2]}(D,P)'.upper()
        if value.endswith('*'):
            return f'{value[:-1]}(D)'.upper()
        return value.upper()

    @staticmethod
    def set_dispersion(value):
        if value.lower() in ["no", "false", "n", "f"]:
            return False
        return value.upper()


def _get_job_info(filename):
    job = {}
    filename = filename.rstrip('/')
    base = os.path.basename(filename)
    path = os.path.dirname(filename)
    if path != '':
        path = f'{path}/'

    if os.path.isfile(filename):
        job['coord_file'] = filename
        job['name'] = base.split('.')[0]
    else:
        job['coord_file'] = False
        job['name'] = base.split('_qforce')[0]

    job['dir'] = f'{path}{job["name"]}_qforce'
    pathways = Pathways(job['dir'], name=job['name'])
    job['pathways'] = pathways
    job['frag_dir'] = str(pathways["fragments"])
    #
    if job['coord_file'] is False:
        init = pathways['init.xyz']
        if init.exists():
            job['coord_file'] = init
        else:
            preopt = pathways['preopt.xyz']
            if preopt.exists():
                job['coord_file'] = preopt
            else:
                raise SystemExit(f"Either '{pathways['init.xyz']}' "
                                 f"or '{pathways['preopt.xyz']}' need to be present")

    # Added calkeeper
    job['logger'] = None
    job['calculators'] = {}
    job['calkeeper'] = CalculationKeeper()
    job['Calculation'] = job['calkeeper'].get_calcls()
    #
    job['md_data'] = pkg_resources.resource_filename('qforce', 'data')
    os.makedirs(job['dir'], exist_ok=True)
    return SimpleNamespace(**job)


def _check_and_copy_settings_file(pathways, config_file):
    """
    If options are provided as a file, copy that to job directory.
    If options are provided as StringIO, write that to job directory.
    """

    settings_file = pathways['settings.ini']

    if config_file is not None:
        if isinstance(config_file, StringIO):
            with open(settings_file, 'w') as fh:
                config_file.seek(0)
                fh.write(config_file.read())
        else:
            shutil.copy2(config_file, settings_file)

    return settings_file


def initialize(filename, config_file, presets=None):
    job_info = _get_job_info(filename)
    settings_file = _check_and_copy_settings_file(job_info.pathways, config_file)

    config = Initialize.from_questions(config=settings_file, presets=presets, check_only=True)
    # get calculators
    for key, Calculator in calculators.items():
        job_info.calculators[key] = Calculator.from_config(getattr(config.calculators, key))
    job_info.logger = QForceLogger(config.logging.filename)
    return config, job_info
