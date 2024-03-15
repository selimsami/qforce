import os
import shutil
from io import StringIO
from types import SimpleNamespace
import pkg_resources
#
from colt import Colt
#
from .qm.qm import QM, implemented_qm_software
from .forcefield.forcefield import ForceField
from .molecule.terms import Terms
from .dihedral_scan import DihedralScan
from .misc import LOGO


class Initialize(Colt):
    @staticmethod
    def _set_config(config):
        config['qm'].update(config['qm']['software'])
        config['qm'].update({'software': config['qm']['software'].value})
        config.update({key: SimpleNamespace(**val) for key, val in config.items()})
        return SimpleNamespace(**config)

    @classmethod
    def _extend_user_input(cls, questions):
        questions.generate_block("qm", QM.colt_user_input)
        questions.generate_block("ff", ForceField.colt_user_input)
        questions.generate_block("scan", DihedralScan.colt_user_input)
        questions.generate_cases("software", {key: software.colt_user_input for key, software in
                                              implemented_qm_software.items()}, block='qm')
        questions.generate_block("terms", Terms.get_questions())

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
    job['frag_dir'] = f'{job["dir"]}/fragments'
    job['md_data'] = pkg_resources.resource_filename('qforce', 'data')
    os.makedirs(job['dir'], exist_ok=True)
    return SimpleNamespace(**job)


def _check_and_copy_settings_file(job_dir, config_file):
    """
    If options are provided as a file, copy that to job directory.
    If options are provided as StringIO, write that to job directory.
    """

    settings_file = os.path.join(job_dir, 'settings.ini')

    if config_file is not None:
        if isinstance(config_file, StringIO):
            with open(settings_file, 'w') as fh:
                config_file.seek(0)
                fh.write(config_file.read())
        else:
            shutil.copy2(config_file, settings_file)

    return settings_file


def initialize(filename, config_file, presets=None):
    print(LOGO)

    job_info = _get_job_info(filename)
    settings_file = _check_and_copy_settings_file(job_info.dir, config_file)

    config = Initialize.from_questions(config=settings_file, presets=presets, check_only=True)

    return config, job_info
