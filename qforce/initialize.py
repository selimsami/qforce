import os
import sys
import argparse
from types import SimpleNamespace
import pkg_resources
from qforce.fit import fit
from colt import Colt
from .polarize import polarize
from .qm.qm import QM, implemented_qm_software
from .molecule.terms import Terms
from .dihedral_scan import DihedralScan


class Initialize(Colt):
    _questions = """
[ff]
# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent
n_equiv = 4 :: int

# Number of first n neighbors to exclude in the forcefield
n_excl = 2 :: int :: [2, 3]

# Lennard jones method for the forcefield
lennard_jones = gromos_auto :: str :: [gromos_auto, gromos, opls, gaff]

# Use externally provided point charges in the file "ext_q" in the job directyory
ext_charges = no :: bool

# Scale QM charges to account for condensed phase polarization (should be set to 1 for gas phase)
charge_scaling = 1.2 :: float

# Additional exclusions (GROMACS format)
exclusions = :: literal

# Set all dihedrals as rigid (no dihedral scans)
all_rigid = no :: bool

# Use D4 method
_d4 = no :: bool

# Residue name printed on the force field file (Max 5 characters)
res_name = MOL :: str

# Polarize a coordinate file and quit (requires itp_file)
_polarize = no :: bool

# Name of itp file (only needed for polarize option)
_itp_file = itp_file_missing :: str

# Make a polarizable FF
_polar = no :: bool

# Scale the C6 dispersion interactions in the polarizable version of the FF
_polar_c6_scale = 0.8 :: float

# Specifically not scale some of the atoms
_polar_not_scale_c6 = :: literal

# Manual polarizabilities in the file ext_alpha
_ext_alpha = no :: bool

"""

    def __init__(self, config, input_arg):
        config = self._set_config(config)
        job = self._get_job_info(input_arg)

        if config.ff._polarize:
            polarize(job, config.ff)

        qm = QM(job, config.qm)
        fit(qm, config, job)

    def _set_config(self, config):
        config['qm'].update(config['qm']['software'])
        config['qm'].update({'software': config['qm']['software'].value})
        config.update({key: SimpleNamespace(**val) for key, val in config.items()})
        config = SimpleNamespace(**config)
        return config

    @classmethod
    def _extend_questions(cls, questions):
        questions.generate_block("qm", QM.get_questions())
        questions.generate_block("scan", DihedralScan.get_questions())
        questions.generate_cases("software", {key: software.questions for key, software in
                                              implemented_qm_software.items()}, block='qm')
        questions.generate_block("terms", Terms.get_questions())

    @classmethod
    def from_config(cls, config, input_arg):
        return cls(config, input_arg)

    def _get_job_info(self, input_arg):
        job = {}
        input_arg = input_arg.rstrip('/')
        base = os.path.basename(input_arg)
        path = os.path.dirname(input_arg)
        if path != '':
            path = f'{path}/'

        if os.path.isfile(input_arg):
            job['coord_file'] = input_arg
            job['name'] = base.split('.')[0]
        else:
            job['coord_file'] = False
            job['name'] = base.split('_qforce')[0]

        job['dir'] = f'{path}{job["name"]}_qforce'
        job['frag_dir'] = f'{job["dir"]}/fragments'
        job['md_data'] = pkg_resources.resource_filename('qforce', 'data')
        os.makedirs(job['dir'], exist_ok=True)
        return SimpleNamespace(**job)

    @staticmethod
    def set_basis(value):
        if value.endswith('**'):
            basis = f'{value[:-2]}(D,P)'
        elif value.endswith('*'):
            basis = f'{value[:-1]}(D)'
        else:
            basis = value
        return basis.upper()

    @staticmethod
    def set_dispersion(value):
        if value.lower() in ["no", "false", "n", "f"]:
            disp = False
        else:
            disp = value.upper()
        return disp


def check_if_file_exists(file):
    if not os.path.exists(file) and not os.path.exists(f'{file}_qforce'):
        sys.exit(f'ERROR: "{file}" does not exist.\n')
    return file


def parse_command_line():
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument('f', type=check_if_file_exists, metavar='file',
                        help=('Input coordinate file mol.ext (ext: pdb, xyz, gro, ...)\n'
                              'or directory (mol or mol_qforce) name.'))
    parser.add_argument('-o', type=check_if_file_exists, metavar='options',
                        help='File name for the optional options.')
    args = parser.parse_args()

    return args.f, args.o


def run_qforce(input_arg, config_file=None):
    Initialize.from_questions(input_arg=input_arg, config=config_file, check_only=True)