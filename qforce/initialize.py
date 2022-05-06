import os
import shutil
from io import StringIO
from types import SimpleNamespace
import pkg_resources
#
from colt import Colt
#
from .qm.qm import QM, implemented_qm_software
from .molecule.terms import Terms
from .dihedral_scan import DihedralScan
from .misc import LOGO


class Initialize(Colt):
    _user_input = """
[ff]
# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent
n_equiv = 4 :: int

# Number of first n neighbors to exclude in the forcefield
n_excl = 2 :: int :: [2, 3]

# Lennard jones method for the forcefield
lennard_jones = opls_auto :: str :: [gromos_auto, gromos, opls_auto, opls, gaff, gaff2, charmm36, ext]

# Use externally provided point charges in the file "ext_q" in the job directyory
ext_charges = no :: bool

# Scale QM charges to account for condensed phase polarization (should be set to 1 for gas phase)
charge_scaling = 1.2 :: float

# If user chooses ext_charges=True, by default fragments will still use the chosen QM method for
# determining fragment charges. This is to avoid spuriously high charges on capping hydrogens.
# However, using QM charges for fragments and ext_charges for the molecule can also result in
# inaccuracies if these two charges are very different.
use_ext_charges_for_frags = no :: bool

# Additional exclusions (GROMACS format)
exclusions = :: literal

# Switch standard non-bonded interactions between two atoms to pair interactions
# (provide atom pairs in each row)
pairs = :: literal

# Path for the external FF library for Lennard-Jones parameters (GROMACS format).
ext_lj_lib = :: folder, optional

# Lennard-Jones fudge parameter for 1-4 interactions for when "lennard_jones" is set to "ext".
ext_lj_fudge = :: float, optional

# Coulomb fudge parameter for 1-4 interactions for when "lennard_jones" is set to "ext".
ext_q_fudge =  :: float, optional

# Lennard-Jones combinations rules for when "lennard_jones" is set to "ext" (GROMACS numbering).
ext_comb_rule =  :: int, optional :: [1, 2, 3]

# Name of the atom type for capping hydrogens for when "lennard_jones" is set to "ext"
ext_h_cap = :: str, optional

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

    @staticmethod
    def _set_config(config):
        config['qm'].update(config['qm']['software'])
        config['qm'].update({'software': config['qm']['software'].value})
        config.update({key: SimpleNamespace(**val) for key, val in config.items()})
        return SimpleNamespace(**config)

    @classmethod
    def _extend_user_input(cls, questions):
        questions.generate_block("qm", QM.colt_user_input)
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
