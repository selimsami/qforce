import os
from types import SimpleNamespace
import pkg_resources
from colt import Colt

from .qm.qm import QM, implemented_qm_software
from .molecule.terms import Terms
from .dihedral_scan import DihedralScan
from qforce.misc import LOGO


class Initialize(Colt):
    _questions = """
[general]
# Debug Mode
debug_mode = False :: bool

[opt]
# Verbose settings for optimizer
verbose = 1 :: int :: [0, 1, 2]

# Noise for initial conditions
noise = 0.0 :: float :: >0

# Optimization type
fit_type = linear :: str :: [linear, non_linear]

# Amount of iterations in linear lsq scipy optimizer
iter = 100 :: int :: >0

# Non-linear optimization method
nonlin_alg = trf :: str :: [trf, lm, compass, bh]

# Averaging before or after fit
average = last :: str :: [before, after, last]

[ff]
# Whether to scan dihedrals
scan_dihedrals = True :: bool

# Whether to compute and write the force field files or not
compute_ff = True :: bool

# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent
n_equiv = 4 :: int

# Number of first n neighbors to exclude in the forcefield
n_excl = 2 :: int :: [2, 3]

# Lennard jones method for the forcefield
lennard_jones = opls_auto :: str :: [gromos_auto, gromos, opls_auto, opls, gaff, ext]

# Use externally provided point charges in the file "ext_q" in the job directyory
ext_charges = no :: bool

# Scale QM charges to account for condensed phase polarization (should be set to 1 for gas phase)
charge_scaling = 1.2 :: float

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

    def __init__(self, config, input_arg):
        self.config = self._set_config(config)
        self.job = self._get_job_info(input_arg)

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


def initialize(input_arg, config, presets=None):
    print(LOGO)
    init = Initialize.from_questions(input_arg=input_arg, config=config, presets=presets,
                                     check_only=True)
    return init.config, init.job
