import shutil
import os
import sys
from ase.io import read, write
from qforce.make_qm_input import make_hessian_input
from qforce.fit import fit_forcefield
from colt.colt import Colt
from .polarize import polarize


class Initialize(Colt):
    _questions = """
    # Directory where the fragments are saved
    frag_lib = ~/qforce_fragments :: folder

    # Number of n equivalent neighbors needed to consider two atoms equivalent
    # Negative values turns off equivalence, 0 makes same elements equivalent
    n_equiv = 4 :: int

    # Number of first n neighbors to exclude in the forcefield
    n_excl = 3 :: int :: [2, 3]

    # Point charges used in the forcefield
    point_charges = cm5 :: str :: [cm5, ext, d4]

    # Lennard jones method for the forcefield
    lennard_jones = gromos_auto :: str :: [gromos_auto, gromos, opls, gaff, d4]

    # Scaling of the vibrational frequencies (not implemented)
    vibr_coef = 1.0 :: float

    # Use Urey-Bradley angle terms
    urey = yes :: bool

    # Ignore non-bonded interactions during fitting
    non_bonded = yes :: bool

    # Make fragments and calculate flexible dihedrals
    # Use 'available' option to skip dihedrals with missing scan data
    fragment = yes :: str :: [yes, no, available]

    # Number of neighbors after bonds can be fragmented (0 or smaller means no fragmentation)
    frag_n_neighbor = 3 :: int

    # Set all dihedrals as rigid
    all_rigid = no :: bool

    # Method for doing the MM relaxed dihedral scan
    scan_method = qforce :: str :: [qforce, gromacs]

    # Number of iterations of dihedral fitting
    n_dihed_scans = 3 :: int

    # Symmetrize either md or qm profile of a specific dihedral by inputting the range
    # For symmetrizing the MD dihedral profile between atoms 77 and 80 where 0-180 is inversely
    # equivalent to 180-360 (reverse the order of the second):
    # md 77 80 = 0 180 - 360 180
    symmetrize_scan = :: literal

    # Make a polarizable FF
    polar = no :: bool

    # Polarize a coordinate file and quit (requires itp_file)
    polarize = no :: bool

    # Scale the C6 dispersion interactions in the polarizable version of the FF
    polar_c6_scale = 0.8 :: float

    # Specifically not scale some of the atoms
    polar_not_scale_c6 = :: literal

    # Manual polarizabilities in the file ext_alpha
    ext_alpha = no :: bool

    # Name of itp file (only needed for polarize option)
    itp_file = itp_file_missing :: str

    # Use Bond-Angle cross term
    cross_bond_angle = no :: bool

    job_script = :: literal
    exclusions = :: literal

    # Number of dihedral scan steps to perform
    scan_no = 23 :: int

    # Step size for the dihedral scan
    scan_step = 15.0 :: float

    # Total charge of the system
    charge = 0 :: int

    # Multiplicity of the system
    multi = 1 :: int

    # QM method to be used
    method = PBEPBE :: str

    # QM basis set to be used
    basis = 6-31+G(D) :: str

    # Dispersion correction to include
    disp = GD3BJ :: str

    # Number of processors for the QM calculation
    nproc = 1 :: int

    # Memory setting for the QM calculation
    mem = 4GB :: str

    """

    def __init__(self, answers):
        for key, value in answers.items():
            setattr(self, key, value)

    @classmethod
    def from_config(cls, config):
        ignored_terms = []
        answers = {}

        for key, value in config.items():
            # if key == 'frag_lib':
            #     answers[key] = os.path.expanduser(value)
            if key == 'basis':
                answers[key] = Initialize.set_basis(value)
            elif key == 'dispersion':
                answers[key] = Initialize.set_dispersion(value)
            elif key == 'method':
                answers[key] = value.upper()
            elif key == 'job_script' and value:
                answers[key] = value.replace('\n\n', '\n').split('\n')
            elif key == 'polar_not_scale_c6':
                answers[key] = Initialize.set_polar_not_scale_c6(value)
            elif key == 'exclusions':
                answers[key] = Initialize.set_exclusions(value)
            elif key in ['non_bonded', 'urey', 'cross_bond_angle'] and not value:
                ignored_terms.append(key)
            elif key in ['fragment']:
                answers[key], ignored_terms = Initialize.set_fragments(value, ignored_terms)

        answers['ignored_terms'] = ignored_terms
        comb, fudge_lj, fudge_q = Initialize.set_non_bonded_props(config['lennard_jones'])
        answers['comb_rule'] = comb
        answers['fudge_lj'] = fudge_lj
        answers['fudge_q'] = fudge_q
        config.update(answers)
        return cls(config)

    def setup(self, file):
        coord = False

        if file.endswith('/'):
            file = file[:-1]
        path = os.path.dirname(file)
        if path != '':
            path = f'{path}/'
            file = os.path.basename(file)

        if '.' in file:
            coord = True
            self.job_name = file.split('.')[0]
            self.coord_file = file
        else:
            self.job_name = file.split('_qforce')[0]

        if self.polarize:
            polarize(self, path)
            sys.exit()

        self.job_dir = f'{path}{self.job_name}_qforce'
        os.makedirs(self.job_dir, exist_ok=True)
        self.frag_dir = f'{self.job_dir}/fragments'
        os.makedirs(self.frag_dir, exist_ok=True)
        os.makedirs(self.frag_lib, exist_ok=True)

        self.fchk_file = self.set_file_name('.fchk')
        self.qm_freq_out = self.set_file_name(('.out', '.log'))

        if coord:
            self.xyz_file = f'{self.job_dir}/init.xyz'
            molecule = read(file)
            write(self.xyz_file, molecule, plain=True)

        if not self.fchk_file and not self.qm_freq_out:
            self.job_type = 'init'
            self.check_requirements()
            make_hessian_input(self)
        elif not self.fchk_file or not self.qm_freq_out:
            missing = ['Checkpoint', 'Output'][[self.fchk_file, self.qm_freq_out].index(False)]
            print('Please provide both the checkpoint and the output file of the Hessian',
                  f'calculation.\n{missing} file is missing.')
        else:
            self.job_type = 'fit'
            self.check_requirements()
            fit_forcefield(self)

    def set_file_name(self, ext):
        files = [f for f in os.listdir(self.job_dir) if f.endswith(ext)]
        f_type = '/'.join(ext)
        if len(files) == 0:
            file = False
        elif len(files) > 2:
            sys.exit(f'ERROR: There are multiple {f_type} files in'
                     f'{self.job_dir}. Please check. \n\n')
        else:
            file = f'{self.job_dir}/{files[0]}'
        return file

    @staticmethod
    def set_basis(value):
        if '**' in value:
            basis = f'{value[:-2]}(D,P)'
        elif '*' in value:
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

    @staticmethod
    def set_non_bonded_props(lj_type):
        if lj_type == 'd4':
            comb_rule = 2
            fudge_lj, fudge_q = 1.0, 1.0
        elif lj_type in ['gromos', 'gromos_auto']:
            comb_rule = 1
            fudge_lj, fudge_q = 1.0, 1.0
        elif lj_type == 'opls':
            comb_rule = 3
            fudge_lj, fudge_q = 0.5, 0.5
        elif lj_type == 'gaff':
            comb_rule = 2
            fudge_lj, fudge_q = 0.5, 0.8333
        return comb_rule, fudge_lj, fudge_q

    @staticmethod
    def set_fragments(value, ignored_terms):
        if value.lower() == 'no':
            value = False
            ignored_terms.extend(['dihedral/flexible'])
        return value, ignored_terms

    @staticmethod
    def set_polar_not_scale_c6(value):
        if value:
            not_scale = value.split()
        else:
            not_scale = []
        return not_scale

    @staticmethod
    def set_symmetrize_scan():
        ...

    @staticmethod
    def set_exclusions(value):
        exclusions = []
        if value:
            for line in value.split('\n'):
                line = [int(i) for i in line.strip().partition('#')[0].split()]
                if len(line) > 1:
                    a1, a2s = line[0], line[1:]
                    for a2 in a2s:
                        pair = tuple(sorted([a1-1, a2-1]))
                        if pair not in exclusions:
                            exclusions.append(pair)
                elif len(line) == 1:
                    print('WARNING: Exclusion lines should contain at least two atom IDs:\n',
                          'First entry is excluded from all the following entries.\n',
                          f'Ignoring the line: {line[0]}\n')
        return exclusions

    def check_requirements(self):
        if self.job_type == "fragment" or self.job_type == "fit":
            if not self.fchk_file or not self.qm_freq_out:
                sys.exit(f"ERROR: You requested a {self.job_type} calculation "
                         "but the required .fchk and/or .out/.log file(s) "
                         f"is/are missing in the directory: {self.job_dir} \n"
                         "Please first perform the Hessian calculation and "
                         "provide the necessary output files.\n\n")
        if self.job_type == "fit" and 'd4' in [self.point_charges, self.lennard_jones]:
            self.check_exe("dftd4")
        lj_dir = f'{self.job_dir}/ext_lj'
        if (self.job_type == "fit" and self.lennard_jones not in ['d4', 'gromos_auto']
                and not os.path.isfile(lj_dir)):
            sys.exit('ERROR: You switched away from the automatic atom type determination with'
                     f'GROMOS but not provided external atom types for "{self.lennard_jones}"\n.'
                     'Please provide atom types for Lennard-Jones interactions in the file '
                     '"ext_lj".\n\n')

    def check_exe(self, exe):
        if exe == "dftd4":
            error = ('To get LJ parameters, you need the DFTD4 '
                     'software installed and the "dftd4" executable in PATH')
            if shutil.which(exe) is None:
                raise FileNotFoundError({error})
