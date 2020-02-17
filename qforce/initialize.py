import shutil
import os
import sys
from ase.io import read, write
from qforce.make_qm_input import make_hessian_input
from qforce.hessian import fit_forcefield
from colt.colt import Colt


class Initialize(Colt):
    _questions = """
    # Directory where the fragments are saved
    frag_lib = ~/qforce_fragments :: str

    # Number of n equivalent neighbors needed to consider two atoms equivalent
    # Negative values turns off equivalence, 0 makes same elements equivalent
    n_equiv = 4 :: int

    # Number of first n neighbors to exclude in the forcefield
    n_excl = 3 :: int :: [2, 3]

    # Point charges used in the forcefield
    point_charges = d4 :: str :: [d4, cm5, esp, ext]

    # Lennard jones method for the forcefield
    lennard_jones = d4 :: str :: [d4, gromos, opls]

    # Combination rule for the Lennard-Jones interactions - see GROMACS manual
    # Default is force field dependent (d4: 2, opls: 3, gromos: 1)
    comb_rule = 0 :: int :: [1, 2, 3]

    # Scaling of the vibrational frequencies (not implemented)
    vibr_coef = 1.0 :: float

    # Use Urey-Bradley angle terms
    urey = yes :: bool

    # Ignore non-bonded interactions during fitting
    non_bonded = yes :: bool

    # Make fragments and calculate flexible dihedrals
    fragment = yes :: bool

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
    def from_config(cls, answers):
        ignored_terms = []

        for key, value in answers.items():
            if key == 'frag_lib':
                answers[key] = os.path.expanduser(value)
            elif key == 'basis':
                answers[key] = Initialize.set_basis(value)
            elif key == 'dispersion':
                answers[key] = Initialize.set_dispersion(value)
            elif key == 'method':
                answers[key] = value.upper()
            elif key == 'job_script' and value:
                answers[key] = value.replace('\n\n', '\n').split('\n')
            elif key == 'exclusions':
                answers[key] = Initialize.set_exclusions(value)
            elif key in ['non_bonded', 'urey', 'cross_bond_angle'] and not value:
                ignored_terms.append(key)
            elif key in ['fragment'] and not value:
                ignored_terms.append('dihedral/flexible')

        answers['ignored_terms'] = ignored_terms
        answers['comb_rule'] = Initialize.set_comb_rule(answers['comb_rule'],
                                                        answers['lennard_jones'])
        return cls(answers)

    def setup(self, file, param):

        if param:  # temporary for fitting
            self.param = param
        else:
            self.param = []

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
    def set_comb_rule(comb_rule, lj_type):
        if comb_rule == 0:
            if lj_type == 'd4':
                comb_rule = 2
            elif lj_type == 'gromos':
                comb_rule = 1
            elif lj_type == 'opls':
                comb_rule = 3
        return comb_rule

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

    def check_exe(self, exe):
        if exe == "dftd4":
            error = ('To get LJ parameters, you need the DFTD4 '
                     'software installed and the "dftd4" executable in PATH')
            if shutil.which(exe) is None:
                raise FileNotFoundError({error})
