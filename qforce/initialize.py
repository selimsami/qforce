import shutil
import os
import sys
from ase.io import read, write
from qforce.make_qm_input import make_hessian_input
from qforce.hessian import fit_forcefield


class Initialize():
    """
    Scope:
    ------
    Read the Q-Force input file containing all necessary and optional
    the commands.

    Input file has single line commands such as:
        one_line = value
    and multi-line commands such as:
        [multi_line]
        value1
        value2

    Also checks if all necessary files and software are present.
    """
    def __init__(self, args):
        ########## GENERAL ##########
        self.frag_lib = os.path.expanduser("~/qforce_fragments")
        self.n_equiv = 4
        ########## QM INPUT #########
        self.scan_no = "35"
        self.scan_step = "10.0"
        self.charge = 0
        self.multi = 1
        self.method = "PBEPBE"
        self.basis = "6-31+G(D)"
        self.disp = "GD3BJ"
        self.nproc = ""
        self.mem = ""
        self.charge_method = "CM5"
        self.job_script = []
        ########## FITTING ##########
        self.vibr_coef = 1.0
        self.fchk_file = ""
        self.qm_freq_out = ""
        self.urey = False
        self.cross = False
        self.ignored_terms = ['urey', 'cross_bond_angle']
        self.n_excl = 3
        self.param = []  # temporary
        self.nofrag = args.nofrag
        self.point_charges = 'd4'
        self.lennard_jones = 'd4'
        #############################

        if args.p:
            self.param = args.p

        if args.o:
            self.read_options(args.o)

        self.initialize(args.f, args.s)

    def initialize(self, file, start):
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

        if not self.fchk_file or not self.qm_freq_out:
            self.job_type = 'init'
            self.check_requirements()
            make_hessian_input(self)
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

    def read_options(self, input_file):
        with open(input_file, "r") as inp:
            in_job_script = False
            for line in inp:
                line = line.strip()
                low_line = line.lower().replace("=", " = ")
                if in_job_script:
                    self.job_script.append(line)
                elif low_line == "" or low_line[0] == ";":
                    continue
                elif len(low_line.split()) > 1 and low_line.split()[1] == "=":
                    prop = low_line.split()[0]
                    value = line.replace("=", " = ").split()[2]
                    # General
                    if prop == "n_equiv":
                        self.n_equiv = int(value)
                    # related to file creation
                    elif prop == "scan_no":
                        self.scan_no = str(value)
                    elif prop == "scan_step":
                        self.scan_step = str(value)
                    elif prop == "charge":
                        self.charge = int(value)
                    elif prop == "multiplicity":
                        self.multi = int(value)
                    elif prop == "method":
                        self.method = value.upper()
                    elif prop == "basis_set":
                        self.set_basis(value.upper())
                    elif prop == "dispersion":
                        if value == "no":
                            self.disp = ""
                        else:
                            self.disp = value.upper()
                    elif prop == "n_procs":
                        self.set_nproc(value)
                    elif prop == "memory":
                        self.set_mem(value)
                    elif prop == "charge_method":
                        self.charge_method = value
                    # related to hessianfitting
                    elif prop == "urey" and value == "yes":
                        self.urey = True
                        self.ignored_terms.remove('urey')
                    elif prop == "cross" and value == "yes":
                        self.cross = True
                        self.ignored_terms.remove('cross_bond_angle')
                    elif prop == "vibrational_coef":
                        self.vibr_coef = float(value)
                    elif prop == "n_excl" and value in ["2", "3"]:
                        self.n_excl = int(value)
                    # related to fragment
                    elif prop == "frag_dir":
                        self.frag_lib = value
                    elif prop == "point_charges":
                        self.point_charges = value
                    elif prop == "lennard_jones":
                        self.lennard_jones = value
                    elif prop == "nb_off" and value == 'yes':
                        self.ignored_terms.append('non_bonded')
                elif ("[" in low_line and "]" in low_line and
                      "job_script" in low_line):
                    in_job_script = True

    def set_nproc(self, nprocs):
        if nprocs != "no":
            self.nproc = ("%nprocshared=" + nprocs + "\n")

    def set_mem(self, memory):
        if memory != "no":
            self.mem = ("%mem=" + memory + "\n")

    def set_basis(self, basis):
        if '**' in basis:
            self.basis = f'{basis[:-2]}(D,P)'
        elif '*' in basis:
            self.basis = f'{basis[:-1]}(D)'
        else:
            self.basis = basis

    def check_requirements(self):
        if self.job_type == "fragment" or self.job_type == "fit":
            if not self.fchk_file or not self.qm_freq_out:
                sys.exit(f"ERROR: You requested a {self.job_type} calculation "
                         "but the required .fchk and/or .out/.log file(s) "
                         f"is/are missing in the directory: {self.job_dir} \n"
                         "Please first perform the Hessian calculation and "
                         "provide the necessary output files.\n\n")
        if self.job_type == "fit":
            self.check_exe("dftd4")

    def check_exe(self, exe):
        if exe == "dftd4":
            error = ('To get LJ parameters, you need the DFTD4 '
                     'software installed and the "dftd4" executable in PATH')
            if shutil.which(exe) is None:
                raise FileNotFoundError({error})
