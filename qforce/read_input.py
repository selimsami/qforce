import shutil
import os
from . import qforce_data

class Input():
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
    def __init__(self, job_type, input_file):
        self.job_type = job_type
        self.relevant_files = []
        #related input creation
        self.coord_file = ""
        self.scan_no = "35"
        self.scan_step = "10.0"
        self.charge = "0"
        self.multi = "1"
        self.method = "wB97XD"
        self.basis = "6-31G(d,p)"
        self.disp = ""
        self.nproc = ""
        self.mem = ""
        self.charge_method = "CM5"
        self.dihedrals = [["d1", "d2", "d3", "d4"]]
        self.pre_input_commands = []
        self.post_input_commands = []
        #related to dihedral scan
        self.gmx = "gmx"
        self.itp_file = ""
        self.top_file = ""
        self.mdp_file = ""
        self.fitting_function = "bellemans"
        self.qm_scan_out = []
        self.extra_files = []
        self.polar_scan = False
        #related to qm vs md
#        self.energy_xvg = "energy.xvg"
#        self.mtot_xvg = "Mtot.xvg"
        #related to seminario method
        self.vibr_coef = 1.0
        self.fchk_file = ""
        self.qm_freq_out = ""
        #related to dipolefitting
        self.traj_dir = ""
        self.fit_percent = -1
        self.n_equiv = 4
        # related to hessianfitting
        self.urey = False
        
        self.read_input(input_file)
        self.check_compulsory_settings()
        self.check_if_files_exist(self.relevant_files)

        
    def read_input(self,input_file):
        with open(input_file, "r") as inp:
            in_section = None
            for line in inp:
                line = line.strip()
                low_line = line.lower().replace("="," = ")
                if low_line is "" or low_line[0] == ";":
                    continue
                elif len(low_line.split()) > 1 and low_line.split()[1] == "=":
                    prop = low_line.split()[0]
                    value = line.replace("="," = ").split()[2]
                    in_section = None
                    #related to file creation
                    if prop == "coord_file":
                        self.coord_file = value
                    elif prop == "scan_no":
                        self.scan_no = str(value) 
                    elif prop == "scan_step":
                        self.scan_step = str(value)
                    elif prop == "charge":
                        self.charge = value
                    elif prop == "multiplicity":
                        self.multi = value
                    elif prop == "method":
                        self.method = value
                    elif prop == "basis_set":
                        self.basis = value
                    elif prop == "dispersion":
                        self.set_disp(value)
                    elif prop == "n_procs":
                        self.set_nproc(value)
                    elif prop == "memory":
                        self.set_mem(value)
                    elif prop == "charge_method":
                        self.charge_method = value
                    #related to dihedral scanning
                    elif prop == "fitting_function":
                        self.fitting_function = value    
                    elif prop == "itp_file":
                        self.itp_file = value
                    elif prop == "top_file":
                        self.top_file = value
                    elif prop == "mdp_file":
                        self.mdp_file = value
                    #related to seminario method
                    elif prop == "fchk_file":
                        self.fchk_file = value
                    elif prop == "qm_freq_out":
                        self.qm_freq_out = value
                    elif prop == "vibrational_coef":
                        self.vibr_coef = float(value)   
                    elif prop == "polar_scan":
                        if value == "yes":
                            self.polar_scan = True
                    #related to qm vs md
#                    elif prop == "energy_xvg":
#                        self.energy_xvg = value
#                    elif prop == "mtot_xvg":
#                        self.mtot_xvg = value
                    #related to dipolefitting
                    elif prop == "traj_dir":
                        self.traj_dir = value
                    elif prop == "n_equivalence":
                        self.n_equiv = int(value)
                    elif prop == "fit_percent":
                        self.fit_percent = float(value)/100
                    #related to hessianfitting
                    elif prop == "urey" and value == "yes":
                        self.urey = True
                        
                elif "[" in low_line and "]" in low_line:
                    no_space = low_line.replace(" ","")
                    open_bra = no_space.index("[") + 1
                    close_bra = no_space.index("]")
                    in_section = no_space[open_bra:close_bra]
                #related to file creation
                elif in_section == "pre_submit_commands":
                    self.pre_input_commands.append(line)
                elif in_section == "post_submit_commands":
                    self.post_input_commands.append(line)
                #related to dihedral scanning
                elif in_section == "qm_scan_out":
                    self.qm_scan_out.append(line.split())
                elif in_section == "extra_files":
                    self.extra_files.append(line.split()[0])
                elif in_section == "dihedrals_scanned":
                    self.add_dihedral(line.split())

    def add_dihedral(self, dihedral):
        if self.dihedrals == [["d1", "d2", "d3", "d4"]]:
            self.dihedrals = []
        dihedral = list(map(str, dihedral))
        self.dihedrals.append(dihedral)

    def set_disp(self,dispersion):
        if dispersion == "no":
            self.disp = ""
        else:
            self.disp = ("EmpiricalDispersion=" + dispersion)
    def set_nproc(self,nprocs):
        if nprocs != "no":
            self.nproc = ("%nprocshared=" + nprocs + "\n")
            
    def set_mem(self,memory):
        if memory != "no":
            self.mem = ("%mem=" + memory + "\n")

    def set_mdp(self):
        if self.mdp_file == "":
            shutil.copy2(f"{qforce_data}/default.mdp", "default.mdp")
            self.mdp_file = "default.mdp"

    def flatten(self, x):
        if type(x) is list:
            return [a for i in x for a in self.flatten(i)]
        else:
            return [x]
        
    def check_if_files_exist(self, files):
        not_exist = "{} file does not exist"
        files = self.flatten(files)
        for file in files:
            file_exists = os.path.isfile(file)
            dir_exists = os.path.isdir(file)
            if file != "" and not (file_exists or dir_exists):
                raise FileNotFoundError({not_exist.format(file)})

    def check_compulsory_settings(self):
        miss = "Missing the name of the {} with the {} option {}"
    
        if "input_" in self.job_type:
            self.relevant_files.append(self.coord_file) 
            self.check_exe("obabel")
            if self.coord_file == "":
                raise NameError({miss.format("coordinate", "one-line",
                                                 '"coord_file = "')})
        if self.job_type == "dihedralfitting":
            self.set_mdp()
            self.relevant_files.append([self.itp_file, self.top_file, 
                                       self.mdp_file, self.qm_scan_out, 
                                       self.extra_files]) 
            self.check_exe("gmx")
            if self.qm_scan_out == []:
                raise NameError({miss.format("QM scan output file",
                                             "multi-line", "[qm_scan_out]")}) 
            if self.itp_file == "":
                raise NameError({miss.format("GROMACS .itp file", "one-line",
                                                '"itp_file = "')})
            if self.top_file == "":
                raise NameError({miss.format("GROMACS .top file", "one-line",
                                                '"top_file = "')})
        if self.job_type == "bondangle":
            self.relevant_files.append([self.fchk_file, self.qm_freq_out])
            if self.fchk_file == "":
                raise NameError({miss.format("QM fchk file",
                                             "one-line", '"fchk_file = "')})
            if self.qm_freq_out == "":
                raise NameError({miss.format("QM frequency calc. output file",
                                             "one-line", '"qm_freq_out"')})
#        if self.job_type == "qmvsmd":
#            self.relevant_files.append([self.energy_xvg, self.mtot_xvg,
#                                        self.traj_dir])
        if self.job_type == "dipolefitting":
            if self.traj_dir == "" and self.coord_file != "":
                self.traj_dir = "{}_traj".format(self.coord_file.split(".")[0])
            elif self.traj_dir == "" and self.coord_file == "":
                raise NameError({miss
                                 .format("either trajectory directory or "
                                         "coordinate file", "one-line", 
                                         '"traj_dir = " or "coord_file = "')})
            self.relevant_files.append([self.traj_dir])
        if self.job_type == "polarize":
            self.relevant_files.append([self.coord_file, self.itp_file])
            if self.coord_file == "":
                raise NameError({miss.format("coordinate", "one-line",
                                                 '"coord_file = "')})
            if self.itp_file == "":
                raise NameError({miss.format("GROMACS .itp file", "one-line",
                                                '"itp_file = "')})
    
    def check_exe(self, exe):
        
        if exe == "obabel":
            error = ('To create QM input files, you need the OpenBabel '
                     'software installed and the "obabel" executable in PATH')
            if shutil.which(exe) is None:
                raise FileNotFoundError({error})
        elif exe == "gmx":
            error = ('To do a dihedral scan, you need to have the GROMACS '
                     'software installed with necessary executables in PATH')
            for gmx in ["gmx", "gmx_mpi", "gmx_d"]:
                if shutil.which(gmx) is not None:
                    self.gmx = gmx
                    break
            else:
                raise FileNotFoundError({error})

