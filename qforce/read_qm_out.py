import numpy as np
import math

class QM():
    """
    Scope:
    ------
    Read a QM input file (fchk or Gaussian output for now)
    In case of Gaussian output - look for different things for different
    job types
    """
    def __init__(self, fchk_file = None, out_files = None, job_type = None):
        #if qm_software == "Gaussian": (in the future)
        self.read_gaussian(fchk_file, out_files, job_type)

    def read_gaussian(self, fchk_file, out_files, job_type):
        if fchk_file != None:
            self.coords = []
            self.atomids = []
            self.hessian = []
            self.dipole = []
            self.rot_tr = []
            self.esp = []
        
            self.read_gaussian_fchk(fchk_file)
            self.prepare()
            
        if out_files != None and job_type != None:
            self.coords = []
            self.atomids = []
            self.nfiles = len(out_files)
            self.energies = []
            self.dipoles = []
            self.angles = []
            self.bond_atoms = []
            self.angle_atoms = []
            self.dihed_atoms = []
            self.bond_r0 = []
            self.angle_theta0 = []
            self.dihed_phi0 = []
            self.init_charges = []
            self.natoms = 0
            self.step = 0
            self.job_type = job_type 
            self.found_atom_ids = False
            self.found_mulliken = False
            self.found_esp = False
            self.files = []
            self.fail_count = 0
            
            for file in out_files:
                self.read_gaussian_out(file)       
            self.numpyfy()     
            if self.job_type == "opt_scan":    
                self.sort_wrt_angles()        
            elif self.job_type == "traj":
                self.check_fail()
     
    def read_gaussian_fchk(self,fchk_file):
        esp_found = False
        with open(fchk_file, "r", encoding='utf-8') as fchk:
            for line in fchk:
                if "Number of atoms  " in line:
                    n_atoms = int(line.split()[4])
                    self.n_atoms = n_atoms
                if "Atomic numbers  " in line:
                    n_line = math.ceil(n_atoms/6)
                    for i in range(n_line):
                        line = fchk.readline()
                        ids = [int(i) for i in line.split()]
                        self.atomids.extend(ids)
                if "Current cartesian coordinates   " in line:
                    n_line = math.ceil(3*n_atoms/5)
                    for i in range(n_line):
                        line = fchk.readline()
                        self.coords.extend(line.split())
                if "RotTr to input orientation   " in line:
                    for i in range(3):
                        line = fchk.readline()
                        self.rot_tr.extend(line.split())
                if "Dipole Moment   " in line:
                        line = fchk.readline()
                        self.dipole.extend(line.split())
                if "Cartesian Force Constants  " in line:
                    n_line = math.ceil(3*n_atoms*(3*n_atoms + 1) / 10)
                    for i in range(n_line):                                         
                        line = fchk.readline()
                        self.hessian.extend(line.split())
                if "ESP Charges  " in line:
                    esp_found = True
                    n_line = math.ceil(n_atoms/5)
                    for i in range(n_line):
                        line = fchk.readline()
                        c = [float(i) for i in line.split()]
                        self.esp.extend(c)
        if not esp_found:
            self.esp = [0 for i in range(n_atoms)]
            print("ESP charges not found in FCHK file. Setting them to zero.")

    def prepare(self):
        bohr2ang = 0.52917721067
        hartree2kjmol = 2625.499638
        
        self.coords = np.asfarray(self.coords, float)
        self.hessian = np.asfarray(self.hessian, float)
        self.dipole = np.asfarray(self.dipole, float)
        self.rot_tr = np.asfarray(self.rot_tr, float)
        self.esp = np.array(self.esp)
        self.coords  = np.reshape(self.coords, (-1,3))

        #convert to input coords
        if self.rot_tr != []:
            rot = np.reshape(self.rot_tr[0:9], (3,3)) 
            tr = np.array(self.rot_tr[9:])
            self.inp_coords = np.dot(self.coords,rot) + tr
        else:
            self.inp_coords = self.coords

        self.coords = self.coords * bohr2ang
        self.hessian = self.hessian * hartree2kjmol / bohr2ang**2

    def read_gaussian_out (self, file):
        
        with open(file, "r", encoding='utf-8') as gaussout:
            file = file.split("/")[-1]
            orientation = "Standard orientation:"
            found_job_specs, normal_term = False, False
            found_energy, found_dipole = False, False
            step = 0
            for line in gaussout:
                line = line.strip()
                
                #read job specs and find job type
                if found_job_specs == False and "--" in line:
                    line = gaussout.readline().strip()
                    if "#" in line:
                        found_job_specs = True
                        job_specs = line
                        line = gaussout.readline().strip()
                        while "--" not in line:
                            job_specs = (job_specs + line)
                            line = gaussout.readline().strip()
                        job_specs = job_specs.lower().replace(" ","")
                        if ("nosymm" in job_specs or "symmetry=none"
                            in job_specs):
                            orientation = "Input orientation:"      
                            
                #find atom names and number of atoms
                elif orientation in line:
                    coord = []
                    for _ in range(5):
                        line = gaussout.readline()
                    while "--" not in line:
                        x, y, z = line.split()[3:6]
                        coord.append([float(x), float(y), float(z)])                                                
                        if self.found_atom_ids == False:
                            self.natoms+= 1
                            self.atomids.append(int(line.split()[1]))
                        line = gaussout.readline()
                    self.found_atom_ids = True
                    
                #read scanned atoms, step size, # of steps
                elif (self.job_type == "opt_scan" and 
                      "The following ModRedundant" in line):
                    line = gaussout.readline()
                    self.scanned_atoms = line.split()[1:5]
                    self.step_no = int(line.split()[6])
                    self.step_size = float(line.split()[7])

                #read initial scan angle
                elif "  Scan  " in line and "!" in line:
                    self.init_angle = float(line.split()[3])  
    
                elif "Charge" in line and "Multiplicity" in line:
                    self.charge = int(line.split()[2])
    
                #read energy
                elif "SCF Done:" in line:
                    energy = round(float(line.split()[4]), 8)
                    found_energy = True
                
                #read dipole moment
                elif "Dipole moment" in line:
                    line = gaussout.readline()
                    mu_x, mu_y, mu_z, mu = line.split()[1:8:2]
                    dipole = [float(mu_x), float(mu_y), float(mu_z), float(mu)]
                    found_dipole = True
                    
#                elif "Quadrupole moment" in line:
#                    line = gaussout.readline()
#                    q_xx, q_yy, q_zz = line.split()[1:6:2]
#                    line = gaussout.readline()
#                    q_xy, q_xz, q_yz = line.split()[1:6:2]
#                    quad = [float(q_xx), float(q_yy), float(q_zz),
#                            float(q_xy), float(q_xz), float(q_yz)]
                
                #Save Mulliken and ESP charges if available
                elif not self.found_mulliken and "Mulliken charges:" in line:
                    self.found_mulliken = True
                    line = gaussout.readline()
                    for i in range(self.natoms):
                        line = gaussout.readline()
                        self.init_charges.append(float(line.split()[2]))
                elif not self.found_esp and "ESP charges:" in line:
                    self.found_esp = True
                    self.init_charges = []
                    line = gaussout.readline()
                    for i in range(self.natoms):
                        line = gaussout.readline()
                        self.init_charges.append(float(line.split()[2]))

                #Save atom no of atoms forming the bond and angle parameters
                if self.job_type == "freq" and "calculate D2E/DX2" in line:
                    param = line.split()[2]
                    atoms = [int(a) - 1 for a in param[2:-1].split(",")]
                    value = float(line.split()[3])
                    if param[0] == "R":
                        self.bond_atoms.append(atoms)
                        self.bond_r0.append(value)
                    elif param[0] == "A":
                        self.angle_atoms.append(atoms)
                        self.angle_theta0.append(value)
                    elif param[0] == "D":
                        self.dihed_atoms.append(atoms)
                        self.dihed_phi0.append(value)

                #check if calculation finished successfully
                elif "Normal termination of Gaussian" in line:
                    normal_term = True
                    
                #On opt_scan mode: save coordinates and energies for each step
                elif self.job_type == "opt_scan" and "-- Stationary" in line:
                    angle = self.init_angle + step * self.step_size
                    angle = round(angle % 360, 4)
                    if (angle in self.angles 
                        and energy < self.energies[self.angles.index(angle)]):
                        self.energies[self.angles.index(angle)] = energy
                        self.coords[self.angles.index(angle)] = coord
                    elif angle not in self.angles:
                        self.angles.append(angle)
                        self.energies.append(energy)
                        self.coords.append(coord)
                    step+=1
        if not found_job_specs or not normal_term:
            print(f"{file} has not finished successfully.")
            self.fail_count+=1
        # for traj job, append all energies, dipoles, times, files
        elif self.job_type == "traj":
            if found_energy and found_dipole:
                self.energies.append(energy)
                self.dipoles.append(dipole)
                self.coords.append(coord)
                self.files.append(file)
            else:
                print(f"{file} has not finished successfully.")
                self.fail_count+=1
        elif self.job_type == "freq":
            self.coords.append(coord)
                
    def numpyfy(self):
        self.angles = np.array(self.angles)
        self.energies = np.array(self.energies)
        self.coords = np.array(self.coords)
        self.dipoles = np.array(self.dipoles)

    def sort_wrt_angles(self):
        order = np.argsort(self.angles)
        self.angles = self.angles[order]
        self.energies = self.energies[order]
        self.coords = self.coords[order]

    def check_fail(self):
        print(f"{len(self.files)} successful calculations")
        if self.fail_count != 0:
            print(f"{self.fail_count} calculations have failed.")
            print("Continuing without them.")
        else:
            print("All calculation(s) have finished successfully.")
        