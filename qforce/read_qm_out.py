import math
import numpy as np
#
from .dftd4 import run_dftd4

class QM():
    """
    Scope:
    ------
    Read a QM input file (fchk or Gaussian output for now)
    In case of Gaussian output - look for different things for different
    job types
    """
    def __init__(self, inp, job_type, fchk_file=False, out_file=False):
        # if qm_software == "Gaussian": (in the future)
        self.read_gaussian(fchk_file, out_file, job_type)

        if job_type == 'freq':
            self.get_nonbonded(inp)

    def read_gaussian(self, fchk_file, out_file, job_type):
        if out_file:
            self.normal_term = False
            self.atomids = []
            self.n_atoms = 0
            self.coords = []
            self.read_gaussian_out(out_file, job_type)

        if fchk_file:
            self.coords = []
            self.atomids = []
            self.hessian = []
            self.dipole = []
            self.rot_tr = []

            self.read_gaussian_fchk(fchk_file)
            self.prepare()

    def get_nonbonded(self, inp):
        if 'd4' in [inp.point_charges, inp.lennard_jones]:
            run_dftd4(inp, self)

        if inp.point_charges == 'ext':
            self.q = np.loadtxt(f'{inp.job_dir}/ext_q')
        elif inp.point_charges == 'cm5':
            self.q = self.cm5

        if inp.lennard_jones == 'ext':
            self.read_external_lennard_jones(inp)

    def read_external_lennard_jones(self, inp):
        unit_type = 'c6/c12'
        all_lj, lj_types, lj_dict = [], [], {}

        with open(f'{inp.job_dir}/ext_lj', 'r') as file:
            for line in file:
                line = line.strip()
                if line == '':
                    continue
                elif line.startswith('@'):
                    unit_type = line.partition('@')[2].strip().lower()
                    print(unit_type)
                elif line.startswith('#'):
                    line = line[1:].split()
                    lj_dict[line[0]] = [float(line[1]), float(line[2])]
                else:
                    lj_types.append(line)

        for lj_type in lj_types:
            all_lj.append(lj_dict[lj_type])

        all_lj = np.array(all_lj)
        if unit_type == 'sigma/epsilon':
            self.c6 = 4 * all_lj[:, 1] * all_lj[:, 0]**6
            self.c12 = 4 * all_lj[:, 1] * all_lj[:, 0]**12
        elif unit_type == 'c6/c12':
            self.c6 = all_lj[:, 0]
            self.c12 = all_lj[:, 1]
        self.c6 = self.c6 * 1e6
        self.c12 = self.c12 * 1e12

    def read_gaussian_fchk(self, fchk_file):
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

    def prepare(self):
        bohr2ang = 0.52917721067
        hartree2kjmol = 2625.499638

        self.coords = np.asfarray(self.coords, float)
        self.hessian = np.asfarray(self.hessian, float)
        self.dipole = np.asfarray(self.dipole, float)
        self.rot_tr = np.asfarray(self.rot_tr, float)
        self.coords = np.reshape(self.coords, (-1, 3))

        # convert to input coords
        if self.rot_tr != []:
            rot = np.reshape(self.rot_tr[0:9], (3, 3))
            tr = np.array(self.rot_tr[9:])
            self.inp_coords = np.dot(self.coords, rot) + tr
        else:
            self.inp_coords = self.coords

        self.coords = self.coords * bohr2ang
        self.hessian = self.hessian * hartree2kjmol / bohr2ang**2

    def read_gaussian_out(self, file, job_type):
        orientation = "Standard orientation:"
        found_job_specs = False

        with open(file, "r", encoding='utf-8') as gaussout:
            for line in gaussout:
                if line.startswith(" NAtoms= "):
                    self.n_atoms = int(line.split()[1])
                elif "Charge" in line and "Multiplicity" in line:
                    self.charge = int(line.split()[2])
#                elif "Dipole moment" in line:
#                    line = gaussout.readline().split()
#                    self.dipoles = [float(d) for d in line[1:8:2]]

                elif job_type == "scan" and "following ModRedundant" in line:
                    self.angles, self.energies = [], []
                    step = 0
                    for line in gaussout:
                        line = line.split()
                        if line == []:
                            break
                        elif line[0] == 'D' and line[5] == 'S':
                            self.scanned_atoms = line[1:5]
                            self.step_no = int(line[6])
                            self.step_size = float(line[7])

                elif "  Scan  " in line and "!" in line:
                    self.init_angle = float(line.split()[3])

                # read job specs and find job type
                elif "--" in line and not found_job_specs:
                    line = gaussout.readline().strip()
                    if "#" in line:
                        found_job_specs = True
                        job_specs = line
                        line = gaussout.readline().strip()
                        while "--" not in line:
                            job_specs = (job_specs + line)
                            line = gaussout.readline().strip()
                        job_specs = job_specs.lower().replace(" ", "")
                        if ("nosymm" in job_specs or "symmetry=none"
                                in job_specs):
                            orientation = "Input orientation:"

                # Find coordinates
                elif orientation in line:
                    coord = []
                    for _ in range(5):
                        line = gaussout.readline()
                    while "--" not in line:
                        coord.append([float(a) for a in line.split()[3:6]])
                        line = gaussout.readline()

                elif job_type == "scan" and "SCF Done:" in line:
                    energy = round(float(line.split()[4]), 8)

                # Get optimized energies, coords for each scan angle
                elif "-- Stationary" in line and job_type == "scan":
                    angle = self.init_angle + step * self.step_size
                    self.angles.append(round(angle % 360, 4))
                    self.energies.append(energy)
                    self.coords.append(coord)
                    step += 1

                elif "Normal termination of Gaussian" in line:
                    self.normal_term = True

                elif "N A T U R A L   B O N D   O R B I T A L" in line:
                    found_wiberg = False
                    self.lone_e = np.zeros(self.n_atoms, dtype='int8')
                    self.n_bonds = []
                    self.b_orders = [[] for _ in range(self.n_atoms)]
                    while "           Charge unit" not in line:
                        line = gaussout.readline()
                        if ("bond index matrix" in line and not found_wiberg):
                            for _ in range(int(np.ceil(self.n_atoms/9))):
                                for atom in range(-3, self.n_atoms):
                                    line = gaussout.readline().split()
                                    if atom >= 0:
                                        order = [float(l) for l in line[2:]]
                                        self.b_orders[atom].extend(order)
                        if ("bond index, Totals" in line and not found_wiberg):
                            found_wiberg = True
                            for i in range(-3, self.n_atoms):
                                line = gaussout.readline()
                                if i >= 0:
                                    n_bonds = round(float(line.split()[2]), 0)
                                    self.n_bonds.append(int(n_bonds))
                        if "Natural Bond Orbitals (Summary)" in line:
                            while "Total Lewis" not in line:
                                line = gaussout.readline()
                                if " LP " in line:
                                    atom = int(line.split()[5])
                                    occ = int(round(float(line.split()[6]), 0))
                                    if occ > 0:
                                        self.lone_e[atom-1] += occ
                elif "Hirshfeld charges, spin densities" in line:
                    self.cm5 = []
                    line = gaussout.readline()
                    for i in range(self.n_atoms):
                        line = gaussout.readline().split()
                        self.cm5.append(float(line[7]))
                    self.cm5 = np.array(self.cm5)

        if job_type == 'freq':
            self.b_orders = np.round(np.array(self.b_orders)*2)/2
        if job_type == 'scan':
            hartree2kjmol = 2625.499638
            order = np.argsort(self.angles)
            self.angles = np.array(self.angles)[order]
            self.coords = np.array(self.coords)[order]
            self.energies = np.array(self.energies)[order] * hartree2kjmol
            self.energies -= self.energies.min()

