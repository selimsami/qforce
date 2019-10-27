import subprocess, os, shutil
import numpy as np
import matplotlib.pyplot as plt
from symfit import Fit, cos, pi, parameters, variables
import seaborn as sns
from .old_read_qm_out import QM

def scan_each_dihedral(inp):
    """
    For each different dihedral in the input file, do dihedral optimization:
    Read scan QM output(s), run GROMACS with the same angles, get the fitting
    data, fit into a choice function of choice, run GROMACS again with fitted
    dihedrals, print the dihedral profile.
    """

    itp = Itp_file(inp.itp_file)
    shutil.copy2(inp.itp_file, itp.opt)

    for scan in inp.qm_scan_out:

        #create a directory for each scan
        scan_name = scan[0].split(".")[0]
        make_scan_dir(scan_name)

        #get QM scan output info and initiate itp
        qm = QM(out_files = scan, job_type = "opt_scan")
        md_energies, opt_md_energies = [], []

        #print job info
        print(f"Scan name: {scan_name}")
        print("-"*(11+len(scan_name)))
        print("Scanned atoms: {} {} {} {}".format(*qm.scanned_atoms))

        # Create the optimized ITP file and get atom/molecule info
        in_dihedrals, itp = find_scanned_dihedral(qm, itp)

        # Prepare the files for a GROMACS run in each scan step directory
        prepare_scan_directories(qm, inp, itp, scan_name)

        # Run gromacs without the scanned dihedral - get energies
        print("Running GROMACS without the scanned dihedral...")
        for step, angle in enumerate(qm.angles, start=1):
            step_dir = f"{scan_name}/step{step}"
            run_gromacs(step_dir, "nodihed", inp)
            md_energy = read_gromacs_energies(step_dir, "nodihed")
            md_energies.append(md_energy)

        # Set minimum energies to zero and compute QM vs MD difference
        # and the dihedral potential to be fitted
        qm.energies = set_minimum_to_zero(qm.energies) * 2625.499638
        md_energies = set_minimum_to_zero(md_energies)
        dihedral_fitting = set_minimum_to_zero(qm.energies - md_energies)

        #fit the data
        print("Fitting the dihedral function...")
        c, r_squared, fitted_dihedral = do_fitting(qm, inp, dihedral_fitting)

        #print optmized dihedrals
        write_opt_dihedral(itp.temp, in_dihedrals, inp, qm, c, r_squared)
        in_dihedrals = True

        #run gromacs again with optimized dihedrals
        print("Running GROMACS with the fitted dihedral...")
        for step, angle in enumerate(qm.angles, start=1):
            step_dir = f"{scan_name}/step{step}"
            itp_loc = f"{step_dir}/{inp.itp_file}"
            write_opt_dihedral(itp_loc, False, inp, qm, c, r_squared)
            run_gromacs(step_dir, "opt", inp)
            md_energy = read_gromacs_energies(step_dir, "opt")
            opt_md_energies.append(md_energy)

        #set minimum energy to zero
        opt_md_energies = set_minimum_to_zero(opt_md_energies)

        #Plot optimized dihedral profile vs QM profile
        plot_dihedral_profile(qm, opt_md_energies, scan_name)

        os.remove(itp.opt)
        shutil.move(itp.temp, itp.opt)
        print(f"QM vs MD dihedral scan profile: {scan_name}.pdf\n")

    print(f"Done! Optimized dihedrals can be found in: {itp.opt}")


class Itp_file():
    def __init__(self, itp_file):
        self.opt = f"opt_{itp_file}"
        self.temp = f"temp_{itp_file}"

def make_scan_dir(scan_name):
    if  os.path.exists(scan_name):
        shutil.rmtree(scan_name)
    os.makedirs(scan_name)

def find_scanned_dihedral(qm, itp):
    """
    Read the itp file, create an initial optimized itp file minus the scanned
    dihedrals. Get molecule residue info and atom names from the ITP file
    """
    atom2, atom3 = qm.scanned_atoms[1:3]
    in_section = ""
    itp.res_ids, itp.res_names, itp.atom_names = [], [], []

    with open(itp.temp,"w") as temp_itp, open(itp.opt, "r") as opt_itp:
            for line in opt_itp:
                low_line = line.strip().lower()
                if low_line == "" or low_line[0] == ";":
                    temp_itp.write(line)
                elif "[" in low_line and "]" in low_line:
                    no_space = low_line.replace(" ","")
                    open_bra = no_space.index("[") + 1
                    close_bra = no_space.index("]")
                    in_section = no_space[open_bra:close_bra]
                    temp_itp.write(line)
                elif in_section == "atoms":
                    itp.res_ids.append(line.split()[2])
                    itp.res_names.append(line.split()[3])
                    itp.atom_names.append(line.split()[4])
                    temp_itp.write(line)
                elif in_section == "dihedrals":
                    a2, a3 = line.split()[1:3]
                    if [a2, a3] == [atom2, atom3] or [a2, a3] == [atom3,atom2]:
                        continue
                    else:
                        temp_itp.write(line)
                else:
                    temp_itp.write(line)
    in_dihedrals = in_section in "dihedrals"
    return in_dihedrals, itp

def set_minimum_to_zero(energies):
    energies = energies - np.amin(energies)
    return energies

def prepare_scan_directories(qm, inp, itp, scan_name):
    """
    Create the .gro file for each scan step from QM optimized geometries
    Add dihedral restrain to each scan step ITP
    Copy all necessary files for a GROMACS run to scan directories
    """
    gro_title = "{} - Scan #{} - Scanned angle: {}\n"
    gro_coor = "{:>5}{:5}{:>5}{:>5}{:>8.3f}{:>8.3f}{:>8.3f}\n"
    dihed_res = "{:>5}{:>5}{:>5}{:>5}{:>5}{:>10.4f}  0.0  50000\n"

    for step, angle in enumerate(qm.angles, start=1):

        step_dir = f"{scan_name}/step{step}"
        os.makedirs(step_dir)

        #create the .gro file for each scan step
        with open(step_dir + "/start.gro","w") as grofile:
            grofile.write(gro_title.format(scan_name, step, angle))

            if inp.polar_scan:
                grofile.write(f"{qm.natoms*2}\n")
            else:
                grofile.write(f"{qm.natoms}\n")

            for n in range(qm.natoms):
                grofile.write(gro_coor.format(itp.res_ids[n], itp.res_names[n],
                                              itp.atom_names[n], n + 1,
                                              *qm.coords[step-1,n,:]/10))
            if inp.polar_scan:
                for n in range(qm.natoms):
                    grofile.write(gro_coor.format(2, "MOL", "D{}".format(n+1),
                                                  qm.natoms + n + 1,
                                                  *qm.coords[step-1,n,:]/10))
            grofile.write("  20.00000  20.00000  20.00000")

        #copy the .itp file to each scan directory & add the dihedral restrain
        shutil.copy2(itp.temp, step_dir + "/" + inp.itp_file)
        with open(step_dir + "/" + inp.itp_file, "a") as itp_step:
            itp_step.write("\n")
            itp_step.write("[ dihedral_restraints ]\n")
            itp_step.write(";  ai   aj   ak   al type       phi   dp   kfac\n")
            itp_step.write(dihed_res.format(*qm.scanned_atoms, "1", angle))

        #copy the .top file and the extra .itp files (if they exist)
        shutil.copy2(inp.top_file,(step_dir + "/" + inp.top_file))
        shutil.copy2(inp.mdp_file,(step_dir + "/" + inp.mdp_file))
        for extra_file in inp.extra_files:
            shutil.copy2(extra_file,(step_dir + "/" + extra_file))

def run_gromacs (directory,em_type, inp):
    grompp = subprocess.Popen([inp.gmx, 'grompp', '-f', inp.mdp_file, '-p',
            'system.top', '-c', 'start.gro', '-o', ('em_' + em_type + '.tpr'),
            '-po', ('em_' + em_type + '.mdp'), '-maxwarn', '10'],
            cwd = directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    grompp.wait()
    check_gromacs_termination(grompp)
    mdrun = subprocess.Popen([inp.gmx, 'mdrun', '-deffnm', ('em_' + em_type)],
            cwd = directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    mdrun.wait()
    check_gromacs_termination(mdrun)

def check_gromacs_termination(process):
    if process.returncode != 0:
        print(process.communicate()[0].decode("utf-8") )
        raise RuntimeError({"GROMACS run has terminated unsuccessfully"})

def read_gromacs_energies(directory, em_type):
    log_dir = "{}/em_{}.log".format(directory, em_type)
    with open(log_dir, "r", encoding='utf-8') as em_log:
        for line in em_log:
            if "Potential Energy  =" in line:
                md_energy = float(line.split()[3])
    return md_energy

def do_fitting(qm, inp, dihedral_fitting):
    angle_rad, y = variables('angle_rad, y')
    if inp.fitting_function == "bellemans":
        c0, c1, c2, c3, c4, c5 = parameters('c0, c1, c2, c3, c4, c5')
        model = {y: c0 + c1 * cos(angle_rad - pi) + c2 * cos(angle_rad - pi)**2
                    + c3 * cos(angle_rad - pi)**3 + c4 * cos(angle_rad - pi)**4
                    + c5 * cos(angle_rad - pi)**5}

    elif inp.fitting_function == "periodic":

        k3, = parameters (
                    'k3')

        k3.min = 0

        model = {y:
                    k3 * (1 + cos(3 * angle_rad ))
                    }

    elif inp.fitting_function == "improper":
        k, f = parameters ('k, f')
        model = {y: 0.5 * k * (f - angle_rad)**2}

    angles_rad = np.deg2rad(qm.angles)
    weights = np.exp(-0.2 * np.sqrt(qm.energies))

    fit = Fit (model, angle_rad = angles_rad, y = dihedral_fitting,
               sigma_y = 1/weights, absolute_sigma = False)

    fit_result = fit.execute()

    params = np.round(list(fit_result.params.values()), 4)
    r_squared = np.round(fit_result.r_squared, 4)

    fitted_dihedral = fit.model(angles_rad, **fit_result.params)[0]

    return params, r_squared, fitted_dihedral

def write_opt_dihedral(dir_name, in_dihedrals, inp, qm, c, r_squared):
    """
    Write the optimized dihedrals into the new itp file.
    3 options can be provided for the dihedral function :
    bellemans, periodic, improper
    """
    opt_d = f" ; optimized dihedral with r-squared: {r_squared:6.4f}\n"
    belle = ("{:>5}{:>5}{:>5}{:>5}    3 {:>11.4f}{:>11.4f}{:>11.4f}{:>11.4f}"
             "{:>11.4f}{:>11.4f}" + opt_d)
    perio = ("{:>5}{:>5}{:>5}{:>5}    1 {:>11.4f}{:>11.4f}{:>5}" + opt_d)
    impro = ("{:>5}{:>5}{:>5}{:>5}    2 {:>11.4f}{:>11.4f}" + opt_d)

    with open(dir_name, "a") as opt_itp:
        if not in_dihedrals:
            opt_itp.write("\n[ dihedrals ]\n")

        if inp.fitting_function == "bellemans":
            opt_itp.write(belle.format(*qm.scanned_atoms, *c))
        elif inp.fitting_function == "periodic":
                n_funct = int(len(c)/2)
                count = 0
                for multi in [3]:
                    if c[n_funct + count] > 1.0:
                        angle = 0
                        force_k = c[n_funct + count]
                        opt_itp.write(perio.format(*qm.scanned_atoms,
                                                   angle, force_k, multi))
                    count+=1
        elif inp.fitting_function == "improper":
            opt_itp.write(impro.format(*qm.scanned_atoms, *c))

def plot_dihedral_profile(qm, md_energies, scan_name):
#    title = "QM vs Q-Force Energies" #for dihedral: {} {} {} {}
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width,height), dpi=300)
    sns.set(font_scale=1.3)
    sns.set_style("ticks", {'legend.frameon':True})
#    plt.title(title.format(*qm.scanned_atoms))
    plt.xlabel('Angle')
    plt.ylabel('Energy (kJ/mol)')
    plt.plot(qm.angles, qm.energies, linewidth = 4, label = 'QM')
    plt.plot(qm.angles, md_energies, linewidth = 4, label='Q-Force')
    plt.xticks(np.arange(0,361,60))
    plt.legend(ncol=2, loc=9)
    plt.tight_layout()
    f.savefig(f"{scan_name}.pdf", bbox_inches='tight')

