import subprocess
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from symfit import Fit, cos, pi, parameters, variables
import seaborn as sns
from . import qforce_data


def scan_dihedral(inp, atoms, scan_id):
    """
    For each different dihedral in the input file, do dihedral optimization:
    Read scan QM output(s), run GROMACS with the same angles, get the fitting
    data, fit into a choice function of choice, run GROMACS again with fitted
    dihedrals, print the dihedral profile.
    """
    md_energies, opt_md_energies = [], []
    frag_id, id_no = scan_id.split('~')
    scan_name = '_'.join([str(a+1) for a in atoms])
    scan_dir = f'{inp.frag_dir}/{scan_name}'
    itp_file = f'{inp.job_dir}/{inp.job_name}_qforce.itp'
    data_file = f'{inp.frag_lib}/{frag_id}/scandata_{id_no}'
    qm_angles, qm_energies = np.loadtxt(data_file, unpack=True)

    remove_scanned_dihedral(itp_file, atoms)

    make_scan_dir(scan_dir)

    # print job info
    print(f"Scan name: {scan_name}")
    print("-"*(11+len(scan_name)))

    # Prepare the files for a GROMACS run in each scan step directory
    prepare_scan_directories(atoms, qm_angles, qm_energies, inp, itp_file, scan_dir)

    # Run gromacs without the scanned dihedral - get energies
    print("Running GROMACS without the scanned dihedral...")
    for step, angle in enumerate(qm_angles):
        step_dir = f"{scan_dir}/step{step}"
        run_gromacs(step_dir, "nodihed", inp)
        md_energy = read_gromacs_energies(step_dir, "nodihed")
        md_energies.append(md_energy)

    # Set minimum energies to zero and compute QM vs MD difference
    # and the dihedral potential to be fitted
    md_energies = set_minimum_to_zero(md_energies)
    dihedral_fitting = set_minimum_to_zero(qm_energies - md_energies)

    # fit the data
    print("Fitting the dihedral function...")
    c, r_squared, fitted_dihedral = do_fitting(qm_angles, qm_energies, inp, dihedral_fitting)

    # print optmized dihedrals
    write_opt_dihedral(itp_file, atoms, c, r_squared)

    # run gromacs again with optimized dihedrals
    print("Running GROMACS with the fitted dihedral...")
    for step, angle in enumerate(qm_angles):
        step_dir = f"{scan_dir}/step{step}"
        itp_loc = f"{step_dir}/{inp.job_name}_qforce.itp"
        write_opt_dihedral(itp_loc, atoms, c, r_squared)
        run_gromacs(step_dir, "opt", inp)
        md_energy = read_gromacs_energies(step_dir, "opt")
        opt_md_energies.append(md_energy)

    # set minimum energy to zero
    opt_md_energies = set_minimum_to_zero(opt_md_energies)

    # Plot optimized dihedral profile vs QM profile
    plot_dihedral_profile(inp, qm_angles, qm_energies, opt_md_energies,
                          scan_name)

    print(f"QM vs MD dihedral scan profile: {scan_name}.pdf\n")


def remove_scanned_dihedral(itp_path, atoms):
    """
    Read the itp file, create an initial optimized itp file minus the scanned
    dihedrals. Get molecule residue info and atom names from the ITP file
    """
    in_section = None
    atoms = [str(a+1) for a in atoms]
    with open(itp_path, 'r') as itp_file:
        itp = itp_file.readlines()

    with open(itp_path, 'w') as itp_file:
        for line in itp:
            low_line = line.strip().lower()
            if "[" in low_line and "]" in low_line:
                no_space = low_line.replace(" ", "")
                open_bra = no_space.index("[") + 1
                close_bra = no_space.index("]")
                in_section = no_space[open_bra:close_bra]
                itp_file.write(line)
            elif in_section == "dihedrals" and len(line.split()) > 3:
                atoms_check = line.split()[0:4]
                # if atoms == atoms_check:
                #     continue
                # else:
                itp_file.write(line)
            else:
                itp_file.write(line)


def make_scan_dir(scan_name):
    if os.path.exists(scan_name):
        shutil.rmtree(scan_name)
    os.makedirs(scan_name)


def set_minimum_to_zero(energies):
    energies = energies - np.amin(energies)
    return energies


def prepare_scan_directories(atoms, qm_angles, qm_energies, inp, itp_dir,
                             scan_dir):
    """
    Create the .gro file for each scan step from QM optimized geometriesl
    Add dihedral restrain to each scan step ITP
    Copy all necessary files for a GROMACS run to scan directories
    """
    dihed_res = "{:>5}{:>5}{:>5}{:>5}{:>5}{:>10.4f}  0.0  1000\n"
    atoms = [a+1 for a in atoms]
    itp_file = f'{inp.job_name}_qforce.itp'
    top_dir = f'{inp.job_dir}/gas.top'

    for step, angle in enumerate(qm_angles):

        step_dir = f"{scan_dir}/step{step}"
        os.makedirs(step_dir)

        # copy the .itp file to each scan directory & add the dihedral restrain
        shutil.copy2(itp_dir, step_dir)
        with open(f'{step_dir}/{itp_file}', "a") as itp_step:
            itp_step.write("\n")
            itp_step.write("[ dihedral_restraints ]\n")
            itp_step.write(";  ai   aj   ak   al type       phi   dp   kfac\n")
            itp_step.write(dihed_res.format(*atoms, "1", angle))

        # copy the .top file and the extra .itp files (if they exist)
        shutil.copy2(top_dir, step_dir)
        shutil.copy2(f'{qforce_data}/default.mdp', step_dir)


def run_gromacs(directory, em_type, inp):
    grompp = subprocess.Popen(['gmx', 'grompp', '-f', 'default.mdp', '-p',
                               'gas.top', '-c', '../../../gas.gro', '-o',
                               ('em_' + em_type + '.tpr'), '-po',
                               ('em_' + em_type + '.mdp'), '-maxwarn', '10'],
                              cwd=directory, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    grompp.wait()
    check_gromacs_termination(grompp)
    mdrun = subprocess.Popen(['gmx', 'mdrun', '-deffnm', ('em_'+em_type)],
                             cwd=directory, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    mdrun.wait()
    check_gromacs_termination(mdrun)


def check_gromacs_termination(process):
    if process.returncode != 0:
        print(process.communicate()[0].decode("utf-8"))
        raise RuntimeError({"GROMACS run has terminated unsuccessfully"})


def read_gromacs_energies(directory, em_type):
    log_dir = "{}/em_{}.log".format(directory, em_type)
    with open(log_dir, "r", encoding='utf-8') as em_log:
        for line in em_log:
            if "Potential Energy  =" in line:
                md_energy = float(line.split()[3])
    return md_energy


def do_fitting(qm_angles, qm_energies, inp, dihedral_fitting):
    angle_rad, y = variables('angle_rad, y')

    c0, c1, c2, c3, c4, c5 = parameters('c0, c1, c2, c3, c4, c5')
    model = {y: c0 + c1 * cos(angle_rad - pi) + c2 * cos(angle_rad - pi)**2
             + c3 * cos(angle_rad - pi)**3 + c4 * cos(angle_rad - pi)**4
             + c5 * cos(angle_rad - pi)**5}

    angles_rad = np.deg2rad(qm_angles)
    weights = 1/np.exp(-0.2 * np.sqrt(qm_energies))

    fit = Fit(model, angle_rad=angles_rad, y=dihedral_fitting,
              sigma_y=weights, absolute_sigma=False)

    fit_result = fit.execute()

    params = np.round(list(fit_result.params.values()), 4)
    r_squared = np.round(fit_result.r_squared, 4)

    fitted_dihedral = fit.model(angles_rad, **fit_result.params)[0]

    return params, r_squared, fitted_dihedral


def write_opt_dihedral(itp_file, atoms, c, r_squared):
    """
    Write the optimized dihedrals into the new itp file.
    3 options can be provided for the dihedral function :
    bellemans, periodic, improper
    """
    opt_d = f" ; optimized dihedral with r-squared: {r_squared:6.4f}\n"
    belle = ("{:>5}{:>5}{:>5}{:>5}    3 {:>11.4f}{:>11.4f}{:>11.4f}{:>11.4f}"
             "{:>11.4f}{:>11.4f}" + opt_d)
    atoms = [a+1 for a in atoms]
    # with open(itp_file, "a") as opt_itp:
    #     opt_itp.write("\n[ dihedrals ]\n")
    #     opt_itp.write(belle.format(*atoms, *c))


def plot_dihedral_profile(inp, qm_angles, qm_energies, md_energies, scan_name):
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width, height), dpi=300)
    sns.set(font_scale=1.3)
    sns.set_style("ticks", {'legend.frameon': True})
    plt.xlabel('Angle')
    plt.ylabel('Energy (kJ/mol)')
    plt.plot(qm_angles, qm_energies, linewidth=4, label='QM')
    plt.plot(qm_angles, md_energies, linewidth=4, label='Q-Force')
    plt.xticks(np.arange(0, 361, 60))
    plt.legend(ncol=2, loc=9)
    plt.tight_layout()
    f.savefig(f"{inp.frag_dir}/{scan_name}.pdf", bbox_inches='tight')
