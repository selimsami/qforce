import subprocess
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase.optimize import BFGS
from ase import Atoms
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d as interpolate
#
from . import qforce_data
from .forcefield import ForceField
from .calculator import QForce
from .forces import get_dihed, get_dist

"""

Fit all dihedrals togethers after  the scans?
If dihedrals are not relaxed it is possible with 1 iteration - can do more also

"""


def scan_dihedrals(fragments, inp, mol):

    move_capping_atoms(fragments)

    for n_run in range(inp.n_dihed_scans):
        for n_fit, frag in enumerate(fragments, start=1):

            print(f'Run {n_run+1}/{inp.n_dihed_scans}, fitting dihedral {n_fit}/{len(fragments)}: '
                  f'{frag.id}')

            scanned = list(frag.terms.get_terms_from_name(name=frag.name,
                                                          atomids=frag.scanned_atomids))[0]
            scanned.equ = np.zeros(6)

            scan_dir = f'{inp.frag_dir}/{frag.id}'
            make_scan_dir(scan_dir)

            if inp.scan_method == 'qforce':
                md_energies, angles = scan_dihed_qforce(frag, scan_dir, inp, mol, n_run)

            elif inp.scan_method == 'gromacs':
                md_energies, angles = scan_dihed_gromacs(frag, scan_dir, inp, mol, n_run)

            params = fit_dihedrals(frag, angles, md_energies, inp)

            for frag2 in fragments:
                for term in frag2.terms.get_terms_from_name(frag.name):
                    term.equ = params

            for term in mol.terms.get_terms_from_name(frag.name):
                term.equ = params
    print('Done!')


def move_capping_atoms(fragments):
    for frag in fragments:
        for cap in frag.caps:
            for coord in frag.coords:
                vec, dist = get_dist(coord[cap['idx']], coord[cap['connected']])
                new_vec = vec / dist * cap['b_length']
                coord[cap['idx']] = coord[cap['connected']] + new_vec


def scan_dihed_qforce(frag, scan_dir, inp, mol, n_run, nsteps=1000):
    md_energies, angles, ignores = [], [], []

    for i, (qm_energy, coord) in enumerate(zip(frag.qm_energies, frag.coords)):
        angle = get_dihed(coord[frag.scanned_atomids])[0]
        angles.append(angle)

        restraints = find_restraints(frag, coord, n_run)

        atom = Atoms(frag.elements, positions=coord,
                     calculator=QForce(frag.terms, ignores, dihedral_restraints=restraints))

        traj_name = f'{scan_dir}/{frag.id}_run{n_run+1}_{np.degrees(angle).round()}.traj'
        log_name = f'{scan_dir}/opt_{frag.id}_run{n_run+1}.log'
        e_minimiz = BFGS(atom, trajectory=traj_name, logfile=log_name)
        e_minimiz.run(fmax=0.01, steps=nsteps)
        frag.coords[i] = atom.get_positions()

        md_energies.append(atom.get_potential_energy())

    return np.array(md_energies), np.array(angles)


def scan_dihed_gromacs(frag, scan_dir, inp, mol, n_run):
    md_energies, angles = [], []

    ff = ForceField(inp, frag, frag.neighbors, exclude_all=frag.remove_non_bonded)

    for i, (qm_energy, coord) in enumerate(zip(frag.qm_energies, frag.coords)):

        angles.append(get_dihed(coord[frag.scanned_atomids])[0])

        step_dir = f"{scan_dir}/step{i}"
        make_scan_dir(step_dir)

        ff.write_gromacs(inp, frag, step_dir, coord)
        shutil.copy2(f'{qforce_data}/default.mdp', step_dir)

        restraints = find_restraints(frag, coord, n_run)
        ff.add_restraints(restraints, step_dir)

        run_gromacs(step_dir, inp, ff.polar_title)
        md_energy = read_gromacs_energies(step_dir)
        md_energies.append(md_energy)

    return np.array(md_energies), np.array(angles)


def find_restraints(frag, coord, n_run):
    restraints = []
    for term in frag.terms['dihedral/flexible']:
        phi0 = get_dihed(coord[term.atomids])[0]
        if (all([term.atomids[i] == frag.scanned_atomids[i] for i in range(3)])):
            # or not all([idx in term.atomids for idx in frag.scanned_atomids[1:3]])):
            # n_run == 0 or
            restraints.append([term.atomids, phi0])
    return restraints


def fit_dihedrals(frag, angles, md_energies, inp):
    angles[angles < 0] += 2*np.pi

    order = np.argsort(angles)
    angles = angles[order]
    md_energies = md_energies[order]
    qm_energies = frag.qm_energies[order]
    md_energies -= md_energies.min()

    if inp.sym_scan and frag.central_atoms in list(inp.sym_scan.keys()):
        _, qm_energies = symmetrize_dihedral(angles, qm_energies, inp.sym_scan[frag.central_atoms])
        angles, md_energies = symmetrize_dihedral(angles, md_energies,
                                                  inp.sym_scan[frag.central_atoms])

    energy_diff = qm_energies - md_energies
    weights = 1/np.exp(-0.2 * np.sqrt(qm_energies))
    params = curve_fit(calc_rb, angles, energy_diff, absolute_sigma=False, sigma=weights)[0]
    r_squared = calc_r_squared(calc_rb, angles, energy_diff, params)

    plot_results(inp, frag, angles, qm_energies, md_energies, r_squared, 'unfit')
    md_energies += calc_rb(angles, *params)
    plot_fit(inp, frag, angles, energy_diff, calc_rb(angles, *params))
    plot_results(inp, frag, angles, qm_energies, md_energies, r_squared, 'scan')

    np.save(f'{inp.frag_dir}/scan_data_{frag.id}', np.vstack((angles, qm_energies, md_energies)))
    return params


def symmetrize_dihedral(angles, energies, regions):
    sym_profile = np.array([])
    energy_sum = np.inf
    angles_deg = np.degrees(angles)
    spacing = get_periodic_angle(abs(angles_deg[1] - angles_deg[0]))

    for region in regions:
        select = get_periodic_range(angles_deg, region['start'], region['end'], spacing)
        ang_reg = angles_deg[select]
        energy_reg = energies[select]

        ang_continious = np.copy(ang_reg)
        ang_continious[ang_reg < region['start']-spacing] += 360
        ip_funct = interpolate(ang_continious, energy_reg, fill_value="extrapolate", kind=2)
        ip_angle = np.arange(region['start'], make_continious(region['start'], region['end'])+1)
        ip_energy = ip_funct(ip_angle)

        if ip_energy.sum() < energy_sum:
            energy_sum = ip_energy.sum()
            if not region['direct']:
                lowest = ip_energy[::-1]
            else:
                lowest = ip_energy

    for region in regions:
        if region['direct']:
            current = lowest
        else:
            current = lowest[::-1]
        sym_profile = np.concatenate((sym_profile, current[:-1]))

    sym_angle = np.radians(np.arange(regions[0]['start'], make_continious(regions[0]['start'],
                                                                          regions[-1]['end'])))
    sym_profile -= sym_profile.min()
    return sym_angle, sym_profile


def get_periodic_angle(angle):
    if angle > 360:
        angle -= 360
    elif angle < 0:
        angle += 360
    return angle


def get_periodic_angles(angles):
    angles[angles > 360] -= 360
    angles[angles < 0] += 360
    return angles


def get_periodic_range(angles, start, end, spacing):
    half_spacing = spacing/2
    if end > start:
        select = (start-half_spacing <= angles) * (angles <= end+half_spacing)
    elif end < start:
        select = (angles >= start-half_spacing) + (angles <= end+half_spacing)
    return select


def make_continious(start, end):
    if end <= start:
        end += 360
    return end


def make_scan_dir(scan_name):
    if os.path.exists(scan_name):
        shutil.rmtree(scan_name)
    os.makedirs(scan_name)


def run_gromacs(directory, inp, polar_title):
    attempt, returncode = 0, 1
    grompp = subprocess.Popen(['gmx_d', 'grompp', '-f', 'default.mdp', '-p',
                               f'gas{polar_title}.top', '-c', f'gas{polar_title}.gro', '-o',
                               'em.tpr', '-po', 'em.mdp', '-maxwarn', '10'],
                              cwd=directory, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    grompp.wait()

    while returncode != 0 and attempt < 5:
        check_gromacs_termination(grompp)
        mdrun = subprocess.Popen(['gmx_d', 'mdrun', '-deffnm', 'em'],
                                 cwd=directory, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        mdrun.wait()
        returncode = mdrun.returncode
        attempt += 1

    check_gromacs_termination(mdrun)


def check_gromacs_termination(process):
    if process.returncode != 0:
        print(process.communicate()[0].decode("utf-8"))
        raise RuntimeError({"GROMACS run has terminated unsuccessfully"})


def read_gromacs_energies(directory):
    log_dir = f"{directory}/em.log"
    with open(log_dir, "r", encoding='utf-8') as em_log:
        for line in em_log:
            if "Potential Energy  =" in line:
                md_energy = float(line.split()[3])
    return md_energy


def plot_results(inp, frag, angles, qm_energies, md_energies, r_squared, title):
    angles_deg = np.degrees(angles)
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width, height), dpi=300)
    sns.set(font_scale=1.3)
    plt.xlabel('Angle')
    plt.ylabel('Energy (kJ/mol)')
    plt.plot(angles_deg, qm_energies, linewidth=4, label='QM')
    plt.plot(angles_deg, md_energies, linewidth=4, label='Q-Force')
    plt.xticks(np.arange(0, 361, 60))
    plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
    if title == 'scan':
        plt.title(f'R-squared = {round(r_squared, 3)}', loc='left')
    plt.tight_layout()
    f.savefig(f"{inp.frag_dir}/{title}_data_{frag.id}.pdf", bbox_inches='tight')
    plt.close()


def plot_fit(inp, frag, angles, diff, fit):
    angles_deg = np.degrees(angles)
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width, height), dpi=300)
    sns.set(font_scale=1.3)
    plt.xlabel('Angle')
    plt.ylabel('Energy (kJ/mol)')
    plt.plot(angles_deg, diff, linewidth=4, label='Diff')
    plt.plot(angles_deg, fit, linewidth=4, label='Fit')
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
    f.savefig(f"{inp.frag_dir}/fit_data_{frag.id}.pdf", bbox_inches='tight')
    plt.close()


def calc_r_squared(funct, x, y, params):
    residuals = y - funct(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - (ss_res / ss_tot)


def calc_rb(angles, c0, c1, c2, c3, c4, c5):
    params = [c0, c1, c2, c3, c4, c5]

    rb = np.full(len(angles), c0)
    for i in range(1, 6):
        rb += params[i] * np.cos(angles-np.pi)**i
    return rb
