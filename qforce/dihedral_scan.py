import subprocess
import os
import shutil
import numpy as np
import seaborn as sns
from ase.optimize import BFGS
import scipy.optimize as optimize
from ase import Atoms
from ase.io import read
from scipy.interpolate import interp1d as interpolate
from numba import jit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
from colt import Colt
#
from .forcefield import ForceField
from .calculator import QForce
from .forces import get_dihed, get_dist

"""

Fit all dihedrals togethers after  the scans?
If dihedrals are not relaxed it is possible with 1 iteration - can do more also

"""


class DihedralScan(Colt):
    _user_input = """
# Perform dihedral scan for flexible dihedrals
do_scan = yes :: bool

# Skip dihedrals with missing scan data and accept scan data that is missing data points.
# False: Exit if scan data for a dihedral is missing or if it has missing data points)
avail_only = no :: bool

# Number of neighbors after bonds can be fragmented (0 or smaller means no fragmentation)
frag_threshold = 3 :: int

# Break C-O type of bonds while creating fragments (O-C is never broken)
break_co_bond = no :: bool

# Method for doing the MM relaxed dihedral scan
method = qforce :: str :: [qforce, gromacs]

# The executable for gromacs - necessary if scan method is gromacs
gromacs_exec = gmx :: str

# Number of iterations of dihedral fitting
n_dihed_scans = 5 :: int

# Symmetrize the dihedral profile of a specific dihedral by inputting the range
# For symmetrizing the dihedral profile between atoms 77 and 80 where 0-180 is inversely
# equivalent to 180-360:
# 77 80 = 0 180 360 : +-
symmetrize = :: literal

# Save extra plots with fitting data
plot_fit = no :: bool

# Directory where the fragments are saved
frag_lib = ~/qforce_fragments :: folder

# Skip the dihedrals that are generated but not computed
batch_run = False :: bool
"""

    def __init__(self, fragments, mol, job, all_config):
        self.frag_dir = job.frag_dir
        self.job_name = job.name
        self.mdp_file = f'{job.md_data}/default.mdp'
        self.config = all_config.scan
        self.symmetrize = self._set_symmetrize()
        self.scan = getattr(self, f'scan_dihed_{self.config.method.lower()}')
        self.move_capping_atoms(fragments)

        fragments, all_dih_terms, weights = self.arrange_data(mol, fragments)
        final_energy, params = self.scan_dihedrals(fragments, mol, all_config, all_dih_terms,
                                                   weights)
        self.finalize_results(fragments, final_energy, all_dih_terms, params)

    def arrange_data(self, mol, fragments):
        all_dih_terms, weights = [], []

        for term in mol.terms['dihedral/flexible']:
            if str(term) not in all_dih_terms:
                all_dih_terms.append(str(term))

        for n_fit, frag in enumerate(fragments, start=1):
            angles = []
            for term in frag.terms['dihedral/flexible']:
                if all([atom in [0, 1] for atom in term.atomids[1:3]]):
                    frag.fit_terms.append({'name': str(term), 'atomids': term.atomids,
                                           'params': np.zeros(6)})

            for i, coord in enumerate(frag.qm_coords):
                angle = get_dihed(coord[frag.scanned_atomids])[0]
                angles.append(angle)

            angles = np.array(angles)

            angles[angles < 0] += 2*np.pi
            order = np.argsort(angles)

            if frag.central_atoms in self.symmetrize.keys():
                angles, qm_energies = self.symmetrize_dihedral(angles, frag.qm_energies,
                                                               self.symmetrize[frag.central_atoms])

            frag.qm_angles = angles[order]
            frag.qm_coords = frag.qm_coords[order]
            frag.coords = frag.qm_coords.copy()
            frag.qm_energies = frag.qm_energies[order]
            frag.fit_weights = np.exp(-0.2 * np.sqrt(frag.qm_energies))  # for curve_fit 1/w
            weights.extend(frag.fit_weights)

        return fragments, all_dih_terms, np.array(weights)

    def finalize_results(self, fragments, final_energy, all_dih_terms, params):
        sum_scans = 0
        bad_fits = []

        for frag in fragments:
            n_scans = frag.qm_energies.size
            final = final_energy[sum_scans:sum_scans+n_scans]
            unfit_energy = final.copy()
            fit_sum = np.zeros(n_scans)

            for term in frag.fit_terms:
                if all(term['atomids'][1:3] == frag.scanned_atomids[1:3]):
                    fit_sum += calc_rb_pot(term['params'], term['angles'])

            unfit_energy -= fit_sum
            r_squared = calc_r_squared(fit_sum, frag.qm_energies-unfit_energy)

            self.plot_results(frag, final, 'scan', r_squared=r_squared)

            np.save(f'{self.frag_dir}/scan_data_{frag.id}', np.vstack((frag.qm_angles,
                                                                       frag.qm_energies,
                                                                       final)))

            if self.config.plot_fit:
                self.plot_fit(frag, frag.qm_energies-unfit_energy, fit_sum, r_squared)
                unfit_energy -= unfit_energy.min()
                self.plot_results(frag, unfit_energy, 'unfit')
                np.save(f'{self.frag_dir}/fit_data_{frag.id}', np.vstack((frag.qm_angles,
                                                                          frag.qm_energies,
                                                                          unfit_energy,
                                                                          fit_sum)))
            energy_diff = frag.qm_energies - final
            if np.any(energy_diff > 2.0) and r_squared < 0.9:
                bad_fits.append(frag.id)
            sum_scans += n_scans

        if bad_fits:
            print('WARNING: R-squared < 0.9 for the dihedral fit of the following fragment(s):')
            for bad_fit in bad_fits:
                print(f'         - {bad_fit}')
            print('         Please check manually to see if you find the accuracy satisfactory.\n')

    def scan_dihedrals(self, fragments, mol, all_config, all_dih_terms, weights):
        for n_run in range(self.config.n_dihed_scans):
            energy_diffs, md_energies = [], []
            for n_fit, frag in enumerate(fragments, start=1):
                print(f'Run {n_run+1}/{self.config.n_dihed_scans}, fitting dihedral '
                      f'{n_fit}/{len(fragments)}: {frag.id}')

                scan_dir = f'{self.frag_dir}/{frag.id}'
                make_scan_dir(scan_dir)

                for term in frag.fit_terms:
                    term['angles'] = []

                md_energy = self.scan(all_config, frag, scan_dir, mol, n_run)
                md_energy -= md_energy.min()

                if frag.central_atoms in self.symmetrize.keys():
                    _, md_energy = self.symmetrize_dihedral(frag.angles, md_energy,
                                                            self.symmetrize[frag.central_atoms])
                frag.energy_diff = frag.qm_energies - md_energy
                energy_diffs.extend(frag.energy_diff)
                md_energies.extend(md_energy)

                # if n_run < 2:
                #     low_e_idx = []
                #     for i, _ in enumerate(md_energy):
                #         neigh_energies = [md_energy[(i+n) % md_energy.size] for n in range(-1, 2)]
                #         low_e_idx.append((i+np.argmin(neigh_energies)-1) % md_energy.size)
                #     frag.coords = frag.coords[low_e_idx]

            params, matrix = self.fit_dihedrals(fragments, energy_diffs, weights, all_dih_terms)

            for frag in fragments:
                for term in frag.terms['dihedral/flexible']:
                    term_idx = all_dih_terms.index(str(term))
                    term.equ += params[6*term_idx:(6*term_idx)+6]

                for term in frag.fit_terms:
                    term_idx = all_dih_terms.index(term['name'])
                    term['params'] += params[6*term_idx:(6*term_idx)+6]

            for term in mol.terms['dihedral/flexible']:
                term_idx = all_dih_terms.index(str(term))
                term.equ += params[6*term_idx:(6*term_idx)+6]

        print('Done!\n')
        final_energy = np.array(md_energies) + np.sum(matrix * params, axis=1)

        return final_energy, params

    @staticmethod
    def move_capping_atoms(fragments):
        for frag in fragments:
            for cap in frag.caps:
                for coord in frag.coords:
                    vec, dist = get_dist(coord[cap['idx']], coord[cap['connected']])
                    new_vec = vec / dist * cap['b_length']
                    coord[cap['idx']] = coord[cap['connected']] + new_vec

    def scan_dihed_qforce(self, all_config, frag, scan_dir, mol, n_run, nsteps=1000):
        md_energies = []

        for i, coord in enumerate(frag.coords):
            restraints = self.find_restraints(frag, frag.qm_coords[i], n_run)
            atom = Atoms(frag.elements, positions=coord,
                         calculator=QForce(frag.terms, dihedral_restraints=restraints))

            traj_name = f'{scan_dir}/{frag.id}_run{n_run+1}_{i:02d}.traj'
            log_name = f'{scan_dir}/opt_{frag.id}_run{n_run+1}.log'
            e_minimiz = BFGS(atom, trajectory=traj_name, logfile=log_name)
            e_minimiz.run(fmax=0.01, steps=nsteps)
            coords = atom.get_positions()
            self.calc_fit_angles(frag, coords)
            md_energies.append(atom.get_potential_energy())
            frag.coords[i] = coords
        return np.array(md_energies)

    def scan_dihed_gromacs(self, all_config, frag, scan_dir, mol, n_run):
        md_energies = []

        ff = ForceField(self.job_name, all_config, frag, frag.neighbors,
                        exclude_all=frag.remove_non_bonded)

        for i, coord in enumerate(frag.coords):
            step_dir = f"{scan_dir}/step{i:02d}"
            make_scan_dir(step_dir)

            ff.write_gromacs(step_dir, frag, coord)
            shutil.copy2(self.mdp_file, step_dir)

            restraints = self.find_restraints(frag, frag.qm_coords[i], n_run)
            ff.add_restraints(restraints, step_dir)

            run_gromacs(step_dir, all_config.scan.gromacs_exec, ff.polar_title)
            md_energy = read_gromacs_energies(step_dir)
            md_energies.append(md_energy)

            coords = read(f'{step_dir}/geom.gro').get_positions()
            self.calc_fit_angles(frag, coords)
            frag.coords[i] = coords
        return np.array(md_energies)

    @staticmethod
    def calc_fit_angles(frag, coords):
        for term in frag.fit_terms:
            term['angles'].append(get_dihed(coords[term['atomids']])[0])

    @staticmethod
    def find_restraints(frag, coord, n_run):
        restraints = []
        for term in frag.terms['dihedral/flexible']:
            phi0 = get_dihed(coord[term.atomids])[0]
            # if any([cap['idx'] in term.atomids for cap in frag.caps]):
            # if not any([idx in term.atomids for idx in frag.scanned_atomids[1:3]]):
            #     phi = get_dihed(frag.qm_coords[0][term.atomids])[0]
            #     restraints.append([term.atomids, phi])
            if (all([term.atomids[i] == frag.scanned_atomids[i] for i in range(3)])
                # or not all([idx in term.atomids for idx in frag.scanned_atomids[1:3]])):
                    or n_run == 0):
                restraints.append([term.atomids, phi0])

        return restraints

    @staticmethod
    def fit_dihedrals(fragments, energy_diffs, weights, all_dih_terms):
        energy_diffs = np.array(energy_diffs)
        n_total_scans = energy_diffs.size
        matrix = calc_multi_rb_matrix(fragments, all_dih_terms, n_total_scans)
        params = optimize.minimize(calc_multi_rb_obj, x0=np.zeros(len(all_dih_terms)*6),
                                   args=(matrix, weights, energy_diffs)).x
        return params, matrix

    @staticmethod
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
            ip_angle = np.arange(region['start'], make_contin(region['start'], region['end'])+1)
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

        sym_angle = np.radians(np.arange(regions[0]['start'], make_contin(regions[0]['start'],
                                                                          regions[-1]['end'])))
        sym_profile -= sym_profile.min()
        return sym_angle, sym_profile

    def plot_results(self, frag, md_energies, title, r_squared=None):
        angles_deg = np.degrees(frag.qm_angles)
        width, height = plt.figaspect(0.6)
        f = plt.figure(figsize=(width, height), dpi=300)
        sns.set(font_scale=1.3)
        plt.xlabel('Angle')
        plt.ylabel('Energy (kJ/mol)')
        plt.plot(angles_deg, frag.qm_energies, linewidth=4, label='QM')
        plt.plot(angles_deg, md_energies, linewidth=4, label='Q-Force')
        plt.xticks(np.arange(0, 361, 60))
        plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
        if r_squared:
            plt.title(f'R-squared = {round(r_squared, 3)}', loc='left')
        plt.tight_layout()
        f.savefig(f"{self.frag_dir}/{title}_data_{frag.id}.pdf", bbox_inches='tight')
        plt.close()

    def plot_fit(self, frag, diff, fit, r_squared):
        angles_deg = np.degrees(frag.qm_angles)
        width, height = plt.figaspect(0.6)
        f = plt.figure(figsize=(width, height), dpi=300)
        sns.set(font_scale=1.3)
        plt.xlabel('Angle')
        plt.ylabel('Energy (kJ/mol)')
        plt.plot(angles_deg, diff, linewidth=4, label='Diff')
        plt.plot(angles_deg, fit, linewidth=4, label='Fit')
        plt.xticks(np.arange(0, 361, 60))
        plt.tight_layout()
        plt.title(f'R-squared = {round(r_squared, 3)}', loc='left')
        plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
        f.savefig(f"{self.frag_dir}/fit_data_{frag.id}.pdf", bbox_inches='tight')
        plt.close()

    def _set_symmetrize(self):
        sym_dict = {}
        if self.config.symmetrize:
            for line in self.config.symmetrize.split('\n'):
                atom_info = line.split()
                if len(atom_info) > 1:
                    atoms = tuple(sorted([int(atom_info[0])-1, int(atom_info[1])-1]))
                    sym_dict[atoms] = []
                    sym_info = line.partition('=')[2]
                    angles, _, direct = sym_info.partition(':')
                    angles = [float(a) for a in angles.split()]
                    direct = [d for d in list(direct) if d != ' ']

                    for i in range(len(angles)-1):
                        sym_dict[atoms].append({'start': get_periodic_angle(angles[i]),
                                                'end': get_periodic_angle(angles[i+1]),
                                                'direct': direct[i] == '+'})
        return sym_dict


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


def make_contin(start, end):
    if end <= start:
        end += 360
    return end


def make_scan_dir(scan_name):
    if os.path.exists(scan_name):
        shutil.rmtree(scan_name)
    os.makedirs(scan_name)


def run_gromacs(directory, gromacs_exec, polar_title):
    attempt, returncode = 0, 1
    grompp = subprocess.Popen([gromacs_exec, 'grompp', '-f', 'default.mdp', '-p',
                               f'gas{polar_title}.top', '-c', f'gas{polar_title}.gro', '-o',
                               'em.tpr', '-po', 'em.mdp', '-maxwarn', '10'],
                              cwd=directory, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    grompp.wait()

    while returncode != 0 and attempt < 5:
        check_gromacs_termination(grompp)
        mdrun = subprocess.Popen([gromacs_exec, 'mdrun', '-deffnm', 'em'],
                                 cwd=directory, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        mdrun.wait()
        returncode = mdrun.returncode
        attempt += 1

    check_gromacs_termination(mdrun)

    trjconv = subprocess.Popen([gromacs_exec, 'trjconv', '-f', 'em.gro', '-s', 'em.tpr', '-pbc',
                                'whole', '-center', '-o', 'geom.gro'], cwd=directory,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.PIPE)
    trjconv.communicate(input=b'0\n0\n')


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


def calc_r_squared(rb, energy_diff):
    residuals = rb - energy_diff
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((energy_diff-np.mean(energy_diff))**2)
    return 1 - (ss_res / ss_tot)


@jit(nopython=True)
def calc_multi_rb_obj(params, matrix, weights, energy_diffs):
    rb = np.sum(matrix * params, axis=1)
    weighted_residuals = (rb - energy_diffs)*weights
    return (weighted_residuals**2).sum() + (params**2).sum()*1e-2


def calc_multi_rb_matrix(fragments, all_dih_terms, n_total_scans):
    scan_sum = 0
    n_dihs = len(all_dih_terms)
    n_terms = n_dihs*6

    matrix = np.zeros((n_total_scans, n_terms))
    for frag in fragments:
        n_scans = len(frag.qm_angles)
        for term in frag.fit_terms:
            term_idx = all_dih_terms.index(term['name'])
            matrix[scan_sum:scan_sum+n_scans,
                   term_idx*6:(term_idx*6)+6] += calc_rb(term['angles'])
        scan_sum += n_scans
    return matrix


def calc_rb(angles):
    rb = np.zeros((len(angles), 6))
    rb[:, 0] = 1
    cos_phi = np.cos(np.array(angles)-np.pi)
    cos_factor = np.ones(len(angles))
    for i in range(1, 6):
        cos_factor *= cos_phi
        rb[:, i] = cos_factor
    return rb


def calc_rb_pot(params, angles):
    rb = np.full(len(angles), params[0])
    for i in range(1, 6):
        rb += params[i]*np.cos(np.array(angles)-np.pi)**i
    return rb
