import subprocess
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase.optimize import BFGS
from ase import Atoms
from scipy.optimize import curve_fit

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

            print(f'\nFitting dihedral {n_fit}/{len(fragments)}: {frag.id} \n')

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

            print(params)


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
        if (n_run == 0
                or all([term.atomids[i] == frag.scanned_atomids[i] for i in range(3)])):
                # or not all([idx in term.atomids for idx in frag.scanned_atomids[1:3]])):
            restraints.append([term.atomids, phi0])
    return restraints


def fit_dihedrals(frag, angles, md_energies, inp):
    angles[angles < 0] += 2*np.pi

    order = np.argsort(angles)
    angles = angles[order]
    md_energies = md_energies[order]
    qm_energies = frag.qm_energies[order]

    if frag.id == 'CO_H12C9O2_0_1_PBEPBE-GD3BJ_6-31+G-D_bf8a6c00dfe92fe61f38991bb2ec92dd~2':
        sym_info = [{'start': 0, 'end': 180, 'direct': True},
                    {'start': 180, 'end': 360, 'direct': False}]
        angles, md_energies, order = symmetrize_potential(angles, md_energies, sym_info)
        qm_energies = qm_energies[order]

    energy_diff = qm_energies - md_energies
    energy_diff -= energy_diff.min()

    weights = 1/np.exp(-0.2 * np.sqrt(qm_energies))
    params = curve_fit(calc_rb, angles, energy_diff, absolute_sigma=False, sigma=weights)[0]
    r_squared = calc_r_squared(calc_rb, angles, energy_diff, params)

    md_energies += calc_rb(angles, *params)
    md_energies -= md_energies.min()

    plot_fit(inp, frag, angles, energy_diff, calc_rb(angles, *params))
    plot_results(inp, frag, angles, qm_energies, md_energies, r_squared)

    np.save(f'{inp.frag_dir}/scan_data_{frag.id}', np.vstack((angles, qm_energies, md_energies)))
    return params


def symmetrize_potential(angles, energy, sym_info):
    regions = []

    for region in sym_info:
        regions.append(RegionRange(region['start'], region['end'], region['direct']))

    sym = Symmetrizer({region: Region(region) for region in regions}, [regions])

    points = [[angle, energy] for angle, energy in zip(np.degrees(angles), energy)]
    points = sym.symmetrize(points)

    points = np.array(points)
    new_angles = np.radians(points[:, 0])
    new_values = points[:, 1]

    order = [np.where(energy == val)[0][0] for val in new_values]

    return new_angles, new_values, order


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


def plot_results(inp, frag, angles, qm_energies, md_energies, r_squared):
    angles_deg = np.degrees(angles)
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width, height), dpi=300)
    sns.set(font_scale=1.3)
    plt.title(f'R-squared = {round(r_squared, 3)}', loc='left')
    plt.xlabel('Angle')
    plt.ylabel('Energy (kJ/mol)')
    plt.plot(angles_deg, qm_energies, linewidth=4, label='QM')
    plt.plot(angles_deg, md_energies, linewidth=4, label='Q-Force')
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
    f.savefig(f"{inp.frag_dir}/scan_data_{frag.id}.pdf", bbox_inches='tight')
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


class Region:

    def __init__(self, re_range, *, buffer_region=5):
        self.start = re_range.start
        self.end = re_range.end
        self.range = buffer_region
        self.direct = re_range.direct

    def __call__(self, value):
        if value < self.start:
            return False
        if value > self.end:
            return False
        return True

    def to_region(self, value, other):
        if not isinstance(other, Region):
            raise ValueError("other needs to be a region")
        # if value already in region, return it
        if self(value) is True:
            return value
        # if value not in other raise Exception
        if other(value) is False:
            raise ValueError("value needs to be in the 'other' region")
        direct = self.direct == other.direct
        # transform
        if direct is True:
            value = self.start + value - other.start
        else:
            value = other.end - value + self.start
        # sanity check
        if self(value) is False:
            raise ValueError("value not in region!")
        #
        return value

    def is_within_range(self, value):
        if self(value) is False:
            return False
        if value < self.start + self.range:
            return True
        if value > self.end - self.range:
            return True
        return False


class RegionRange:

    def __init__(self, start, end, direct=True):
        self.start = float(start)
        self.end = float(end)
        self.direct = direct

    def __str__(self):
        return f"Region({self.start},{self.end})"

    def __repr__(self):
        return f"Region({self.start},{self.end})"

    def __hash__(self):
        return hash(str(RegionRange) + f"{str(id(self))}")


class Symmetrizer:

    def __init__(self, regions, pairs):
        self.regions = regions
        self.joined_regions = pairs

    def symmetrize(self, points):
        points = self._get_regions(points)

        for regions in self.joined_regions:
            self._symmetrize(points, regions)

        return self._cleanup_points(points)

    def _cleanup_points(self, points):
        # get rid of duplicates
        points = {angle: value for values in points.values()
                  for angle, value in values}
        #
        points = [[angle, value] for angle, value in points.items()]
        # sort output
        points.sort(key=lambda x: x[0])
        return points

    def _get_regions(self, points):
        output = {region: [] for region in self.regions}

        for angle, value in points:
            found = False
            for region, validator in self.regions.items():
                if validator(angle) is True:
                    output[region].append([angle, value])
                    found = True
                    break
            if found is False:
                print(f"Could not find:\nangle = {angle}, value = {value}")
        return output

    def _get_smallest(self, pairs, validators):
        smallest = pairs[0]
        validator = validators[0]
        for i, (angle, value) in enumerate(pairs):
            if (value < smallest[1]):
                smallest = (angle, value)
                validator = validators[i]
        return smallest, validator

    def _symmetrize(self, points, regions):
        out = tuple(points[region] if region.direct is True else points[region][::-1]
                    for region in regions)
        zipped = zip(*out)
        #
        validators = [self.regions[region] for region in regions]
        results = []
        #
        for pairs in zipped:
            results.append(self._get_smallest(pairs, validators))
            for i, (angle, value) in enumerate(pairs):
                validator = validators[i]
                if validator.is_within_range(angle) is True:
                    results.append(([angle, value], validator))
        # set results
        for region in regions:
            r1 = self.regions[region]
            points[region] = [[r1.to_region(angle, r2), value] for (angle, value), r2 in results]
