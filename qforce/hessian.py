import scipy.optimize as optimize
# import scipy.linalg as la
# import scipy.optimize.nnls as nnls
from scipy.linalg import eigh
import numpy as np
from .molecule import Molecule
from .read_qm_out import QM
from .read_forcefield import Forcefield
from .write_forcefield import write_ff
from .dihedral_scan import scan_dihedral
from .dftd4 import get_nonbonded
from .fragment import fragment
from .forces import (calc_bonds, calc_angles, calc_pairs, calc_imp_diheds,
                     calc_rb_diheds, calc_cross_bondangle)
# , calc_g96angles
from .elements import elements
# from .decorators import timeit, print_timelog


def fit_forcefield(inp, qm=None, mol=None):
    """
    Scope:
    ------
    Fit MD hessian to the QM hessian.

    TO DO:
    ------
    - Move calc_energy_forces to forces and clean it up
    - Include LJ, Coulomb flex dihed forces in the fitting as numbers

    CHECK
    -----
    - Does having (0,inf) fitting bound cause problems? metpyrl lower accuracy
      for dihed! Having -inf, inf causes problems for PTEG-1 (super high FKs)
    - Fix acetone angle! bond-angle coupling?)
    - Charges from IR intensities - together with interacting polarizable FF?
    """

    qm = QM("freq", fchk_file=inp.fchk_file, out_file=inp.qm_freq_out)
    mol = Molecule(qm.coords, qm.atomids, inp, qm=qm)

    get_nonbonded(inp, mol, qm)

    fit_results, md_hessian = fit_hessian(inp, mol, qm)

    # Fit - add dihedrals - fit again >> Is it enough? More iteration?
    # if not inp.nofrag:
    #     fragment(inp, mol, qm)
    #     fit_results, md_hessian = fit_hessian(inp, mol, qm)

    calc_qm_vs_md_frequencies(inp, qm, md_hessian)

    make_ff_params_from_fit(mol, fit_results, inp, qm)

    # temporary
#    fit_dihedrals(inp, mol, qm)


def calc_qm_vs_md_frequencies(inp, qm, md_hessian):
    qm_freq, qm_vec = calc_vibrational_frequencies(qm.hessian, qm)
    md_freq, md_vec = calc_vibrational_frequencies(md_hessian, qm)
    write_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, inp)


def fit_hessian(inp, mol, qm):
    hessian, full_md_hessian_1d = [], []
    non_fit = []
    qm_hessian = np.copy(qm.hessian)

    print("Calculating the MD hessian matrix elements...")
    full_md_hessian = calc_hessian(qm.coords, mol, inp)

    count = 0
    print("Fitting the MD hessian parameters to QM hessian values")
    for i in range(mol.n_atoms*3):
        for j in range(i+1):
            hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
            if all([h == 0 for h in hes]) or np.abs(qm_hessian[count]) < 1e+1:
                qm_hessian = np.delete(qm_hessian, count)
                full_md_hessian_1d.append(np.zeros(mol.n_terms))
            else:
                count += 1
                hessian.append(hes[:-1])
                full_md_hessian_1d.append(hes[:-1])
                non_fit.append(hes[-1])
    print("Done!\n")

    difference = qm_hessian - np.array(non_fit)
    # la.lstsq or nnls could also be used:
    fit = optimize.lsq_linear(hessian, difference, bounds=(0, np.inf)).x
    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    return fit, full_md_hessian_1d


def fit_dihedrals(inp, mol, qm):
    """
    Temporary - to be removed
    """

    from .fragment import check_one_fragment
    for atoms in mol.dih.flex.atoms:
        frag_name, _, _, _ = check_one_fragment(inp, mol, atoms)
        scan_dihedral(inp, mol, atoms, frag_name)
#        for atoms in mol.dih.flex.atoms:
#            frag_name, _, _ = check_one_fragment(inp, mol, atoms)
#            scan_dihedral(inp, mol, atoms, frag_name)


def calc_hessian(coords, mol, inp):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.n_atoms, 3*mol.n_atoms, mol.n_terms+1))

    for a in range(mol.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += 0.003
            f_plus = calc_forces(coords, mol, inp)
            coords[a][xyz] -= 0.006
            f_minus = calc_forces(coords, mol, inp)
            coords[a][xyz] += 0.003
            diff = - (f_plus - f_minus) / 0.006
            full_hessian[a*3 + xyz, :, :] = diff.reshape(3*mol.n_atoms,
                                                         mol.n_terms + 1)
    return full_hessian


def calc_forces(coords, mol, inp):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """
    force = np.zeros((mol.n_atoms, mol.n_terms+1, 3))

    for atoms, r0, term in zip(mol.bonds.atoms + mol.angles.urey.atoms,
                               mol.bonds.minima + mol.angles.urey.minima,
                               mol.bonds.term_ids + mol.angles.urey.term_ids):
        force = calc_bonds(coords, np.array(atoms), r0, term, force)

    for atoms, r0s, term in zip(mol.angles.cross.atoms,
                                mol.angles.cross.minima,
                                mol.angles.cross.term_ids):
        force = calc_cross_bondangle(coords, np.array(atoms), r0s, term, force)

    for atoms, t0, term in zip(mol.angles.atoms, mol.angles.minima,
                               mol.angles.term_ids):
        force = calc_angles(coords, np.array(atoms), t0, term, force)  # g96
    for atoms, p0, term in zip(mol.dih.rigid.atoms + mol.dih.imp.atoms,
                               mol.dih.rigid.minima + mol.dih.imp.minima,
                               mol.dih.rigid.term_ids + mol.dih.imp.term_ids):
        force = calc_imp_diheds(coords, np.array(atoms), p0, term, force)

    # for i, j, c6, c12, qq in mol.pair_list:
    #     force = calc_pairs(coords, i, j, c6, c12, qq, force)

    if not inp.nofrag:
        for atoms, params in zip(mol.dih.flex.atoms, mol.dih.flex.minima):
            force = calc_rb_diheds(coords, atoms, params, force)

    force = np.swapaxes(force, 1, 2)

    return force


def calc_vibrational_frequencies(upper, qm):
    """
    Calculate the MD vibrational frequencies by diagonalizing its Hessian
    """
    const_amu = 1.6605389210e-27
    const_avogadro = 6.0221412900e+23
    const_speedoflight = 299792.458
    kj2j = 1e3
    ang2meter = 1e-10
    to_omega2 = kj2j/ang2meter**2/(const_avogadro*const_amu)  # 1/s**2
    to_waveno = 1e-5/(2.0*np.pi*const_speedoflight)  # cm-1

    e = elements()
    matrix = np.zeros((3*qm.n_atoms, 3*qm.n_atoms))
    count = 0

    for i in range(3*qm.n_atoms):
        for j in range(i+1):
            mass_i = e.mass[qm.atomids[int(np.floor(i/3))]]
            mass_j = e.mass[qm.atomids[int(np.floor(j/3))]]
            matrix[i, j] = upper[count]/np.sqrt(mass_i*mass_j)
            matrix[j, i] = matrix[i, j]
            count += 1
    val, vec = eigh(matrix)
    vec = np.reshape(np.transpose(vec), (3*qm.n_atoms, qm.n_atoms, 3))[6:]

    for i in range(qm.n_atoms):
        vec[:, i, :] = vec[:, i, :] / np.sqrt(e.mass[qm.atomids[i]])

    freq = np.sqrt(val[6:] * to_omega2) * to_waveno
    return freq, vec


def write_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, inp):
    """
    Scope:
    ------
    Create the following files for comparing QM reference to the generated
    MD frequencies/eigenvalues.

    Output:
    ------
    JOBNAME_qforce.freq : QM vs MD vibrational frequencies and eigenvectors
    JOBNAME_qforce.nmd : MD eigenvectors that can be played in VMD with:
                                vmd -e filename
    """
    e = elements()
    freq_file = f"{inp.job_dir}/{inp.job_name}_qforce.freq"
    nmd_file = f"{inp.job_dir}/{inp.job_name}_qforce.nmd"

    with open(freq_file, "w") as f:
        f.write(" mode  QM-Freq   MD-Freq     Diff.  %Error\n")
        for i, (q, m) in enumerate(zip(qm_freq, md_freq)):
            diff = q - m
            err = diff / q * 100
            f.write(f"{i+7:>4}{q:>10.1f}{m:>10.1f}{diff:>10.1f}{err:>8.2f}\n")
        f.write("\n\n         QM vectors              MD Vectors\n")
        f.write(50*"=")
        for i, (qm1, md1) in enumerate(zip(qm_vec, md_vec)):
            f.write(f"\nMode {i+7}\n")
            for qm2, md2 in zip(qm1, md1):
                f.write("{:>8.3f}{:>8.3f}{:>8.3f}{:>10.3f}{:>8.3f}{:>8.3f}\n"
                        .format(*qm2, *md2))
    with open(nmd_file, "w") as nmd:
        nmd.write(f"nmwiz_load {inp.job_name}_qforce.nmd\n")
        nmd.write(f"title {inp.job_name}\n")
        nmd.write("names")
        for ids in qm.atomids:
            nmd.write(f" {e.sym[ids]}")
        nmd.write("\nresnames")
        for i in range(qm.n_atoms):
            nmd.write(" RES")
        nmd.write("\nresnums")
        for i in range(qm.n_atoms):
            nmd.write(" 1")
        nmd.write("\ncoordinates")
        for c in qm.coords:
            nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
        for i, m in enumerate(md_vec):
            nmd.write(f"\nmode {i+7}")
            for c in m:
                nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
    print(f"QM vs MD vibrational frequencies can be found in: {freq_file}")
    print(f"Vibrational modes (can be run in VMD) is located in: {nmd_file}\n")


def make_ff_params_from_fit(mol, fit, inp, qm, polar=False):
    """
    Scope:
    -----
    Convert units, average over equivalent minima and prepare everything
    to be written as a forcefield file.
    """
    ff = Forcefield()
    e = elements()
    bohr2nm = 0.052917721067
    ff.mol_type = inp.job_name
    ff.natom = mol.n_atoms
    ff.box = [10., 10., 10.]
    ff.n_mol = 1
    ff.coords = list(qm.coords/10)
    mass = [round(e.mass[i], 5) for i in qm.atomids]
    atom_no = range(1, mol.n_atoms + 1)
    atoms = []
    atom_dict = {}

    for i, a in enumerate(qm.atomids):
        sym = e.sym[a]
        if sym not in atom_dict:
            atom_dict[sym] = 1
        else:
            atom_dict[sym] += 1
        atoms.append(f'{sym}{atom_dict[sym]}')

    for i, (sigma, epsilon) in enumerate(zip(qm.sigma, qm.epsilon)):
        unique = mol.types[mol.list[i][0]]
        ff.atom_types.append([unique, 0, 0, "A", sigma*0.1, epsilon])

    for n, at, a_uniq, a, m in zip(atom_no, mol.types, mol.atoms, atoms, mass):
        ff.atoms.append([n, at, 1, "MOL", a, n, qm.q[a_uniq], m])

    if polar:
        alphas = qm.alpha*bohr2nm**3
        drude = {}
        n_drude = 1
        ff.atom_types.append(["DP", 0, 0, "S", 0, 0])

        for i, alpha in enumerate(alphas):
            if alpha > 0:
                drude[i] = mol.n_atoms+n_drude
                ff.atoms[i][6] += 8
                # drude atoms
                ff.atoms.append([drude[i], 'DP', 2, 'MOL', f'D{atoms[i]}',
                                 i+1, -8., 0.])
                ff.coords.append(ff.coords[i])
                # polarizability
                ff.polar.append([i+1, drude[i], 1, alpha])
                n_drude += 1
        ff.exclu = [[] for _ in ff.atoms]
        ff.natom = len(ff.atoms)
        for i, alpha in enumerate(alphas):
            if alpha > 0:
                # exclusions for balancing the drude particles
                for j in mol.neighbors[inp.nrexcl-2][i]+mol.neighbors[inp.nrexcl-1][i]:
                    if alphas[j] > 0:
                        ff.exclu[drude[i]-1].extend([drude[j]])
                for j in mol.neighbors[inp.nrexcl-1][i]:
                    ff.exclu[drude[i]-1].extend([j+1])
                ff.exclu[drude[i]-1].sort()
                # thole polarizability
                for neigh in [mol.neighbors[n][i] for n in range(inp.nrexcl)]:
                    for j in neigh:
                        if i < j and alphas[j] > 0:
                            ff.thole.append([i+1, drude[i], j+1, drude[j], "2",
                                             2.6, alpha, alphas[j]])

    for i, term in enumerate(mol.bonds.term_ids):
        atoms = [a+1 for a in mol.bonds.atoms[i]]
        param = fit[term] * 100
        equiv_terms = np.where(np.array(mol.bonds.term_ids) == term)
        minimum = np.array(mol.bonds.minima)[equiv_terms].mean() * 0.1
        ff.bonds.append(atoms + [1, minimum, param])

    for i, term in enumerate(mol.angles.term_ids):
        atoms = [a+1 for a in mol.angles.atoms[i]]
        param = fit[term]
        eq = np.where(np.array(mol.angles.term_ids) == term)
        minimum = np.degrees(np.array(mol.angles.minima)[eq].mean())
        ff.angles.append(atoms + [1, minimum, param])

    if inp.urey:
        for i, term in enumerate(mol.angles.urey.term_ids):
            param = fit[term] * 100
            eq = np.where(np.array(mol.angles.urey.term_ids) == term)
            minimum = np.array(mol.angles.urey.minima)[eq].mean() * 0.1
            ff.angles[i][3] = 5
            ff.angles[i].extend([minimum, param])

    for i, term in enumerate(mol.dih.rigid.term_ids):
        atoms = [a+1 for a in mol.dih.rigid.atoms[i]]
        param = fit[term]
#        eq = np.where(np.array(mol.dih.rigid.term_ids)==term)
#        minimum = np.degrees(np.array(mol.dih.rigid.minima)[eq].mean())
        minimum = np.degrees(mol.dih.rigid.minima[i])
        ff.dihedrals.append(atoms + [2, minimum, param])

    for i, term in enumerate(mol.dih.imp.term_ids):
        atoms = [a+1 for a in mol.dih.imp.atoms[i]]
        param = fit[term]
#        eq = np.where(np.array(mol.dih.imp.term_ids)==term)
#        minimum = np.degrees(np.array(mol.dih.imp.minima)[eq].mean())
        minimum = np.degrees(mol.dih.imp.minima[i])
        ff.impropers.append(atoms + [2, minimum, param])

    if inp.nofrag:
        for i, term in enumerate(mol.dih.flex.term_ids):
            atoms = [a+1 for a in mol.dih.flex.atoms[i]]
            ff.flexible.append(atoms + [3, term+1])
    else:
        for i, (term, params) in enumerate(zip(mol.dih.flex.term_ids,
                                               mol.dih.flex.minima)):
            atoms = [a+1 for a in mol.dih.flex.atoms[i]]
            ff.flexible.append(atoms + [3] + list(params))

    for i, term in enumerate(mol.dih.constr.term_ids):
        atoms = [a+1 for a in mol.dih.constr.atoms[i]]
        ff.constrained.append(atoms + [3, term+1])

    write_ff(ff, inp, polar)
    print("Q-Force force field parameters (.itp, .top) can be found in the "
          f"directory: {inp.job_dir}/")
