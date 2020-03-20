import scipy.optimize as optimize
import numpy as np
from .read_qm_out import QM
from .forcefield import ForceField
from .dihedral_scan import scan_dihedral
from .molecule import Molecule
from .fragment import fragment
from .frequencies import calc_qm_vs_md_frequencies


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

    qm = QM(inp, "freq", fchk_file=inp.fchk_file, out_file=inp.qm_freq_out)

    mol = Molecule(inp, qm)

    fit_results, md_hessian = fit_hessian(inp, mol, qm, ignore_flex=True)
    average_unique_minima(mol.terms)

    if inp.fragment:
        fragments = fragment(inp, mol, qm)

    calc_qm_vs_md_frequencies(inp, qm, md_hessian)
    ff = ForceField(inp, mol, qm.coords, inp.job_dir)
    ff.write_gromacs(inp, mol)

    print(f'Q-Forcefield parameters (.itp, .top) can be found in the directory: {inp.job_dir}\n')

    # temporary
    # if inp.fragment:
    #     fit_dihedrals(inp, mol, qm, fragments)


def fit_hessian(inp, mol, qm, ignore_flex=True):
    hessian, full_md_hessian_1d = [], []
    non_fit = []
    qm_hessian = np.copy(qm.hessian)

    print("Calculating the MD hessian matrix elements...")
    full_md_hessian = calc_hessian(qm.coords, mol, inp, ignore_flex)

    count = 0
    print("Fitting the MD hessian parameters to QM hessian values")
    for i in range(mol.topo.n_atoms*3):
        for j in range(i+1):
            hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
            if all([h == 0 for h in hes]) or np.abs(qm_hessian[count]) < 1e+1:
                qm_hessian = np.delete(qm_hessian, count)
                full_md_hessian_1d.append(np.zeros(mol.terms.n_fitted_terms))
            else:
                count += 1
                hessian.append(hes[:-1])
                full_md_hessian_1d.append(hes[:-1])
                non_fit.append(hes[-1])

    difference = qm_hessian - np.array(non_fit)
    # la.lstsq or nnls could also be used:
    fit = optimize.lsq_linear(hessian, difference, bounds=(0, np.inf)).x
    print("Done!\n")

    for term in mol.terms:
        if term.idx < len(fit):
            term.fconst = fit[term.idx]

    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    return fit, full_md_hessian_1d


def fit_dihedrals(inp, mol, qm, fragments):
    """
    Temporary - to be removed
    """

    for frag in fragments:
        term = list(mol.terms.get_terms_from_name(frag.name))[0]
        scan_dihedral(inp, term.atomids, frag.id, fit=False)


def calc_hessian(coords, mol, inp, ignore_flex):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms,
                             mol.terms.n_fitted_terms+1))

    for a in range(mol.topo.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += 0.003
            f_plus = calc_forces(coords, mol, inp, ignore_flex)
            coords[a][xyz] -= 0.006
            f_minus = calc_forces(coords, mol, inp, ignore_flex)
            coords[a][xyz] += 0.003
            diff = - (f_plus - f_minus) / 0.006
            full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_terms+1, 3*mol.topo.n_atoms).T
    return full_hessian


def calc_forces(coords, mol, inp, ignore_flex):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """
    if ignore_flex:
        ignores = ['dihedral/flexible', 'dihedral/constr']
    else:
        ignores = []

    force = np.zeros((mol.terms.n_fitted_terms+1, mol.topo.n_atoms, 3))

    with mol.terms.add_ignore(ignores):
        for term in mol.terms:
            term.do_fitting(coords, force)

    return force


def average_unique_minima(terms):
    unique_terms = {}
    averaged_terms = ['bond', 'angle']
    for name in [term_name for term_name in averaged_terms if term_name in terms.term_names]:
        for term in terms[name]:
            if str(term) in unique_terms.keys():
                term.equ = unique_terms[str(term)]
            else:
                eq = np.where(np.array(list(oterm.idx for oterm in terms[name])) == term.idx)
                minimum = np.array(list(oterm.equ for oterm in terms[name]))[eq].mean()
                term.equ = minimum
                unique_terms[str(term)] = minimum

    # For Urey, recalculate length based on the averaged bonds/angles
    for term in terms['urey']:
        if str(term) in unique_terms.keys():
            term.equ = unique_terms[str(term)]
        else:
            bond1_atoms = sorted(term.atomids[:2])
            bond2_atoms = sorted(term.atomids[1:])
            bond1 = [bond.equ for bond in terms['bond'] if all(bond1_atoms == bond.atomids)][0]
            bond2 = [bond.equ for bond in terms['bond'] if all(bond2_atoms == bond.atomids)][0]
            angle = [ang.equ for ang in terms['angle'] if all(term.atomids == ang.atomids)][0]
            urey = (bond1**2 + bond2**2 - 2*bond1*bond2*np.cos(angle))**0.5
            term.equ = urey
            unique_terms[str(term)] = urey
