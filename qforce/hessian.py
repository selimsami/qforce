import scipy.optimize as optimize
import numpy as np


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
            if all([h == 0 for h in hes]) or np.abs(qm_hessian[count]) < 0.0001:
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
        ignores = ['dihedral/flexible']
    else:
        ignores = []

    force = np.zeros((mol.terms.n_fitted_terms+1, mol.topo.n_atoms, 3))

    with mol.terms.add_ignore(ignores):
        for term in mol.terms:
            term.do_fitting(coords, force)

    return force
