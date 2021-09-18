import scipy.optimize as optimize
import numpy as np


def fit_hessian(config, mol, qm, n_iter):
    print('Running fit_hessian')
    hessian, full_md_hessian_1d = [], []
    non_fit = []
    qm_hessian = np.copy(qm.hessian)

    print("Calculating the MD hessian matrix elements...")
    full_md_hessian = calc_hessian(qm.coords, mol)

    count = 0
    print("Fitting the MD hessian parameters to QM hessian values")
    for i in range(mol.topo.n_atoms*3):
        for j in range(i+1):
            hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
            if all([h == 0 for h in hes]) or np.abs(qm_hessian[count]) < 0.0001:
                qm_hessian = np.delete(qm_hessian, count)
                # full_md_hessian_1d.append(np.zeros(mol.terms.n_fitted_terms))
                full_md_hessian_1d.append(np.zeros(mol.terms.n_fitted_params))
            else:
                count += 1
                hessian.append(hes[:-1])
                full_md_hessian_1d.append(hes[:-1])
                non_fit.append(hes[-1])

    difference = qm_hessian - np.array(non_fit)
    # la.lstsq or nnls could also be used:
    print(f'Running optimizer for up to {n_iter} iterations...')
    result = optimize.lsq_linear(hessian, difference, bounds=(0, np.inf), max_iter=n_iter, verbose=1)
    print(f'It ran for {result.nit} iterations')
    fit = result.x
    print("Done!\n")

    print('Assigning constants to terms...')
    print(f'len(fit) = {len(fit)}')

    sorted_terms = sorted(mol.terms, key=lambda trm: trm.idx)
    seen_idx = [0]  # Have 0 as seen to avoid unnecessary shifting
    # for term in mol.terms:
    index = 0
    for i, term in enumerate(sorted_terms):
        # if term.idx < len(fit):
        if term.idx < mol.terms.n_fitted_terms:
            # term.fconst = np.array([fit[term.idx]])
            if term.idx not in seen_idx:  # Since 0 is seen by default, this won't be True in the first iteration
                index += sorted_terms[i-1].n_params  # Increase index by the number of parameters of the previous term
                seen_idx.append(term.idx)
            term.fconst = fit[index:index+term.n_params]
            print(f'Term {term} with idx {term.idx} has fconst {term.fconst}')
    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    average_unique_minima(mol.terms, config)

    print('Finished fit_hessian')
    return full_md_hessian_1d


def calc_hessian(coords, mol):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    # full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms, mol.terms.n_fitted_terms+1))
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms, mol.terms.n_fitted_params+1))

    for a in range(mol.topo.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += 0.003
            f_plus = calc_forces(coords, mol)
            coords[a][xyz] -= 0.006
            f_minus = calc_forces(coords, mol)
            coords[a][xyz] += 0.003
            diff = - (f_plus - f_minus) / 0.006
            # full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_terms+1, 3*mol.topo.n_atoms).T
            full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_params+1, 3*mol.topo.n_atoms).T
    return full_hessian


def calc_forces(coords, mol):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """

    # force = np.zeros((mol.terms.n_fitted_terms + 1, mol.topo.n_atoms, 3))
    force = np.zeros((mol.terms.n_fitted_params+1, mol.topo.n_atoms, 3))

    with mol.terms.add_ignore(['dihedral/flexible']):
        sorted_terms = sorted(mol.terms, key=lambda trm: trm.idx)
        seen_idx = [0]  # Have 0 as seen to avoid unnecessary shifting
        # for term in mol.terms:
        index = 0
        for i, term in enumerate(sorted_terms):
            if term.idx not in seen_idx:  # Since 0 is seen by default, this won't be True in the first iteration
                index += sorted_terms[i-1].n_params  # Increase index by the number of parameters of the previous term
                seen_idx.append(term.idx)
            term.do_fitting(coords, force, index)

    return force


def average_unique_minima(terms, config):
    print('Entering average_unique_minima')
    # print('Terms:')
    # print(terms)
    unique_terms = {}
    trms = terms.term_names
    # averaged_terms = ['bond', 'angle', 'dihedral/inversion']
    averaged_terms = [x for x in ['bond', 'morse', 'morse_mp', 'angle', 'dihedral/inversion'] if config.__dict__[x]]
    print(f'Averaged terms: {averaged_terms}')
    for name in [term_name for term_name in averaged_terms]:
        for term in terms[name]:
            if str(term) in unique_terms.keys():
                term.equ = unique_terms[str(term)]
            else:
                eq = np.where(np.array(list(oterm.idx for oterm in terms[name])) == term.idx)
                minimum = np.abs(np.array(list(oterm.equ for oterm in terms[name]))[eq]).mean()
                term.equ = minimum
                unique_terms[str(term)] = minimum

    # For Urey, recalculate length based on the averaged bonds/angles
    if config.urey:
        for term in terms['urey']:
            if str(term) in unique_terms.keys():
                term.equ = unique_terms[str(term)]
            else:
                bond1_atoms = sorted(term.atomids[:2])
                bond2_atoms = sorted(term.atomids[1:])
                if config.morse:  # Morse bond
                    bond1 = [bond.equ for bond in terms['morse'] if all(bond1_atoms == bond.atomids)][0]
                    bond2 = [bond.equ for bond in terms['morse'] if all(bond2_atoms == bond.atomids)][0]
                    angle = [ang.equ for ang in terms['angle'] if all(term.atomids == ang.atomids)][0]
                    urey = (bond1**2 + bond2**2 - 2*bond1*bond2*np.cos(angle))**0.5
                    term.equ = urey
                    unique_terms[str(term)] = urey
                elif config.morse_mp:  # Morse multi-parameter bond
                    bond1 = [bond.equ for bond in terms['morse_mp'] if all(bond1_atoms == bond.atomids)][0]
                    bond2 = [bond.equ for bond in terms['morse_mp'] if all(bond2_atoms == bond.atomids)][0]
                    angle = [ang.equ for ang in terms['angle'] if all(term.atomids == ang.atomids)][0]
                    urey = (bond1**2 + bond2**2 - 2*bond1*bond2*np.cos(angle))**0.5
                    term.equ = urey
                    unique_terms[str(term)] = urey
                else:  # Regular bond
                    bond1 = [bond.equ for bond in terms['bond'] if all(bond1_atoms == bond.atomids)][0]
                    bond2 = [bond.equ for bond in terms['bond'] if all(bond2_atoms == bond.atomids)][0]
                    angle = [ang.equ for ang in terms['angle'] if all(term.atomids == ang.atomids)][0]
                    urey = (bond1 ** 2 + bond2 ** 2 - 2 * bond1 * bond2 * np.cos(angle)) ** 0.5
                    term.equ = urey
                    unique_terms[str(term)] = urey

    print('Leaving average_unique_minima')
