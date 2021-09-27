import json

import scipy.optimize as optimize
import numpy as np
import noisyopt as nopt

from .misc import check_continue


def nllsqfuncfloat(params, qm, qm_hessian, mol, sorted_terms, loss=None):
    return 0.5 * np.sum(nllsqfunc(params, qm, qm_hessian, mol, sorted_terms, loss)**2)


def nllsqfunc(params, qm, qm_hessian, mol, sorted_terms, loss=None):  # Residual function to minimize
    hessian = []
    non_fit = []
    # print("Calculating the MD hessian matrix elements...")
    full_md_hessian = calc_hessian_nl(qm.coords, mol, params)

    # print("Fitting the MD hessian parameters to QM hessian values")
    for i in range(mol.topo.n_atoms * 3):
        for j in range(i + 1):
            hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
            hessian.append(hes[:-1])
            non_fit.append(hes[-1])

    hessian = np.array(hessian)
    agg_hessian = np.sum(hessian, axis=1)  # Aggregate contribution of terms
    difference = qm_hessian - np.array(non_fit)

    # Compute residual vector
    residual = agg_hessian - difference

    # Append loss to history
    if loss is not None:
        loss.append(0.5 * np.sum(residual**2))

    return residual


def read_input_params(pinput, sorted_terms, mol):
    in_file = open(pinput)
    dct = json.load(in_file)
    in_file.close()
    seen_idx = [0]  # Have 0 as seen to avoid unnecessary shifting
    # for term in mol.terms:
    index = 0
    for i, term in enumerate(sorted_terms):
        if term.idx < mol.terms.n_fitted_terms:
            if term.idx not in seen_idx:  # Since 0 is seen by default, this won't be True in the first iteration
                index += sorted_terms[i - 1].n_params  # Increase index by the number of parameters of the previous term
                seen_idx.append(term.idx)
            x0[index:index + term.n_params] = np.array(dct[str(term)])


def write_opt_params(psave, sorted_terms, mol):
    with open(psave, 'w') as f:
        dct = {}
        for term in sorted_terms:
            if term.idx < mol.terms.n_fitted_terms:
                dct[str(term)] = term.fconst.tolist()
        json.dump(dct, f, indent=4)


def fit_hessian_nl(config, mol, qm, pinput, psave, process_file):
    print('Running fit_hessian_nl')

    qm_hessian = np.copy(qm.hessian)

    if config.opt.average == 'before':
        print('Running average_unique_minima before optimization...')
        average_unique_minima(mol.terms, config)

    print('Running non-linear optimizer')
    # Sort mol terms
    sorted_terms = sorted(mol.terms, key=lambda trm: trm.idx)

    # Create initial conditions for optimization
    x0 = 400 * np.ones(mol.terms.n_fitted_params)  # Initial values for term params
    # Read them from pinput if given
    if pinput is not None:
        print(f'Reading input params from {pinput}...')
        read_input_params(pinput, sorted_terms, mol)

    # Add noise if necessary
    # np.random.seed(0)
    print(f'Adding up to {config.opt.noise*100}% noise to the initial conditions')
    for i, ele in enumerate(x0):
        x0[i] += config.opt.noise * ele * (2 * np.random.random() - 1)
    print(f'x0: {x0}')

    # Get function value for x0 and ask the user if continue
    value = 0.5 * np.sum(nllsqfunc(x0, qm, qm_hessian, mol, sorted_terms)**2)
    print(f'Initial loss: {value}')
    check_continue(config)

    # Optimize
    result = None
    loss = []
    print(f'verbose = {config.opt.verbose}')
    args = (qm, qm_hessian, mol, sorted_terms, loss)
    if config.opt.nonlin_alg == 'trf':
        print('Running trf optimizer...')
        result = optimize.least_squares(nllsqfunc, x0, args=args, bounds=(0, np.inf), method='trf',
                                        ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=config.opt.verbose)
    elif config.opt.nonlin_alg == 'lm':  # VERY CAREFUL WITH THIS ONE, SINCE NO BOUNDS ARE SUPPORTED
        print('Running lm optimizer...')
        result = optimize.least_squares(nllsqfunc, x0, args=args, method='lm',
                                        ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=config.opt.verbose)
    elif config.opt.nonlin_alg == 'compass':
        print('Running compass optimizer...')
        disp = False if config.opt.verbose == 0 else True
        bounds = [(0, np.inf)] * x0.shape[0]
        result = nopt.minimizeCompass(nllsqfuncfloat, x0, args=args, bounds=bounds, disp=disp,
                                      paired=False)
    elif config.opt.nonlin_alg == 'bh':
        print('Running bh optimizer...')
        disp = False if config.opt.verbose == 0 else True
        bounds = [(0, np.inf)] * x0.shape[0]
        min_kwargs = {'args': args, 'bounds': bounds}
        result = optimize.basinhopping(nllsqfuncfloat, x0, minimizer_kwargs=min_kwargs,
                                       niter=config.opt.iter, disp=disp)

    fit = result.x
    if process_file is not None:
        np.savetxt(process_file, loss)

    print('Assigning constants to terms...')
    print(f'len(fit) = {len(fit)}')

    seen_idx = [0]  # Have 0 as seen to avoid unnecessary shifting
    index = 0
    for i, term in enumerate(sorted_terms):
        # if term.idx < len(fit):
        if term.idx < mol.terms.n_fitted_terms:
            if term.idx not in seen_idx:  # Since 0 is seen by default, this won't be True in the first iteration
                index += sorted_terms[i-1].n_params  # Increase index by the number of parameters of the previous term
                seen_idx.append(term.idx)
            term.fconst = fit[index:index + term.n_params]
            print(f'Term {term} with idx {term.idx} has fconst {term.fconst}')

    # If psave, write fit to .json
    if psave is not None:
        print(f'Writing fit to {psave}...')
        write_opt_params(psave, sorted_terms, mol)

    if config.opt.average == 'after':
        print('Running average_unique_minima after optimization but before hessian computation...')
        average_unique_minima(mol.terms, config)

    # Calculate final full_md_hessian_1d
    full_md_hessian = calc_hessian_nl(qm.coords, mol, fit)
    full_md_hessian_1d = []
    for i in range(mol.topo.n_atoms * 3):
        for j in range(i + 1):
            hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
            full_md_hessian_1d.append(hes[:-1])

    full_md_hessian_1d = np.array(full_md_hessian_1d)
    full_md_hessian_1d = np.sum(full_md_hessian_1d, axis=1)  # Aggregate contribution of terms

    if config.opt.average == 'last':
        print('Running average_unique_minima after optimization and hessian computation...')
        average_unique_minima(mol.terms, config)

    print('Finished fit_hessian_nl')
    return full_md_hessian_1d


def fit_hessian(config, mol, qm):
    print('Running fit_hessian')

    if config.opt.average == 'before':
        print('Running average_unique_minima before optimization...')
        average_unique_minima(mol.terms, config)

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
                full_md_hessian_1d.append(np.zeros(mol.terms.n_fitted_terms))
            else:
                count += 1
                hessian.append(hes[:-1])
                full_md_hessian_1d.append(hes[:-1])
                non_fit.append(hes[-1])

    difference = qm_hessian - np.array(non_fit)
    # la.lstsq or nnls could also be used:
    print(f'Running optimizer for up to {config.opt.iter} iterations...')
    result = optimize.lsq_linear(hessian, difference, bounds=(0, np.inf),
                                 max_iter=config.opt.iter, verbose=config.opt.verbose)
    fit = result.x
    print("Done!\n")

    print('Assigning constants to terms...')
    print(f'len(fit) = {len(fit)}')

    for term in mol.terms:
        if term.idx < len(fit):
            term.fconst = np.array([fit[term.idx]])
            print(f'Term {term} with idx {term.idx} has fconst {term.fconst}')

    if config.opt.average == 'after':
        print('Running average_unique_minima after optimization but before hessian computation...')
        average_unique_minima(mol.terms, config)
        # Recalculate full_md_hessian_1d
        full_md_hessian = calc_hessian(qm.coords, mol)
        qm_hessian = np.copy(qm.hessian)
        full_md_hessian_1d = []
        count = 0
        for i in range(mol.topo.n_atoms * 3):
            for j in range(i + 1):
                hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
                if all([h == 0 for h in hes]) or np.abs(qm_hessian[count]) < 0.0001:
                    qm_hessian = np.delete(qm_hessian, count)
                    full_md_hessian_1d.append(np.zeros(mol.terms.n_fitted_terms))
                else:
                    count += 1
                    full_md_hessian_1d.append(hes[:-1])

    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    if config.opt.average == 'last':
        print('Running average_unique_minima after optimization and hessian computation...')
        average_unique_minima(mol.terms, config)

    print('Finished fit_hessian')
    return full_md_hessian_1d


def calc_hessian(coords, mol):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms, mol.terms.n_fitted_terms+1))

    for a in range(mol.topo.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += 0.003
            f_plus = calc_forces(coords, mol)
            coords[a][xyz] -= 0.006
            f_minus = calc_forces(coords, mol)
            coords[a][xyz] += 0.003
            diff = - (f_plus - f_minus) / 0.006
            full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_terms+1, 3*mol.topo.n_atoms).T
    return full_hessian


def calc_hessian_nl(coords, mol, params):
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms, mol.terms.n_fitted_terms+1))

    for a in range(mol.topo.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += 0.003
            f_plus = calc_forces_nl(coords, mol, params)
            coords[a][xyz] -= 0.006
            f_minus = calc_forces_nl(coords, mol, params)
            coords[a][xyz] += 0.003
            diff = - (f_plus - f_minus) / 0.006
            full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_terms+1, 3*mol.topo.n_atoms).T
    return full_hessian


def calc_forces(coords, mol):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """

    force = np.zeros((mol.terms.n_fitted_terms+1, mol.topo.n_atoms, 3))

    with mol.terms.add_ignore(['dihedral/flexible']):
        for term in mol.terms:
            term.do_fitting(coords, force)

    return force


def calc_forces_nl(coords, mol, params):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """

    force = np.zeros((mol.terms.n_fitted_terms+1, mol.topo.n_atoms, 3))

    with mol.terms.add_ignore(['dihedral/flexible']):
        sorted_terms = sorted(mol.terms, key=lambda trm: trm.idx)
        seen_idx = [0]  # Have 0 as seen to avoid unnecessary shifting
        index = 0
        for i, term in enumerate(sorted_terms):
            if term.idx not in seen_idx:  # Since 0 is seen by default, this won't be True in the first iteration
                index += sorted_terms[i-1].n_params  # Increase index by the number of parameters of the previous term
                seen_idx.append(term.idx)
            term.do_fitting(coords, force, index, params)

    return force


def average_unique_minima(terms, config):
    print('Entering average_unique_minima')
    unique_terms = {}
    trms = ['bond', 'angle', 'morse', 'morse_mp', 'morse_mp2', 'dihedral/inversion']
    averaged_terms = [x for x in trms if config.terms.__dict__[x]]
    print(f'Averaged terms: {averaged_terms}')
    for name in averaged_terms:
        for term in terms[name]:
            if str(term) in unique_terms.keys():
                term.equ = unique_terms[str(term)]
            else:
                eq = np.where(np.array(list(oterm.idx for oterm in terms[name])) == term.idx)
                minimum = np.abs(np.array(list(oterm.equ for oterm in terms[name]))[eq]).mean()
                term.equ = minimum
                unique_terms[str(term)] = minimum

    # For Urey, recalculate length based on the averaged bonds/angles
    if config.terms.urey:
        for term in terms['urey']:
            if str(term) in unique_terms.keys():
                term.equ = unique_terms[str(term)]
            else:
                bond1_atoms = sorted(term.atomids[:2])
                bond2_atoms = sorted(term.atomids[1:])
                terms_bond_str = _get_terms_bond_str(config)
                bond1 = [bond.equ for bond in terms[terms_bond_str] if all(bond1_atoms == bond.atomids)][0]
                bond2 = [bond.equ for bond in terms[terms_bond_str] if all(bond2_atoms == bond.atomids)][0]
                angle = [ang.equ for ang in terms['angle'] if all(term.atomids == ang.atomids)][0]
                # urey = (bond1**2 + bond2**2 - 2*bond1*bond2*np.cos(angle))**0.5
                urey = np.sqrt(bond1 ** 2 + bond2 ** 2 - 2 * bond1 * bond2 * np.cos(angle))
                term.equ = urey
                unique_terms[str(term)] = urey

    print('Leaving average_unique_minima')


def _get_terms_bond_str(config) -> str:
    if config.terms.morse:
        return 'morse'
    elif config.terms.morse_mp:
        return 'morse_mp'
    elif config.terms.morse_mp2:
        return 'morse_mp2'
    else:  # regular bond
        return 'bond'
