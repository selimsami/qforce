from scipy import optimize
import numpy as np
#
from .molecule.bond_and_angle_terms import MorseBondTerm


def fit_hessian(logger, config, mol, qm):
    hessian, full_md_hessian_1d = [], []
    non_fit = []
    qm_hessian = np.copy(qm.hessian)

    logger.info("Calculating the MD hessian matrix elements...")
    full_md_hessian = calc_hessian(qm.coords, mol)
    print("config = ", config.dihedral)

    count = 0

    logger.info("Fitting the MD hessian parameters to QM hessian values")

    min_arr = np.full(mol.terms.n_fitted_terms, -np.inf)
    max_arr = np.full(mol.terms.n_fitted_terms, np.inf)
    bnd_ndx = np.unique([term.idx for term in mol.terms['bond']])
    ang_ndx = np.unique([term.idx for term in mol.terms['angle']])
    # imp_ndx = np.unique([term.idx for term in mol.terms['dihedral/improper']])
    # aa_ndx = np.unique([term.idx for term in mol.terms['cross_angle_angle']])

    # if 'cross_bond_angle' in mol.terms and len(mol.terms['cross_bond_angle']) > 0:
    #     ndx = np.unique([term.idx for term in mol.terms['cross_bond_angle']])
    #     min_arr[ndx] = -500
    #     max_arr[ndx] = 500

    # name = 'dihedral/improper'
    # if name in mol.terms and len(mol.terms[name]) > 0:
    #     ndx = np.unique([term.idx for term in mol.terms[name]])
    #     min_arr[ndx] = -1000
    #     max_arr[ndx] = 1000

    min_arr[bnd_ndx] = 0
    if len(mol.terms['angle']) != 0:
        min_arr[ang_ndx] = 0
    # min_arr[aa_ndx] = 0
    # if imp_ndx:
    #     min_arr[imp_ndx] = 0

    for i in range(mol.topo.n_atoms*3):
        for j in range(i+1):
            hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
            if all(h == 0 for h in hes) or np.abs(qm_hessian[count]) < 0.0001:
                qm_hessian = np.delete(qm_hessian, count)
                full_md_hessian_1d.append(np.zeros(mol.terms.n_fitted_terms))
            else:
                count += 1
                hessian.append(hes[:-1])
                full_md_hessian_1d.append(hes[:-1])
                non_fit.append(hes[-1])

    difference = qm_hessian - np.array(non_fit)

    # la.lstsq or nnls could also be used:
    fit = optimize.lsq_linear(hessian, difference, bounds=(min_arr, max_arr)).x
    # fit = optimize.lsq_linear(hessian, difference, bounds=(-np.inf, np.inf)).x
    logger.info("Done!\n")

    for term in mol.terms:
        if term.idx < len(fit):
            term.fconst = fit[term.idx]
    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    err = full_md_hessian_1d-qm.hessian
    mae = np.abs(err).mean()
    rmse = (err**2).mean()**0.5
    max_err = np.max(np.abs(err))
    # print('mae:', mae*0.2390057361376673)
    # print('rmse:', rmse*0.2390057361376673)
    # print('max_err:', max_err*0.2390057361376673)

    average_unique_minima(mol.terms, config)

    return full_md_hessian_1d


def multi_hessian_fit(logger, config, mol, qms, qmens, qmgrads, fit_flexible=False):
    if len(qms) == 0:
        raise ValueError("at least one frequency needs to be computed!")

    full_hessian = []
    full_differences = []

    min_arr = np.full(mol.terms.n_fitted_terms, -np.inf)
    max_arr = np.full(mol.terms.n_fitted_terms, np.inf)
    bnd_ndx = np.unique([term.idx for term in mol.terms['bond']])
    ang_ndx = np.unique([term.idx for term in mol.terms['angle']])
    # imp_ndx = np.unique([term.idx for term in mol.terms['dihedral/improper']])
    # aa_ndx = np.unique([term.idx for term in mol.terms['cross_angle_angle']])

    # if 'cross_bond_angle' in mol.terms and len(mol.terms['cross_bond_angle']) > 0:
    #     ndx = np.unique([term.idx for term in mol.terms['cross_bond_angle']])
    #     min_arr[ndx] = -500
    #     max_arr[ndx] = 500

    # name = 'dihedral/improper'
    # if name in mol.terms and len(mol.terms[name]) > 0:
    #     ndx = np.unique([term.idx for term in mol.terms[name]])
    #     min_arr[ndx] = -1000
    #     max_arr[ndx] = 1000

    min_arr[bnd_ndx] = 0
    if len(mol.terms['angle']) != 0:
        min_arr[ang_ndx] = 0
    # min_arr[aa_ndx] = 0
    # if imp_ndx:
    #     min_arr[imp_ndx] = 0

    logger.info("Calculating the MM hessian matrix elements for all structures ...")
    for qm in qms:
        hessian, full_md_hessian_1d = [], []
        non_fit = []
        qm_hessian = np.copy(qm.hessian)

        full_md_hessian = calc_hessian(qm.coords, mol, fit_flexible=fit_flexible)
        count = 0
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
        #
        difference = qm_hessian - np.array(non_fit)
        full_differences += list(difference)
        full_hessian += hessian

    if fit_flexible is True:
        ignore_terms = ['charge_flux']
    else:
        ignore_terms = ['dihedral/flexible', 'charge_flux']

    energy_min, force_min = calc_forces(qm.coords, mol)

    w = 100

    force_min *= -1  # convert from force to gradient
    force_min = force_min.reshape(mol.terms.n_fitted_terms+1, mol.topo.n_atoms*3)
    full_hessian += list(w*force_min[:-1].T)
    full_differences += list(w*(np.zeros(qm.coords.size) - force_min[-1]))
    #

    weight_e = 100
    weight_f = 10


    with mol.terms.add_ignore(ignore_terms):
        for qmen in qmens:
            # Ax=B
            energy, _ = calc_forces(qmen.coords, mol)
            full_differences.append(qmen.energy - energy[-1])
            full_hessian.append(energy[:-1])

        full_qm_forces = []
        full_mm_forces = []
        full_qm_energies = []
        full_mm_energies = []

        for qmgrad in qmgrads:
            mm_energy, mm_force = calc_forces(qmgrad.coords, mol)
            mm_energy -= energy_min
            mm_force *= -1  # convert from force to gradient
            mm_force = mm_force.reshape(mol.terms.n_fitted_terms+1, mol.topo.n_atoms*3)

            full_qm_forces.append(qmgrad.gradient)
            full_mm_forces.append(mm_force[:-1].T)
            full_qm_energies.append(qmgrad.energy)
            full_mm_energies.append(mm_energy[:-1])

            full_hessian += list(weight_f*mm_force[:-1].T)
            full_differences += list(weight_f*(np.array(qmgrad.gradient).flatten() - mm_force[-1]))
            # full_hessian += list(qmgrad.weight*weight_f*mm_force[:-1].T)
            # full_differences += list(qmgrad.weight*weight_f*(np.array(qmgrad.gradient).flatten() - mm_force[-1]))
            #
            full_differences.append(weight_e*(qmgrad.energy - mm_energy[-1]))
            full_hessian.append(weight_e*mm_energy[:-1])
            # full_differences.append(qmgrad.weight*weight_e*(qmgrad.energy - mm_energy[-1]))
            # full_hessian.append(qmgrad.weight*weight_e*mm_energy[:-1])

    logger.info("Fitting the MD hessian parameters to QM hessian values")
    fit = optimize.lsq_linear(full_hessian, full_differences, bounds=(min_arr, max_arr)).x
    # fit = optimize.lsq_linear(hessian, difference, bounds=(-np.inf, np.inf)).x
    logger.info("Done!\n")

    with mol.terms.add_ignore(ignore_terms):
        for term in mol.terms:
            if term.idx < len(fit):
                term.set_fitparameters(fit)
    # TODO: is this correct? Check
    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    if len(qmgrads) > 0:
        full_qm_energies = np.array(full_qm_energies)
        print(full_qm_energies.shape)
        full_mm_energies = np.array(full_mm_energies)
        print(full_mm_energies.shape)
        full_qm_forces = np.array(full_qm_forces)
        print(full_qm_forces.shape)
        full_mm_forces = np.array(full_mm_forces)
        print(full_mm_forces.shape)
        full_mm_forces = np.sum(full_mm_forces * fit, axis=2)
        full_mm_energies = np.sum(full_mm_energies * fit, axis=1)

        for struct in range(len(qmgrads)):
            print('struct', struct)
            print('QM:\n', full_qm_energies[struct], '\n', full_qm_forces[struct])
            print('MM:\n', full_mm_energies[struct], '\n', full_mm_forces[struct].reshape((mol.n_atoms, 3)))
            print('\n')

    emin = np.sum(energy_min[:-1] * fit)

    print('EMIN', emin)



    err = full_md_hessian_1d-qm.hessian
    mae = np.abs(err).mean()
    rmse = (err**2).mean()**0.5
    max_err = np.max(np.abs(err))
    print('mae:', mae*0.2390057361376673)
    print('rmse:', rmse*0.2390057361376673)
    print('max_err:', max_err*0.2390057361376673)

    # do it
    average_unique_minima(mol.terms, config)

    return full_md_hessian_1d


def calc_hessian(coords, mol, fit_flexible=False):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms,
                             mol.terms.n_fitted_terms+1))

    if fit_flexible is True:
        ignore_terms = ['charge_flux']
    else:
        ignore_terms = ['dihedral/flexible', 'charge_flux']
    #
    with mol.terms.add_ignore(ignore_terms):
        for a in range(mol.topo.n_atoms):
            for xyz in range(3):
                coords[a][xyz] += 1e-5
                _, f_plus = calc_forces(coords, mol)
                coords[a][xyz] -= 2e-5
                _, f_minus = calc_forces(coords, mol)
                coords[a][xyz] += 1e-5
                diff = - (f_plus - f_minus) / 2e-5
                full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_terms+1, 3*mol.topo.n_atoms).T
    return full_hessian


def calc_forces(coords, mol):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """
    energies = np.zeros(mol.terms.n_fitted_terms+1)
    force = np.zeros((mol.terms.n_fitted_terms+1, mol.topo.n_atoms, 3))

    for term in mol.terms:
        term.do_fitting(coords, energies, force)

    return energies, force


def average_unique_minima(terms, config):
    unique_terms = {}
    averaged_terms = ['bond', 'angle']

    if 'dihedral/inversion' in terms:
        averaged_terms.append('dihedral/inversion')

    for name in [term_name for term_name in averaged_terms]:
        for term in terms[name]:
            if str(term) in unique_terms.keys():
                if isinstance(term, MorseBondTerm):
                    term.equ[0] = unique_terms[str(term)]
                else:
                    term.equ = unique_terms[str(term)]
            else:
                eq = np.where(np.array(list(oterm.idx for oterm in terms[name])) == term.idx)
                if isinstance(term, MorseBondTerm):
                    minimum = np.abs(np.array(list(oterm.equ[0] for oterm in terms[name]))[eq]).mean()
                    term.equ[0] = minimum
                else:
                    minimum = np.abs(np.array(list(oterm.equ for oterm in terms[name]))[eq]).mean()
                    term.equ = minimum

                unique_terms[str(term)] = minimum

    # For Urey, recalculate length based on the averaged bonds/angles
    if 'urey' in terms:
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
