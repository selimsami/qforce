from sklearn.linear_model import Ridge
import numpy as np
#
from .molecule.bond_and_angle_terms import MorseBondTerm


def multi_fit(logger, config, mol, structs):
    """
    Doing linear/Ridge regression with all energies, forces and hessian terms:
    Ax = B

    A: MM contributions to each e/f/h item by each fitted FF term (shape: n_items x n_fit_terms)
    B: Total QM e/f/h minus MM contributions from non-fitted terms (like non-bonded) (shape: n_items)

    """

    A = []
    B = []
    weights = []

    # Compute energies for the lowest energy structure, to subtract from other energies
    e_lowest, f_lowest = calc_forces(mol.qm_minimum_coords, mol)
    # f_lowest *= -1  # convert from force to gradient
    # f_lowest = f_lowest.reshape(mol.terms.n_fitted_terms+1, mol.topo.n_atoms*3)
    # qm_force = np.zeros((mol.n_atoms, 3))
    # A += list(f_lowest[:-1].T)
    # B += list(qm_force.flatten() - f_lowest[-1])
    # weights += [1e7]*qm_force.size
    # A.append(e_lowest[:-1] - e_lowest[:-1])
    # B.append(0 - e_lowest[-1] + e_lowest[-1])
    # weights.append(1e7)

    logger.info("Calculating the MM hessian matrix elements for all structures ...")
    for weight, qm in structs.hessitr():
        hessian, full_md_hessian_1d = [], []
        non_fit = []
        qm_hessian = np.copy(qm.hessian)

        full_md_hessian = calc_hessian(qm.coords, mol)
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

        difference = qm_hessian - np.array(non_fit)
        A += hessian
        B += list(difference)
        weights += [weight]*difference.size

    logger.info("Calculating energies/forces for all additional structures...")
    for weight_e, qmen in structs.enitr():
        mm_energy, _ = calc_forces(qmen.coords, mol)
        B.append(qmen.energy - mm_energy[-1] + e_lowest[-1])
        A.append(mm_energy[:-1] - e_lowest[:-1])
        weights.append(weight_e)

    full_qm_forces = []
    full_mm_forces = []
    full_qm_energies = []
    full_mm_energies = []

    # scale for energy structs, grad structs only have weight_f stored, but the ratio is constant!
    factor_e = structs.energy_weight / structs.gradient_weight

    from ase.units import kB as KB
    from ase.units import kJ as KJ
    from ase.units import mol as MOL

    t = 500
    kbt = KB / KJ * MOL * t
    e_avg = (3*mol.n_atoms-6)*kbt/2

    for weight_f, qmgrad in structs.graditr():
        scale_weight = min(np.exp((e_avg-qmgrad.energy)/e_avg), 1)
        print(qmgrad.energy, scale_weight)
        weight_f *= scale_weight

        weight_e = factor_e * weight_f
        mm_energy, mm_force = calc_forces(qmgrad.coords, mol)
        mm_force *= -1  # convert from force to gradient
        mm_force = mm_force.reshape(mol.terms.n_fitted_terms+1, mol.topo.n_atoms*3)

        full_qm_forces.append(qmgrad.gradient)
        full_mm_forces.append(mm_force[:-1].T)
        full_qm_energies.append(qmgrad.energy)
        full_mm_energies.append(mm_energy[:-1]-e_lowest[:-1])

        A += list(mm_force[:-1].T)
        B += list(qmgrad.gradient.flatten() - mm_force[-1])
        weights += [weight_f]*qmgrad.gradient.size

        A.append(mm_energy[:-1] - e_lowest[:-1])
        B.append(qmgrad.energy - mm_energy[-1] + e_lowest[-1])
        weights.append(weight_e)

    logger.info("Fitting the force field parameters...")
    reg = Ridge(alpha=1e-6, fit_intercept=False).fit(A, B, sample_weight=weights)
    fit = reg.coef_
    logger.info("Done!\n")

    with mol.terms.add_ignore('charge_flux'):
        for term in mol.terms:
            if term.idx < len(fit):
                term.set_fitparameters(fit)

    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    nqmgrads = sum(1 for _ in structs.graditr())

    if nqmgrads > 0:
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

        for struct in range(nqmgrads):
            print('struct', struct)
            print('QM:\n', full_qm_energies[struct], '\n', full_qm_forces[struct])
            print('MM:\n', full_mm_energies[struct], '\n', full_mm_forces[struct].reshape((mol.n_atoms, 3)))
            print('\n')

    err = full_md_hessian_1d-qm.hessian
    mae = np.abs(err).mean()
    rmse = (err**2).mean()**0.5
    max_err = np.max(np.abs(err))
    print('mae:', mae*0.2390057361376673)
    print('rmse:', rmse*0.2390057361376673)
    print('max_err:', max_err*0.2390057361376673)

    return full_md_hessian_1d


def calc_hessian(coords, mol):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms,
                             mol.terms.n_fitted_terms+1))

    with mol.terms.add_ignore('charge_flux'):
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
