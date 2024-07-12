import numpy as np
import scipy.optimize as optimize


def fit_charge_flux(qm_out, qm_energy_out, qm_gradient_out, mol, weight_dipole=0.1):
    print(qm_out.coords)
    print('TOTAL DIP', calc_dipole(qm_out.coords, mol))

    differences, cf_contrib = calc_dipole_derivative(qm_out.dipole_deriv, qm_out.coords.copy(), mol.non_bonded.q, mol)

    if len(qm_energy_out+qm_gradient_out) > 0:
        frame_diff, frame_cf_contrib = calc_additional_frames(qm_energy_out, qm_gradient_out, mol, weight_dipole)
        differences = np.append(differences, frame_diff)
        cf_contrib = np.append(cf_contrib, frame_cf_contrib, axis=0)

    fit = optimize.lsq_linear(cf_contrib, differences)
    print(fit)
    for term in mol.terms['charge_flux']:
        term.fconst = fit.x[term.flux_idx]
    dq_mm_dip_der = np.sum(cf_contrib * fit.x, axis=1)

    err = dq_mm_dip_der-differences
    mae = np.abs(err).mean()
    rmse = (err**2).mean()**0.5
    max_err = np.max(np.abs(err))
    print('mae:', mae)
    print('rmse:', rmse)
    print('max_err:', max_err)


def calc_additional_frames(qm_energy_out, qm_gradient_out, mol, weight_dipole):
    difference, dq_dipole = [], []
    for out in qm_energy_out+qm_gradient_out:
        mm_dip = calc_dipole(out.coords, mol)
        dq_mm_dip = fit_charge_flux_terms(out.coords, mol)
        mm_dip_cf = np.array([calc_dipole_from_charge(out.coords, q) for q in dq_mm_dip.T])
        diff = out.dipole - mm_dip
        difference.append(diff.flatten()*weight_dipole)
        dq_dipole.append(mm_dip_cf.T*weight_dipole)

    difference = np.array(difference).flatten()
    dq_dipole = np.array(dq_dipole).reshape((-1, mol.terms.n_fitted_flux_terms))
    return difference, dq_dipole


def calc_dipole_derivative(qm_dipole_deriv, coords, charge, mol, disp = 1e-5):
    dipole_derivative = np.zeros((mol.topo.n_atoms, 3, 3))
    dq_dipole_derivative = np.zeros((mol.topo.n_atoms, 3, 3, mol.terms.n_fitted_flux_terms))

    for a in range(mol.topo.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += disp
            dd_plus = calc_dipole(coords, mol)
            dq_plus = fit_charge_flux_terms(coords, mol)
            dq_dd_plus = calc_dipole_from_charge(coords[:, :, np.newaxis], dq_plus)

            coords[a][xyz] -= 2*disp
            dd_minus = calc_dipole(coords, mol)
            dq_minus = fit_charge_flux_terms(coords, mol)
            dq_dd_minus = calc_dipole_from_charge(coords[:, :, np.newaxis], dq_minus)

            coords[a][xyz] += disp
            dipole_derivative[a, xyz] = (dd_plus - dd_minus) / (2*disp)
            dq_dipole_derivative[a, xyz] = (dq_dd_plus - dq_dd_minus) / (2*disp)

    difference = qm_dipole_deriv - dipole_derivative
    difference = difference.flatten()
    dq_dipole_derivative = dq_dipole_derivative.reshape((-1, mol.terms.n_fitted_flux_terms))

    return difference, dq_dipole_derivative


def calc_dipole(coords, mol):
    mu_from_charges = (coords*mol.non_bonded.q[:, np.newaxis]).sum(axis=0)
    mu_from_atomic_dipoles = compute_atomic_dipoles(mol, coords)
    return mu_from_charges+mu_from_atomic_dipoles


def calc_dipole_from_charge(coords, charges):
    return (coords*charges[:, np.newaxis]).sum(axis=0)


def fit_charge_flux_terms(coords, mol):
    q_flux = np.zeros((mol.terms.n_fitted_flux_terms, mol.topo.n_atoms))
    for term in mol.terms['charge_flux']:
        term.do_flux_fitting(coords, q_flux)
    return q_flux.T


def compute_charge_flux_terms(coords, mol):
    q_flux = np.zeros(mol.topo.n_atoms)
    for term in mol.terms['charge_flux']:
        term.do_flux(coords, q_flux)

    return q_flux


def compute_atomic_dipoles(mol, coords):
    atomic_dipoles = np.zeros(3)
    for term in mol.terms['local_frame']:
        atomic_dipoles += term.convert_multipoles_to_cartesian_frame(coords)[0]
    return atomic_dipoles
