import numpy as np
import scipy.optimize as optimize


def fit_dipole_derivative(qm_out, mol):
    """
    Scope:
    -----

    """

    dip_der, dq_dip_der = calc_dipole_derivative(qm_out.coords.copy(), mol.non_bonded.q, mol)

    difference = qm_out.dipole_deriv - dip_der
    difference = difference.flatten()
    dq_dip_der = dq_dip_der.reshape((-1, mol.terms.n_fitted_flux_terms))

    fit = optimize.lsq_linear(dq_dip_der, difference)
    print(fit)

    for term in mol.terms['charge_flux']:
        term.fconst = fit.x[term.flux_idx]
    dq_mm_dip_der = np.sum(dq_dip_der * fit.x, axis=1)

    err = dq_mm_dip_der-difference
    mae = np.abs(err).mean()
    rmse = (err**2).mean()**0.5
    max_err = np.max(np.abs(err))
    print('mae:', mae)
    print('rmse:', rmse)
    print('max_err:', max_err)

    #         full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_terms+1, 3*mol.topo.n_atoms).T
    # return full_hessian


def calc_dipole_derivative(coords, charge, mol, disp = 1e-5):
    dipole_derivative = np.zeros((mol.topo.n_atoms, 3, 3))
    dq_dipole_derivative = np.zeros((mol.topo.n_atoms, 3, 3, mol.terms.n_fitted_flux_terms))

    for a in range(mol.topo.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += disp
            dd_plus = calc_dipole(coords, charge)
            dq_plus = fit_charge_flux(coords, mol)
            dq_dd_plus = calc_dipole(coords[:, :, np.newaxis], dq_plus)

            coords[a][xyz] -= 2*disp
            dd_minus = calc_dipole(coords, charge)
            dq_minus = fit_charge_flux(coords, mol)
            dq_dd_minus = calc_dipole(coords[:, :, np.newaxis], dq_minus)

            coords[a][xyz] += disp
            dipole_derivative[a, xyz] = (dd_plus - dd_minus) / (2*disp)
            dq_dipole_derivative[a, xyz] = (dq_dd_plus - dq_dd_minus) / (2*disp)

    return dipole_derivative, dq_dipole_derivative


def calc_dipole(coords, charges):
    return (coords*charges[:, np.newaxis]).sum(axis=0)


def fit_charge_flux(coords, mol):
    q_flux = np.zeros((mol.terms.n_fitted_flux_terms, mol.topo.n_atoms))

    for term in mol.terms['charge_flux']:
        term.do_flux_fitting(coords, q_flux)

    return q_flux.T


def compute_charge_flux(coords, mol):
    q_flux = np.zeros(mol.topo.n_atoms)

    for term in mol.terms['charge_flux']:
        term.do_flux(coords, q_flux)

    return q_flux

#
#     j_b = params[0]
#     j_a = params[1]
#     j_bb = params[2]
# #     j_ba = params[3]
#
#     angle, bond1, bond2 = get_angle(coords)
#
# #     hh_dist = get_dist(coords[2], coords[1])[1]
#
#     # dangle = np.cos(angle)-np.cos(np.radians(a_eq))
#     dangle =  angle - np.radians(a_eq)
#
#     dbond1 = bond1 - b_eq
#     dbond2 = bond2 - b_eq
# #     dhh = hh_dist - hh_eq
#
#     dq_o, dq_h1, dq_h2 = 0, 0, 0
#
#     dq_h1 += j_b * dbond1
#     dq_h2 += j_b * dbond2
#
#     dq_h1 += j_bb * dbond2
#     dq_h2 += j_bb * dbond1
#
#     dq_h1 += j_a * dangle
#     dq_h2 += j_a * dangle
#
# #     dq_h1 += j_ba * dbond1 * dhh
# #     dq_h2 += j_ba * dbond2 * dhh
#
#     dq_o -= dq_h1 + dq_h2
#
#     return np.array([q_o+dq_o, q_h+dq_h1, q_h+dq_h2])

