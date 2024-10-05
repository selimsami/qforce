from copy import deepcopy

from qforce.cli import initialize
from qforce.main import runjob2
from qforce.fit import *
from scipy.optimize import minimize

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np


def compute_AB_matrix(logger, mol, structs):
    """
    Doing linear/Ridge regression with all energies, forces and hessian terms:
    Ax = B

    A: MM contributions to each e/f/h item by each fitted FF term (shape: n_items x n_fit_terms)
    B: Total QM e/f/h minus MM contributions from non-fitted terms (like non-bonded) (shape: n_items)
    w: weights

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

    for weight_f, qmgrad in structs.graditr():
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
    return A, B, weights


def compute_hessian(coords, mol):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms), dtype=float)

    with mol.terms.add_ignore('charge_flux'):
        for a in range(mol.topo.n_atoms):
            for xyz in range(3):
                coords[a][xyz] += 1e-5
                _, f_plus = compute_forces(coords, mol)
                coords[a][xyz] -= 2e-5
                _, f_minus = compute_forces(coords, mol)
                coords[a][xyz] += 1e-5
                diff = - (f_plus - f_minus) / 2e-5
                full_hessian[a*3+xyz] = diff.flatten()
    return full_hessian


def compute_forces(coords, mol):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """
    energy = 0.0
    force = np.zeros((mol.topo.n_atoms, 3))

    for term in mol.terms:
        energy += term.do_force(coords, force)

    return energy, force


def compute_struct_rmsd(logger, mol, structs):
    """
    Doing linear/Ridge regression with all energies, forces and hessian terms:
    Ax = B

    A: MM contributions to each e/f/h item by each fitted FF term (shape: n_items x n_fit_terms)
    B: Total QM e/f/h minus MM contributions from non-fitted terms (like non-bonded) (shape: n_items)
    w: weights

    """

    A = []
    B = []
    weights = []

    # Compute energies for the lowest energy structure, to subtract from other energies
    e_lowest, f_lowest = compute_forces(mol.qm_minimum_coords, mol)
    # f_lowest *= -1  # convert from force to gradient
    # f_lowest = f_lowest.reshape(mol.terms.n_fitted_terms+1, mol.topo.n_atoms*3)
    # qm_force = np.zeros((mol.n_atoms, 3))
    # A += list(f_lowest[:-1].T)
    # B += list(qm_force.flatten() - f_lowest[-1])
    # weights += [1e7]*qm_force.size
    # A.append(e_lowest[:-1] - e_lowest[:-1])
    # B.append(0 - e_lowest[-1] + e_lowest[-1])
    # weights.append(1e7)
    natoms = mol.topo.n_atoms

    logger.info("Calculating the MM hessian matrix elements for all structures ...")
    for weight, qm in structs.hessitr():
        hessian = []
        non_fit = []
        # lower triangular
        qm_hessian = np.copy(qm.hessian)
        # full matrix
        full_md_hessian = compute_hessian(qm.coords, mol)
        # transform in lower triangular matrix
        md_hessian = (full_md_hessian[np.tril_indices(3*natoms)] + full_md_hessian[np.triu_indices(3*natoms)])/2.0

        A += list(md_hessian.flatten())
        B += list(qm_hessian.flatten())
        weights += [weight]*qm_hessian.size

    logger.info("Calculating energies/forces for all additional structures...")
    for weight_e, qmen in structs.enitr():
        mm_energy, _ = compute_forces(qmen.coords, mol)
        A.append(mm_energy - e_lowest)
        B.append(qmen.energy)
        weights.append(weight_e)

    # scale for energy structs, grad structs only have weight_f stored, but the ratio is constant!
    factor_e = structs.energy_weight / structs.gradient_weight

    for weight_f, qmgrad in structs.graditr():
        weight_e = factor_e * weight_f
        mm_energy, mm_force = compute_forces(qmgrad.coords, mol)
        mm_force *= -1.0  # convert from force to gradient

        A += list(mm_force.flatten())
        B += list(qmgrad.gradient.flatten())
        weights += [weight_f]*qmgrad.gradient.size

        A.append(mm_energy - e_lowest)
        B.append(qmgrad.energy)
        weights.append(weight_e)
    return A, B, weights


def multi_fit(logger, mol, structs):
    """
    Doing linear/Ridge regression with all energies, forces and hessian terms:
    Ax = B

    A: MM contributions to each e/f/h item by each fitted FF term (shape: n_items x n_fit_terms)
    B: Total QM e/f/h minus MM contributions from non-fitted terms (like non-bonded) (shape: n_items)

    """

    A, B, weights = compute_AB_matrix(logger, mol, structs)

    logger.info("Fitting the force field parameters...")
    reg = Ridge(alpha=1e-6, fit_intercept=False).fit(A, B, sample_weight=weights)
    fit = reg.coef_
    logger.info("Done!\n")

    with mol.terms.add_ignore('charge_flux'):
        for term in mol.terms:
            if term.idx < len(fit):
                term.set_fitparameters(fit)

    return reg


def compute_rmsd(logger, mol, structs):
    # do force fitting with current equilibrium structure
    # return multi_fit(logger, mol, structs)
    # compute rmsd
    A, B, weights = compute_struct_rmsd(logger, mol, structs)
    logger.info("computing the RMSD...")
    return np.sqrt(mean_squared_error(A, B, sample_weight=weights))


class Fitter:

    def __init__(self, mol, equfits=['bond']):
        self.equfits = equfits
        self.mol = deepcopy(mol)
        self.parameters = self._get_fitparameters(equfits)

    def _get_fitparameters(self, equfits):

        parameters = {}

        for termcls in equfits:

            for term in self.mol.terms[termcls]:
                constants = term.constants()
                for constant in constants:
                    values = parameters.get(constant)
                    if values is None:
                        parameters[constant] = [term]
                    else:
                        values.append(term)

        with self.mol.terms.add_ignore(equfits):
            for term in self.mol.terms[termcls]:
                constants = term.constants()
                for constant in constants:
                    values = parameters.get(constant)
                    if values is not None:
                        values.append(term)


        for terms in self.averagize_parameters(equfits).values():
            if len(terms) == 1:
                continue

            # assume that its just one constant!!!!
            constants = [term.constants()[0] for term in terms]
            start = constants[0]

            values = []
            for constant in constants:
                values += parameters[constant]
                del parameters[constant]

            parameters[start] = values

        return parameters

    def averagize_parameters(self, equfits):
        # not sure if it should be done or not
        sorted_terms = {}

        for termcls in equfits:
            for term in self.mol.terms[termcls]:
                name = str(term)
                values = sorted_terms.get(name)
                if values is None:
                    sorted_terms[name] = [term]
                else:
                    values.append(term)

        return sorted_terms

    def optimize(self, logger, structs):

        # assume ordered dictionaries!
        names = [key for key in self.parameters.keys()]

        def _helper(values):
            values = {name: value for name, value in zip(names, values)}
            for terms in self.parameters.values():
                for term in terms:
                    term.update_constants(values)
            return compute_rmsd(logger, self.mol, structs)
        # assumes standard bond and angle terms!
        start_values = []
        for terms in self.parameters.values():
            start_values.append(terms[0].equ)
        #
        multi_fit(logger, self.mol, structs)

        print("equilibrium values = ", start_values)

        start_values = []
        for terms in self.parameters.values():
            start_values.append(terms[0].equ + 1.0)

        print("start_values = ", start_values)

        for _ in range(20):
            # do first multi_fit
            res = minimize(_helper, start_values) #  method='Powell')
            start_values = res.x
            multi_fit(logger, self.mol, structs)
        return res

    def gradient(self, logger, structs, diff):
        start_values = []
        for terms in self.parameters.values():
            start_values.append(terms[0].equ)

        def update(values):
            for terms, value in zip(self.parameters.values(), values):
                for term in terms:
                    term.equ = value

        grad = [0.0 for _ in range(len(start_values))]
        for i in range(len(start_values)):
            values = deepcopy(start_values)
            values[i] += diff
            update(values)
            val1 = compute_rmsd(logger, self.mol, structs)
            values[i] -= 2.0*diff
            update(values)
            val2 = compute_rmsd(logger, self.mol, structs)
            grad[i] = (val1 - val2)/(2.0*diff)
        return grad


config = {
        'file': 'methanol.xyz',
        'options': 'settings.ini',
        }

config, job = initialize(config)

mol, structs = runjob2(config, job)
fit = Fitter(mol, equfits=['bond'])


print("-------------------------------------")
res = fit.optimize(job.logger, structs)
print("-------------------------------------")
