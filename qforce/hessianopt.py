from copy import deepcopy
from scipy import optimize
from scipy.optimize import minimize, lsq_linear
import numpy as np


class TermsOptimizer:

    def __init__(self, mol, start_param=None):
        self._terms = deepcopy(mol.terms)
        self._natoms = mol.topo.n_atoms
        self.n_fitted_terms = mol.terms.n_fitted_terms
        if start_param is not None:
            assert self.n_fitted_terms == len(start_param)
            start_param = list(start_param) + [1000]
        self._start_param = None # start_param
        self._params = self._get_parameters()
        self._i = 0
        self._j = 0

    def _get_parameters(self):
        params = [None for _ in range(self.n_fitted_terms+1)]
        nmax = len(params)+1
        with self._terms.add_ignore(['dihedral/flexible', 'charge_flux']):
            for term in self._terms:
                params[term.idx] = term.fconst
        return params
    
    def _get_start_parameters(self):
        if self._start_param is not None:
            return self._start_param 
        start_values = np.array([val if val is not None else 1000
                                 for val in self._get_parameters()])
        return start_values

    def _update(self, params):
        nmax = len(params)+1
        with self._terms.add_ignore(['dihedral/flexible', 'charge_flux']):
            for term in self._terms:
                if term.idx != self.n_fitted_terms:
                    term.fconst = params[term.idx]

    def _build_Aymat(self, qms):
        A = []
        y = []
        for qm in qms:
            hessian, full_md_hessian_1d = [], []
            non_fit = []
            qm_hessian = np.copy(qm.hessian)

            full_md_hessian = calc_hessian(qm.coords, self._terms, self._natoms, self.n_fitted_terms)
            count = 0
            for i in range(self._natoms*3):
                for j in range(i+1):
                    hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
                    if all([h == 0 for h in hes]) or np.abs(qm_hessian[count]) < 0.0001:
                        qm_hessian = np.delete(qm_hessian, count)
                        full_md_hessian_1d.append(np.zeros(self.n_fitted_terms))
                    else:
                        count += 1
                        hessian.append(hes[:-1])
                        full_md_hessian_1d.append(hes[:-1])
                        non_fit.append(hes[-1])

            difference = qm_hessian - np.array(non_fit)
            y += list(difference)
            A += hessian

        # solve A*x = y for x
        return np.array(A), np.array(y)

    def _force(self, coords):
        energy = 0.0
        force = np.zeros((self._natoms, 3), dtype=float)
        with self._terms.add_ignore(['dihedral/flexible', 'charge_flux']):
            for term in self._terms:
                energy += term.do_force(coords, force)
        return energy, force

    def _hessian(self, coords, disp=1e-5, as_lowertri=False):
        hessian = np.zeros((3*self._natoms, 3*self._natoms), dtype=float)
        dispt2 = 2.0*disp

        with self._terms.add_ignore(['dihedral/flexible', 'charge_flux']):
            for a in range(self._natoms):
                for xyz in range(3):
                    f_plus = np.zeros((self._natoms, 3), dtype=float)
                    f_minus = np.zeros((self._natoms, 3), dtype=float)
                    coords[a][xyz] += disp
                    for term in self._terms:
                        term.do_force(coords, f_plus)
                    coords[a][xyz] -= dispt2
                    for term in self._terms:
                        term.do_force(coords, f_minus)
                    coords[a][xyz] += 1e-5
                    diff = - (f_plus - f_minus) / dispt2
                    hessian[a*3+xyz] = diff.reshape(self._natoms*3)
        if as_lowertri is False:
            return hessian
        tril = np.tril_indices(self._natoms*3)
        return hessian[tril]

    def hessian_linearopt(self, qms):
        A, y = self._build_Aymat(qms)
        return lsq_linear(A, y).x

    def hessian_linear_optimizer(self, qms):
        A, y = self._build_Aymat(qms)

        def _helper(x):
            x = np.array(x)
            yref = np.dot(A, x)
            assert len(yref) == len(y)

            res = np.sqrt(np.sum([(_y - _y0)**2 for _y, _y0 in zip(y, yref)]))
            print(res)
            return res

        start_values = self._get_start_parameters()[:-1]
        res = minimize(_helper, start_values)
        return res.x

    def hessian_optimization(self, qms, *, use_jac=False, method = 'SLSQP'):

        def _helper(params):
            # set params to the correct values
            self._update(params)

            n = 0
            sum_value = 0.0

            for qm in qms:
                md_hessian = self._hessian(qm.coords, as_lowertri=True).flatten()
                qm_hessian = qm.hessian.flatten()
                n += qm_hessian.size
                sum_value += np.sum((x-y)*(x-y) for x, y in zip(md_hessian, qm_hessian))

            res = np.sqrt(sum_value/n)
            return res

        def _gradient_helper(params, tol=0.02):

            grad = np.zeros((len(params),), dtype=float)
            twicetol = 2.0*tol

            for i in range(len(params)):
                params[i] += tol
                pval = _helper(params)
                params[i] -= twicetol
                mval = _helper(params)
                params[i] += tol
                grad[i] = (pval - mval)/twicetol
            self._j += 1
            print("computed gradient! ", self._j, grad)
            return grad


        # set all values to 1000...choose smarter defaults...
        start_values = self._get_start_parameters()

        if use_jac is True:
            res = minimize(_helper, start_values, jac=_gradient_helper, method=method)
        else:
            res = minimize(_helper, start_values, method=method)

        return res


def calc_hessian(coords, terms, natoms, n_fitted_terms):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*natoms, 3*natoms,
                             n_fitted_terms+1))

    with terms.add_ignore(['dihedral/flexible', 'charge_flux']):
        for a in range(natoms):
            for xyz in range(3):
                coords[a][xyz] += 1e-5
                _, f_plus = calc_forces(coords, terms, natoms, n_fitted_terms)
                coords[a][xyz] -= 2e-5
                _, f_minus = calc_forces(coords, terms, natoms, n_fitted_terms)
                coords[a][xyz] += 1e-5
                diff = - (f_plus - f_minus) / 2e-5
                full_hessian[a*3+xyz] = diff.reshape(n_fitted_terms+1, 3*natoms).T
    return full_hessian


def calc_forces(coords, terms, natoms, n_fitted_terms):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """
    energy = np.zeros(n_fitted_terms+1)
    force = np.zeros((n_fitted_terms+1, natoms, 3))

    for term in terms:
        energy[term.idx] += term.do_fitting(coords, force)

    return energy, force
