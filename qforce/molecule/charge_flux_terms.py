import numpy as np
from abc import abstractmethod
#
from .baseterms import TermABC, TermFactory
from ..forces import get_dist, get_angle


class ChargeFluxBaseTerm(TermABC):

    def do_flux(self, crd, q_flux):
        self._calc_flux(crd, q_flux, self.fconst)

    def do_flux_fitting(self, crd, q_flux):
        """compute fitting contributions"""
        self._calc_flux(crd, q_flux[self.flux_idx], 1.0)

    @classmethod
    def get_term(cls, atomids, eq, f_type):
        return cls(atomids, eq, f_type)

    @abstractmethod
    def _calc_flux(self, crd):
        """compute the charge flux for a given term"""


class BondChargeFluxTerm(ChargeFluxBaseTerm):
    name = 'BondChargeFluxTerm'

    def _calc_forces(self, crd, force, fconst):
        return 0.

    def _calc_flux(self, crd, q_flux, j_param):
        a2, a1, a3 = self.atomids
        r = get_dist(crd[a1], crd[a3])[1]
        dr = r-self.equ
        # alpha = 2.2305741871043736
        # dr = 1-np.exp(-alpha*(r-self.equ))
        q_flux[a2] += j_param*dr
        q_flux[a1] -= j_param*dr


class BondPrimeChargeFluxTerm(BondChargeFluxTerm):
    name = 'BondPrimeChargeFluxTerm'


class AngleChargeFluxTerm(ChargeFluxBaseTerm):
    name = 'AngleChargeFluxTerm'

    def _calc_forces(self, crd, force, fconst):
        return 0.

    def _calc_flux(self, crd, q_flux, j_param):
        a2, a3, a1, a4 = self.atomids
        theta = get_angle([crd[a3], crd[a1], crd[a4]])[0]
        # dtheta = np.cos(theta)-np.cos(self.equ)
        dtheta = theta-self.equ
        q_flux[a2] += j_param*dtheta
        q_flux[a1] -= j_param*dtheta


class AnglePrimeChargeFluxTerm(AngleChargeFluxTerm):
    name = 'AnglePrimeChargeFluxTerm'


class BondBondChargeFluxTerm(ChargeFluxBaseTerm):
    name = 'BondBondChargeFluxTerm'

    def _calc_forces(self, crd, force, fconst):
        return 0.

    def _calc_flux(self, crd, q_flux, j_param):
        a2, a1, a3 = self.atomids

        r1 = get_dist(crd[a1], crd[a2])[1]
        dr1 = r1-self.equ[0]
        r2 = get_dist(crd[a1], crd[a3])[1]
        dr2 = r2-self.equ[1]

        q_flux[a2] += j_param*dr1*dr2
        q_flux[a3] -= j_param*dr1*dr2


class BondAngleChargeFluxTerm(ChargeFluxBaseTerm):
    name = 'BondAngleChargeFluxTerm'

    def _calc_forces(self, crd, force, fconst):
        return 0.

    def _calc_flux(self, crd, q_flux, j_param):
        a2, a3, a1, a4 = self.atomids
        theta = get_angle([crd[a3], crd[a1], crd[a4]])[0]
        dtheta = theta-self.equ[0]

        r = get_dist(crd[a1], crd[a2])[1]
        dr = r-self.equ[1]

        q_flux[a2] += 2*j_param*dtheta*dr
        q_flux[a3] -= j_param*dtheta*dr
        q_flux[a4] -= j_param*dtheta*dr


class AngleAngleChargeFluxTerm(ChargeFluxBaseTerm):
    name = 'AngleAngleChargeFluxTerm'

    def _calc_forces(self, crd, force, fconst):
        return 0.

    def _calc_flux(self, crd, q_flux, j_param):
        a2, a1, a3, a4, a1, a5 = self.atomids
        theta1 = get_angle([crd[a2], crd[a1], crd[a3]])[0]
        dtheta1 = theta1-self.equ[0]
        theta2 = get_angle([crd[a4], crd[a1], crd[a5]])[0]
        dtheta2 = theta2-self.equ[1]

        q_flux[a2] += j_param*dtheta1*dtheta2
        q_flux[a3] += j_param*dtheta1*dtheta2

        q_flux[a4] -= j_param*dtheta1*dtheta2
        q_flux[a5] -= j_param*dtheta1*dtheta2


class ChargeFluxTerms(TermFactory):
    name = 'ChargeFluxTerms'

    _term_types = {
        'bond': BondChargeFluxTerm,
        'bond_prime': BondPrimeChargeFluxTerm,
        'angle': AngleChargeFluxTerm,
        'angle_prime': AnglePrimeChargeFluxTerm,
        '_bond_bond': BondBondChargeFluxTerm,
        '_bond_angle': BondAngleChargeFluxTerm,
        '_angle_angle': AngleAngleChargeFluxTerm,
    }

    _always_on = []
    _default_off = ['bond', 'angle', 'bond_prime', 'angle_prime', '_bond_bond', '_bond_angle', '_angle_angle']

    @classmethod
    def get_terms(cls, topo, non_bonded, settings):
        if not any(val for val in settings.values()):
            return cls.get_terms_container()

        terms = cls.get_terms_container()

        # helper functions to improve readability
        def add_term(name, atomids, *args):
            terms[name].append(cls._term_types[name].get_term(atomids, *args))

        if topo.n_atoms == 2:
            central_atoms = [0]
        else:
            central_atoms = np.where(topo.n_neighbors > 1)[0]

        for a1 in central_atoms:
            neighs = topo.neighbors[0][a1]
            # bonds = [bond for bond in topo.bonds if a1 in bond]
            angles = [angle for angle in topo.angles if a1 == angle[1]]

            for a2 in topo.neighbors[0][a1]:
                dist = get_dist(topo.coords[a1], topo.coords[a2])[1]
                if settings.get('bond'):
                    add_term('bond', [a2, a1, a2], dist, 'bond')

                for a3 in topo.neighbors[0][a1]:
                    if a2 <= a3:
                        continue

                    if settings.get('_bond_bond'):
                        dist2 = get_dist(topo.coords[a1], topo.coords[a3])[1]
                        add_term('_bond_bond', [a2, a1, a3], [dist, dist2], 'bond_bond')

            for a2, _, a3 in angles:
                if settings.get('bond_prime'):
                    dist = get_dist(topo.coords[a1], topo.coords[a3])[1]
                    add_term('bond_prime', [a2, a1, a3], dist, 'bond_prime')
                    dist = get_dist(topo.coords[a1], topo.coords[a2])[1]
                    add_term('bond_prime', [a3, a1, a2], dist, 'bond_prime')

                theta = get_angle(topo.coords[[a2, a1, a3]])[0]

                if settings.get('angle'):
                    add_term('angle', [a2, a2, a1, a3], theta, 'angle')
                    add_term('angle', [a3, a3, a1, a2], theta, 'angle')

                if settings.get('angle_prime'):
                    options = [option for option in neighs if option != a2 and option != a3]
                    if options:
                        for a4 in options:
                            add_term('angle_prime', [a4, a3, a1, a2], theta, 'angle_prime')

                if settings.get('_bond_angle'):
                    for a4 in neighs:
                        dist = get_dist(topo.coords[a1], topo.coords[a4])[1]
                        if a4 in [a2, a3]:
                            add_term('_bond_angle', [a4, a2, a1, a3], [theta, dist], 'bond_angle')
                        else:
                            add_term('_bond_angle', [a4, a2, a1, a3], [theta, dist], 'bond_angle_prime')

                if settings.get('_angle_angle'):
                    for a4, _, a5 in angles:
                        if a2 == a4 and a3 == a5:
                            continue
                        theta2 = get_angle(topo.coords[[a4, a1, a5]])[0]
                        if a4 in [a2, a3] or a5 in [a2, a3]:
                            add_term('_angle_angle', [a2, a1, a3, a4, a1, a5], [theta, theta2], 'angle_angle')
                        else:
                            add_term('_angle_angle', [a2, a1, a3, a4, a1, a5], [theta, theta2], 'angle_angle_prime')

        return terms
