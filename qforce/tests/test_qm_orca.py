import numpy as np
import pytest

from qforce_examples import Orca_default
from ase.units import Hartree, mol, kJ

from qforce.qm.orca import ReadORCA
from test_qm_gaussian import TestReadHessian as Gaussian_hessian
from test_qm_gaussian import TestReadScan as Gaussian_scan

class TestReadHessian(Gaussian_hessian):
    @staticmethod
    @pytest.fixture(scope='class')
    def hessian():
        class Config(dict):
            charge_method = "cm5"
            charge = 0
            multiplicity = 1

        (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds,
         b_orders, lone_e, point_charges) = ReadORCA().hessian(Config(),
                                                               Orca_default['out_file'],
                                                               Orca_default['hess_file'],
                                                               Orca_default['pc_file'],
                                                               Orca_default['coord_file'],)

        return n_atoms, charge, multiplicity, elements, coords, hessian, \
               n_bonds,  b_orders, lone_e, point_charges

    def test_coords(self, hessian):
        (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds,
         b_orders, lone_e, point_charges) = hessian
        assert all(np.isclose(coords[0, :],
                              [-5.48129672124137, 1.91902042205872,
                               -0.07175480174836], rtol=0.01))

    def test_point_charges(self, hessian):
        (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds,
         b_orders, lone_e, point_charges) = hessian
        assert all(np.isclose(point_charges, [-0.080974, -0.041775,
                                              0.026889, 0.025293, 0.025293,
                                              -0.080972, 0.024433, 0.024433,
                                              0.026858, 0.025277, 0.025277],
                              atol=0.00001))

class TestReadScan(Gaussian_scan):
    @staticmethod
    @pytest.fixture(scope='class')
    def scan():
        class Config(dict):
            charge_method = "cm5"

        (n_atoms, coords, angles, energies, point_charges) = ReadORCA().scan(Config(),
                                                                   Orca_default['fragments'])

        return n_atoms, coords, angles, energies, point_charges

    def test_coords(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert all(np.isclose(coords[0][0],
                              [-5.481060, 1.918927, -0.071752], rtol=0.01))
        assert len(coords) == 24

    def test_angles(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert np.isclose(angles[0], -180, atol=0.01)
        assert np.isclose(angles[1], -180+15, atol=0.01)
        assert len(angles) == 24

    def test_energies(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        energy = ['-118.99155758', '-118.99088404', '-118.98920342',
                  '-118.98742741', '-118.98664049', '-118.98736535',
                  '-118.98914027', '-118.99085467', '-118.99156414',
                  '-118.99089879', '-118.98920912', '-118.98741123',
                  '-118.98663730', '-118.98741272', '-118.98920222',
                  '-118.99090278', '-118.99156663', '-118.99085654',
                  '-118.98914716', '-118.98737038', '-118.98663578',
                  '-118.98742144', '-118.98919569', '-118.99088786']
        energy = np.array([float(point) for point in energy])
        energies = energies * kJ / Hartree / mol
        assert all(np.isclose(energies, energy, atol=0.01))