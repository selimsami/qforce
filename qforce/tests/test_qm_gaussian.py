import numpy as np
import pytest
from ase.units import Hartree, mol, kJ

from qforce_examples import Gaussian_default
from qforce.qm.gaussian import ReadGaussian


class TestReadHessian():
    @staticmethod
    @pytest.fixture(scope='class')
    def hessian():
        class Config(dict):
            charge_method = "cm5"

        (n_atoms, charge, multiplicity, elements, coords, hessian,
         b_orders, point_charges) = ReadGaussian().hessian(Config(),
                                                           Gaussian_default['out_file'],
                                                           Gaussian_default['fchk_file'])

        return n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges

    def test_n_atoms(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert n_atoms == 11

    def test_charge(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert charge == 0

    def test_multiplicity(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert multiplicity == 1

    def test_elements(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert all(elements == [6, 6, 1, 1, 1, 6, 1, 1, 1, 1, 1])

    def test_coords(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert all(np.isclose(coords[0, :],
                              [-1.277008, -0.260352, 0.000062], rtol=0.01))

    def test_hessian(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert np.isclose(hessian[0], 4561.082672635644, rtol=0.1)
        assert np.isclose(hessian[1], -419.45253955313194, rtol=0.1)
        assert np.isclose(hessian[2], 4823.0334552113545, rtol=0.1)

    def test_b_orders(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert all(np.isclose(b_orders[0],
                              [0.0000, 1.0313, 0.9237, 0.9234, 0.9234, 0.0091, 0.0024, 0.0024,
                               0.0102, 0.0009, 0.0009], atol=0.1))

    def test_point_charges(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert all(np.isclose(point_charges, [-0.241874, -0.165764, 0.081536, 0.080396, 0.080395,
                                              -0.241873, 0.08243, 0.08243, 0.081533, 0.080396,
                                              0.080396], atol=0.00001))


class TestReadScan():
    @staticmethod
    @pytest.fixture(scope='class')
    def scan():
        class Config(dict):
            charge_method = "cm5"

        (n_atoms, coords, angles,
         energies, point_charges) = ReadGaussian().scan(Config(), Gaussian_default['fragments'])

        return n_atoms, coords, angles, energies, point_charges

    def test_n_atoms(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert n_atoms == 11

    def test_coords(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert all(np.isclose(coords[0][0], [1.277261, -0.260282, -6.9e-05], rtol=0.01))
        assert len(coords) == 24

    def test_angles(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert np.isclose(angles[0], 180, atol=0.01)
        assert np.isclose(angles[1], 180+15, atol=0.01)
        assert len(angles) == 24

    def test_energies(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        energy = '-118.965912,-118.9652606,-118.9636263,-118.9618925,-118.96' \
                 '11238,-118.9618386,-118.9635621,-118.9652261,-118.9659119,' \
                 '-118.9652705,-118.9636241,-118.9618785,-118.9611235,-118.9' \
                 '618784,-118.9636246,-118.9652708,-118.96509119,-118.965226' \
                 '1,-118.9635621,-118.9618387,-118.9611238,-118.9618925,-118' \
                 '.9636264,-118.9652607'
        energy = energy.split(',')
        energy = np.array([float(point) for point in energy])
        energies = energies * kJ / Hartree / mol
        assert all(np.isclose(energies, energy, atol=0.01))
