import numpy as np
import pytest

from qforce_examples import xTB_default
from ase.units import Hartree, mol, kJ

from qforce.qm.xtb import ReadxTB


class TestReadHessian():
    @staticmethod
    @pytest.fixture(scope='class')
    def hessian():
        class Config(dict):
            charge_method = "xtb"
            charge = 0
            multiplicity = 1

        (n_atoms, charge, multiplicity, elements, coords, hessian,
         b_orders, point_charges) = ReadxTB().hessian(Config(),
                                                      xTB_default['hess_file'],
                                                      xTB_default['pc_file'],
                                                      xTB_default['coord_file'],
                                                      xTB_default['wbo_file'],)

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
        assert all(np.isclose(coords[0, :], [-5.47755155927663, 1.91096783090613,
                                             -0.07170582428722], rtol=0.01))

    def test_hessian(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert np.isclose(hessian[0], 4561.082672635644, rtol=0.1)
        assert np.isclose(hessian[1], -14.672083575455416, rtol=0.1)
        assert np.isclose(hessian[2], 4823.0334552113545, rtol=0.1)

    def test_b_orders(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert all(np.isclose(b_orders[0],
                              [0.0000, 1.0313, 0.9237, 0.9234, 0.9234, 0.0091, 0.0024, 0.0024,
                               0.0102, 0.0009, 0.0009], atol=0.1))

    def test_point_charges(self, hessian):
        n_atoms, charge, multiplicity, elements, coords, hessian, b_orders, point_charges = hessian
        assert all(np.isclose(point_charges, [-0.10206769, -0.04738168, 0.03075230, 0.03360933,
                                              0.03360297, -0.10205990, 0.02778609, 0.02778578,
                                              0.03077589, 0.03359926, 0.03359765], atol=0.00001))


class TestReadScan():
    @staticmethod
    @pytest.fixture(scope='class')
    def scan():
        class Config(dict):
            charge_method = "cm5"

        (n_atoms, coords, angles,
         energies, point_charges) = ReadxTB().scan(Config(), xTB_default['fragments_coord'])

        return n_atoms, coords, angles, energies, point_charges

    def test_n_atoms(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert n_atoms == 11

    def test_coords(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert all(np.isclose(coords[0][0], [-5.47742873919790, 1.91100460922811,
                                             -0.07168838231625], rtol=0.01))
        assert len(coords) == 24

    def test_angles(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        assert np.isclose(angles[0], -180, atol=0.1)
        assert np.isclose(angles[1], -180+15, atol=0.1)
        assert len(angles) == 24

    def test_energies(self, scan):
        (n_atoms, coords, angles, energies, point_charges) = scan
        energy = '-10.500828139210,-10.500245567210,-10.498793187023'
        energy = energy.split(',')
        energy = np.array([float(point) for point in energy])
        energies = energies * kJ / Hartree / mol
        assert all(np.isclose(energies[:3], energy, atol=0.01))
