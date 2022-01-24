import numpy as np
import pytest

from qforce_examples import Orca_default, ORCA_DMF, Orca_default_NBO6
from ase.units import Hartree, mol, kJ

from qforce.qm.orca import ReadORCA
from .test_qm_gaussian import TestReadHessian as Gaussian_hessian
from .test_qm_gaussian import TestReadScan as Gaussian_scan
from .test_qm_gaussian import TestReadLP as Gaussian_LP


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
        assert all(np.isclose(point_charges, [-0.052487, -0.017287, 0.017356,
                                              0.015348, 0.015348, -0.05249,
                                              0.013127, 0.013127, 0.017322,
                                              0.01534, 0.01534],
                              atol=0.0001))

    def test_hessian(self, hessian):
        (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds,
         b_orders, lone_e, point_charges) = hessian
        assert np.isclose(hessian[0], 4336.9313407, rtol=0.1)
        assert np.isclose(hessian[1], -35.78124679, rtol=0.1)
        assert np.isclose(hessian[2], 5317.32106175, rtol=0.1)

class TestReadScan(Gaussian_scan):
    @staticmethod
    @pytest.fixture(scope='class')
    def scan():
        class Config(dict):
            charge_method = "cm5"

        (n_atoms, coords, angles, energies, point_charges) = ReadORCA().scan(Config(),
                                                                   Orca_default['fragments_out'])

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
        energy = ['-119.07752449', '-119.07687528', '-119.07524691',
                  '-119.07350525', '-119.07272311', '-119.07344151',
                  '-119.07517421', '-119.07683536', '-119.07752403',
                  '-119.07688738', '-119.07524888', '-119.07348511',
                  '-119.07272154', '-119.07349001', '-119.07524223',
                  '-119.07689394', '-119.07752531', '-119.07683264',
                  '-119.07517599', '-119.07344223', '-119.07271719',
                  '-119.07349512', '-119.07524031', '-119.07688126']

        energy = np.array([float(point) for point in energy])
        energies = energies * kJ / Hartree / mol
        assert all(np.isclose(energies, energy, atol=0.01))

class TestReadHessian_NBO6(TestReadHessian):
    '''Test if NBO6 output could be read correctly'''
    @staticmethod
    @pytest.fixture(scope='class')
    def hessian():
        class Config(dict):
            charge_method = "cm5"
            charge = 0
            multiplicity = 1

        (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds,
         b_orders, lone_e, point_charges) = ReadORCA().hessian(Config(),
                                                               Orca_default_NBO6['out_file'],
                                                               Orca_default_NBO6['hess_file'],
                                                               Orca_default_NBO6['pc_file'],
                                                               Orca_default_NBO6['coord_file'],)

        return n_atoms, charge, multiplicity, elements, coords, hessian, \
               n_bonds,  b_orders, lone_e, point_charges
        
class TestReadLP(Gaussian_LP):
    @staticmethod
    @pytest.fixture(scope='class')
    def hessian():
        class Config(dict):
            charge_method = "cm5"
            charge = 0
            multiplicity = 1

        (n_atoms, charge, multiplicity, elements, coords, hessian, n_bonds,
         b_orders, lone_e, point_charges) = ReadORCA().hessian(Config(),
                                                               ORCA_DMF['out_file'],
                                                               ORCA_DMF['hess_file'],
                                                               ORCA_DMF['pc_file'],
                                                               ORCA_DMF['coord_file'],)

        return n_atoms, charge, multiplicity, elements, coords, hessian, \
               n_bonds,  b_orders, lone_e, 
