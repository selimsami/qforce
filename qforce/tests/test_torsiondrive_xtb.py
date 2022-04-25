import numpy as np
import pytest

from ase.units import Hartree, mol, kJ

from qforce_examples import xTB_default
from qforce.qm.torsiondrive_xtb import TorsiondrivexTB


class Test_xtb_torsiondrive_read():
    @staticmethod
    @pytest.fixture(scope='class')
    def torsiondrive():
        (n_atoms, coord_list, angle_list, energies, point_charges) = \
            TorsiondrivexTB.read(xTB_default['torsiondrive_energy'])
        return n_atoms, coord_list, angle_list, energies, point_charges['xtb']

    def test_n_atoms(self, torsiondrive):
        (n_atoms, coord_list, angle_list, energies, point_charges) = torsiondrive
        assert n_atoms == 11

    def test_coord_list(self, torsiondrive):
        (n_atoms, coord_list, angle_list, energies,
         point_charges) = torsiondrive
        assert len(coord_list) == 24
        assert np.allclose(coord_list[0][0],
                           [-5.4794123852, 1.9116552737, -0.0539906279])

    def test_angle_list(self, torsiondrive):
        (n_atoms, coord_list, angle_list, energies,
         point_charges) = torsiondrive
        assert len(angle_list) == 24
        assert angle_list[0] == -165

    def test_energies(self, torsiondrive):
        (n_atoms, coord_list, angle_list, energies,
         point_charges) = torsiondrive
        assert len(energies) == 24
        assert np.isclose(energies[0] / Hartree / mol * kJ, -10.500245247)

    def test_point_charges(self, torsiondrive):
        (n_atoms, coord_list, angle_list, energies,
         point_charges) = torsiondrive
        assert len(point_charges) == 11
        assert np.isclose(point_charges[0], -0.10210149)
