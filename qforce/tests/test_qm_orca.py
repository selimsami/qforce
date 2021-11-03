from pkg_resources import resource_filename

import numpy as np

from qforce.qm.orca import ReadORCA


def test_read_orca_hess():
    hess_file = resource_filename(__name__, 'orca_data/opt.hess')
    hessian = ReadORCA._read_orca_hess(hess_file)
    assert np.isclose(hessian[0], 7.1312338030E-01, rtol=0.01)
    assert np.isclose(hessian[1], -2.5810030509E-02, rtol=0.01)
    assert np.isclose(hessian[2], 6.6351936698E-01, rtol=0.01)


def test_read_orca_esp():
    charge_file = resource_filename(__name__, 'orca_data/charge.pc_chelpg')
    n_atoms, point_charges = ReadORCA._read_orca_esp(charge_file)
    assert n_atoms == 42
    assert np.isclose(point_charges[0], 0.691092, rtol=0.01)


def test_read_orca_cm5():
    charge_file = resource_filename(__name__, 'orca_data/hessian.log')
    n_atoms, point_charges = ReadORCA._read_orca_cm5(charge_file)
    assert n_atoms == 42
    assert np.isclose(point_charges[0], 0.131143, rtol=0.01)


def test_read_orca_xyz():
    xyz_file = resource_filename(__name__, 'orca_data/opt.xyz')
    n_atoms, elements, coords = ReadORCA._read_orca_xyz(xyz_file)
    assert n_atoms == 42
    assert elements[0] == 6
    assert np.isclose(coords[0, 0], -3.65036832744174, rtol=0.01)


def test_read_orca_nbo_analysis():
    log = resource_filename(__name__, 'orca_data/hessian.log')
    n_bonds, b_orders, lone_e = ReadORCA._read_orca_nbo_analysis(log, 42)
    assert n_bonds[0] == 4
    assert np.isclose(b_orders[0][1], 1.2984, rtol=0.01)
    assert lone_e[7] == 2

def test_read_orca_dat():
    dat = resource_filename(__name__, 'orca_data/relaxscanact.dat')
    parameter, energy = ReadORCA._read_orca_dat(dat)
    assert np.isclose(parameter[0], -179.62000000, rtol=0.01)
    assert np.isclose(parameter[1], -164.62000000, rtol=0.01)
    assert np.isclose(energy[0], -44.58283483 , rtol=0.01)

def test_read_orca_allxyz():
    xyz_file = resource_filename(__name__, 'orca_data/scan.allxyz')
    n_atoms, elements, coords = ReadORCA._read_orca_allxyz(xyz_file)
    assert len(coords) == 24