import os

import numpy as np
import pytest
from subprocess import call
from qforce.main import run_qforce
from qforce_examples import xTB_default

@pytest.mark.slow
class Test_runxTB():
    '''Do a fresh run and test if the result changes over time.'''
    @staticmethod
    @pytest.fixture(scope='session')
    def propane(tmp_path_factory):
        outdir = tmp_path_factory.mktemp('propane')
        setting = outdir / 'settings'
        setting.write_text('''[ff]
charge_scaling = 1.0
[qm]
software = xtb
[qm::software(xtb)]
xtb_command = --gfn 2
[scan]
frag_lib = {}/qforce_fragments
 '''.format(str(outdir)))
        xyz = outdir / 'propane.xyz'
        xyz.symlink_to(xTB_default['xyz_file'])

        # Generate the input file
        try:
            run_qforce(input_arg=str(xyz),
                       config=str(setting))
        except SystemExit:
            pass

        cwd = os.getcwd()
        os.chdir(outdir / 'propane_qforce')
        run_hessian = outdir / 'propane_qforce' / 'propane_hessian.inp'
        call(run_hessian.read_text(), shell=True)

        # Generate the fragments
        try:
            run_qforce(input_arg=str(xyz),
                       config=str(setting))
        except SystemExit:
            pass

        frag_dir = outdir / 'propane_qforce' / 'fragments'
        inp = [file for file in os.listdir(frag_dir) if file.endswith('.inp')][0]
        os.chdir('fragments')
        run_fragment = frag_dir / inp

        call(run_fragment.read_text(), shell=True)

        # Generate the topology
        try:
            mol = run_qforce(input_arg=str(xyz),
                       config=str(setting))
        except SystemExit:
            pass

        os.chdir(cwd)
        return mol

    # def test_top(self, propane):
    #     '''Test if the whole process runs.'''
    #     assert isinstance(propane, pmd.gromacs.GromacsTopologyFile)

    def test_charge(self, propane):
        '''Test if the charges are generated in the same fashion.'''
        ref_charge = [-0.10206769,
   -0.04738168,
    0.03075230,
    0.03360933,
    0.03360297,
   -0.10205990,
    0.02778609,
    0.02778578,
    0.03077589,
    0.03359926,
    0.03359765,
]
        assert np.allclose(ref_charge, propane.non_bonded.q, atol=0.01)

    def test_elements(self, propane):
        assert [6, 6, 1, 1, 1, 6, 1, 1, 1, 1, 1] == list(propane.elements)

    def test_angles(self, propane):
        angles = list(propane.terms['angle'])
        assert len(angles) == 18
        assert list(angles[0].atomids) == [0, 1, 5]
        assert np.isclose(angles[0].fconst, 457.945, atol=0.01)
        assert np.isclose(np.degrees(angles[0].equ), 111.62, atol=0.01)

    def test_bonds(self, propane):
        bonds = list(propane.terms['bond'])
        assert len(bonds) == 10
        assert list(bonds[0].atomids) == [0, 1]
        assert np.isclose(bonds[0].fconst, 1769.660, atol=0.01)
        assert np.isclose(bonds[0].equ, 1.5242, atol=0.01)

    def test_dihedrals(self, propane):
        diheds = list(propane.terms['dihedral/flexible'])
        assert len(diheds) == 2
        assert list(diheds[0].atomids) == [2, 0, 1, 5]
        assert np.isclose(diheds[0].equ[0], 4.169, atol=0.01)

    def test_defaults(self, propane):
        assert propane.non_bonded.comb_rule == 3
        assert propane.non_bonded.fudge_lj == 0.5
        assert propane.non_bonded.fudge_q == 0.5

