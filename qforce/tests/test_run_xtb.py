import os

import numpy as np
import pytest
import parmed as pmd
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

        os.chdir('fragments')
        run_fragment = outdir / 'propane_qforce' / 'fragments' / 'CC_H8C3_d91b46644317dee9c2b868166c66a18c~1.inp'
        call(run_fragment.read_text(), shell=True)

        # Generate the topology
        try:
            run_qforce(input_arg=str(xyz),
                       config=str(setting))
        except SystemExit:
            pass

        top = pmd.load_file(str(outdir / 'propane_qforce' / 'gas.top'),
                            xyz = str(outdir / 'propane_qforce' / 'gas.gro'))
        os.chdir(cwd)
        return top

    def test_top(self, propane):
        '''Test if the whole process runs.'''
        assert isinstance(propane, pmd.gromacs.GromacsTopologyFile)

    def test_charge(self, propane):
        '''Test if the charges are generated in the same fashion.'''
        charges = [atom.charge for atom in propane.atoms]
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
        assert np.allclose(ref_charge, charges, atol=0.01)

    def test_elements(self, propane):
        atom_ids = [atom.atomic_number for atom in propane.atoms]
        assert [6, 6, 1, 1, 1, 6, 1, 1, 1, 1, 1] == atom_ids

    def test_angles(self, propane):
        assert len(propane.angles) == 18
        assert propane.angles[0].funct == 5
        assert propane.angles[0].atom1.idx == 0
        assert propane.angles[0].atom2.idx == 1
        assert propane.angles[0].atom3.idx == 5
        assert np.isclose(propane.angles[0].type.k, 54.72, atol=0.01)
        assert np.isclose(propane.angles[0].type.theteq, 111.62, atol=0.01)

    def test_bonds(self, propane):
        assert len(propane.bonds) == 10
        assert propane.bonds[0].funct == 1
        assert propane.bonds[0].atom1.idx == 0
        assert propane.bonds[0].atom2.idx == 1
        assert np.isclose(propane.bonds[0].type.k, 211.478, atol=0.01)
        assert np.isclose(propane.bonds[0].type.req, 1.5242, atol=0.01)

    def test_dihedrals(self, propane):
        assert len(propane.rb_torsions) == 2
        assert propane.rb_torsions[0].atom1.idx == 2
        assert propane.rb_torsions[0].atom2.idx == 0
        assert propane.rb_torsions[0].atom3.idx == 1
        assert propane.rb_torsions[0].atom4.idx == 5
        assert propane.rb_torsions[0].funct == 3
        assert np.isclose(propane.rb_torsions[0].type.c0, 0.9964, atol=0.01)

    def test_defaults(self, propane):
        assert propane.defaults.comb_rule == 3
        assert propane.defaults.fudgeLJ == 0.5
        assert propane.defaults.fudgeQQ == 0.5
        assert propane.defaults.nbfunc == 1
