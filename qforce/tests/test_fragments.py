import pytest

from qforce_examples import Gaussian_default
from qforce.main import run_qforce


def run(input_arg, config):
    try:
        run_qforce(input_arg=input_arg, config=config)
    except SystemExit:
        pass


@pytest.mark.parametrize("batch_run,exist", [(True, False), (False, True)])
def test_BatchRun(batch_run, exist, tmpdir):
    setting = tmpdir.join('settings')
    setting.write('''[scan]
batch_run = {}
frag_lib = {}/qforce_fragments
    '''.format(batch_run, tmpdir))
    tmpdir.join('propane.xyz').mksymlinkto(
        Gaussian_default['xyz_file'])
    # first run setup calculation
    run(input_arg=tmpdir.join('propane.xyz').strpath,
        config=tmpdir.join('settings').strpath)
    # add hessian output
    tmpdir.join('propane_qforce').join('1_hessian').join('propane_3fbff44995be158b3fd2daeef2df6f33_hessian.log').mksymlinkto(
        Gaussian_default['out_file'])
    tmpdir.join('propane_qforce').join('1_hessian').join('propane_3fbff44995be158b3fd2daeef2df6f33_hessian.fchk').mksymlinkto(
        Gaussian_default['fchk_file'])
    # Second run setup fragments
    run(input_arg=tmpdir.join('propane.xyz').strpath,
        config=tmpdir.join('settings').strpath)
    # Fragment file generated
    assert tmpdir.join('propane_qforce').join('2_fragments').join('CC_H8C3_d91b46644317dee9c2b868166c66a18c~1').join(
        'CC_H8C3_d91b46644317dee9c2b868166c66a18c~1.inp').isfile()

    tmpdir.join('propane_qforce').join('2_fragments').remove()

    # Second run
    run(input_arg=tmpdir.join('propane.xyz').strpath,
        config=tmpdir.join('settings').strpath)
    # Fragment file generated again if batch_run is False
    # Fragment file not generated again if batch_run is True
    assert tmpdir.join('propane_qforce').join('2_fragments').join('CC_H8C3_d91b46644317dee9c2b868166c66a18c~1').join(
                       'CC_H8C3_d91b46644317dee9c2b868166c66a18c~1.inp').isfile() is exist
