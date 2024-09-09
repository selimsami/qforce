import pytest

from qforce_examples import Gaussian_default
from qforce.main import run_qforce


@pytest.mark.parametrize("batch_run,exist", [(True, False), (False, True)])
def test_BatchRun(batch_run, exist, tmpdir):
    setting = tmpdir.join('settings')
    setting.write('''[scan]
batch_run = {}
frag_lib = {}/qforce_fragments
    '''.format(batch_run, tmpdir))
    tmpdir.mkdir('propane_qforce')
    tmpdir.join('propane_qforce').join('propane_hessian.log').mksymlinkto(
        Gaussian_default['out_file'])
    tmpdir.join('propane_qforce').join('propane_hessian.fchk').mksymlinkto(
        Gaussian_default['fchk_file'])
    tmpdir.join('propane.xyz').mksymlinkto(
        Gaussian_default['xyz_file'])

    # First run
    try:
        run_qforce(input_arg=tmpdir.join('propane.xyz').strpath,
                   config=tmpdir.join('settings').strpath)
    except SystemExit:
        pass
    # Fragment file generated
    assert tmpdir.join('propane_qforce').join('fragments').join(
        'CC_H8C3_d91b46644317dee9c2b868166c66a18c~1.inp').isfile()
    tmpdir.join('propane_qforce').join('fragments').remove()

    # Second run
    try:
        run_qforce(input_arg=tmpdir.join('propane.xyz').strpath,
                   config=tmpdir.join('settings').strpath)
    except SystemExit:
        pass
    # Fragment file generated again if batch_run is False
    # Fragment file not generated again if batch_run is True
    assert tmpdir.join('propane_qforce').join('fragments').join(
        'CC_H8C3_d91b46644317dee9c2b868166c66a18c~1.inp').isfile() is exist
