from .polarize import polarize
from .initialize import initialize
from .qm.qm import QM
from .qm.qm_base import HessianOutput
from .forcefield import ForceField
from .molecule import Molecule
from .fragment import fragment
from .dihedral_scan import DihedralScan
from .frequencies import calc_qm_vs_md_frequencies
from .hessian import fit_hessian

from .misc import check_if_file_exists, LOGO
from colt import from_commandline
from colt.validator import Validator


# define new validator
Validator.overwrite_validator("file", check_if_file_exists)


@from_commandline("""
# Input coordinate file mol.ext (ext: pdb, xyz, gro, ...)
# or directory (mol or mol_qforce) name.
file = :: file

# File name for the optional options.
options = :: file, optional, alias=o
""", description={
    'logo': LOGO,
    'alias': 'qforce',
    'arg_format': {
        'name': 12,
        'comment': 60,
        },
})
def run(file, options):
    run_qforce(input_arg=file, config=options)


def run_qforce(input_arg, ext_q=None, ext_lj=None, config=None, presets=None):
    config, job = initialize(input_arg, config, presets)

    if config.ff._polarize:
        polarize(job, config.ff)

    qm = QM(job, config.qm)
    qm_hessian_out = qm.read_hessian()

    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    md_hessian = fit_hessian(config.terms, mol, qm_hessian_out)

    if len(mol.terms['dihedral/flexible']) > 0 and config.scan.do_scan:
        fragments = fragment(mol, qm, job, config)
        DihedralScan(fragments, mol, job, config)

    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)
    ff = ForceField(job.name, config, mol, mol.topo.neighbors)
    ff.write_gromacs(job.dir, mol, qm_hessian_out.coords)

    print_outcome(job.dir)


def run_hessian_fitting_for_external(job_dir, qm_data, ext_q=None, ext_lj=None,
                                     config=None, presets=None):
    config, job = initialize(job_dir, config, presets)

    qm_hessian_out = HessianOutput(config.qm.vib_scaling, **qm_data)

    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    md_hessian = fit_hessian(config.terms, mol, qm_hessian_out)
    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)

    ff = ForceField(job.name, config, mol, mol.topo.neighbors)
    ff.write_gromacs(job.dir, mol, qm_hessian_out.coords)

    print_outcome(job.dir)

    return mol.terms


def print_outcome(job_dir):
    print(f'Output files can be found in the directory: {job_dir}.')
    print('- Q-Force force field parameters in GROMACS format (gas.gro, gas.itp, gas.top).')
    print('- QM vs MM vibrational frequencies, pre-dihedral fitting (frequencies.txt,'
          ' frequencies.pdf).')
    print('- Vibrational modes which can be visualized in VMD (frequencies.nmd).')
    print('- QM vs MM dihedral profiles (if any) in "fragments" folder as ".pdf" files.')
