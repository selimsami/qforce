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


def runjob(config, job, ext_q=None, ext_lj=None):
    if config.ff._polarize:
        polarize(job, config.ff)

    # setup qm calculation
    qm = QM(job, config.qm)
    # do the preoptimization if selected
    qm.preopt()
    # get hessian output
    qm_hessian_out = qm.get_hessian()

    # check molecule
    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    # change the order
    fragments = None
    if len(mol.terms['dihedral/flexible']) > 0 and config.scan.do_scan:
        # get fragments with qm
        fragments = fragment(mol, qm, job, config)

    # hessian fitting
    md_hessian = fit_hessian(config.terms, mol, qm_hessian_out)

    # do the scans
    if fragments is not None:
        DihedralScan(fragments, mol, job, config)

    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)
    ff = ForceField(job.name, config, mol, mol.topo.neighbors)
    ff.write_gromacs(job.dir, mol, qm_hessian_out.coords)

    print_outcome(job.dir)


def run_qforce(input_arg, ext_q=None, ext_lj=None, config=None, presets=None):
    """Execute Qforce from python directly """
    config, job = initialize(input_arg, config, presets)
    #
    runjob(config, job, ext_q=ext_q, ext_lj=ext_lj)


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
