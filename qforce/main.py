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


def run_qforce(input_arg, ext_q=None, ext_lj=None, config=None, presets=None):
    config, job = initialize(input_arg, config, presets)

    if config.ff._polarize:
        polarize(job, config.ff)

    qm = QM(job, config.qm)
    qm_hessian_out = qm.read_hessian()

    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    md_hessian = fit_hessian(config.terms, mol, qm_hessian_out)

    if config.scan.do_scan:
        fragments = fragment(mol, qm, job, config.scan)
        DihedralScan(fragments, mol, job, config)

    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)
    ff = ForceField(job.name, config, mol, mol.topo.neighbors)
    ff.write_gromacs(job.dir, mol, qm_hessian_out.coords)

    print_outcome(job.dir)


def run_hessian_fitting_for_external(job_dir, qm_data, ext_q=None, ext_lj=None,
                                     config=None, presets=None):
    """

    Parameters
    ----------
    job_dir : str
        Path for the Q-Force job directory. "_qforce" is added automatically to the end.
    qm_data : dict
        DESCRIPTION.
    config : str, optional
        Options file for Q-Force. See documentation for the format.
    presets : str, optional
        Presets for the options file for Q-Force. See documentation for the format.


    """

    # Next: Read atom types and LJ (list vs string)

    config, job = initialize(job_dir, config, presets)

    qm_hessian_out = HessianOutput(**qm_data)

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
    print('- QM vs MM vibrational frequencies (frequencies.txt, frequencies.pdf).')
    print('- Vibrational modes which can be visualized in VMD (frequencies.nmd).')
    print('- QM vs MM dihedral profiles (if any) in "fragments" folder as ".pdf" files.')
