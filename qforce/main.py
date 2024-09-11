from calkeeper import CalculationKeeper, CalculationIncompleteError

from .initialize import initialize
from .qm.qm import QM
from .qm.qm_base import HessianOutput
from .forcefield.forcefield import ForceField
from .molecule import Molecule
from .fragment import fragment
from .no_fragment_scanning import do_nofrag_scanning
from .dihedral_scan import DihedralScan
from .frequencies import calc_qm_vs_md_frequencies
from .hessian import fit_hessian, multi_hessian_fit, multi_hessian_newfit
from .charge_flux import fit_charge_flux
from .misc import LOGO
from .logger import LoggerExit
from .additionalstructures import AdditionalStructures


def runjob(config, job, ext_q=None, ext_lj=None):
    # setup qm calculation
    qm = QM(job, config.qm)
    # do the preoptimization if selected
    qm.preopt()
    # get hessian output
    qm_hessian_out, qm_energy_out, qm_gradient_out = qm.get_hessian()
    main_hessian = qm_hessian_out[0]
        
    structs = AdditionalStructures.from_config(config.addstructs)
    structs.create(qm)

    # check molecule
    ffcls = ForceField.implemented_md_software.get(config.ff.output_software, None)
    if ffcls is None:
        raise ValueError(f"Forcefield '{config.ff.output_software}' unknown!")
    mol = Molecule(config, job, main_hessian, ffcls, ext_q, ext_lj)

    # change the order
    fragments = None
    if len(mol.terms['dihedral/flexible']) > 0 and config.scan.do_scan:
        # get fragments with qm
        fragments = fragment(mol, qm, job, config)

    # hessian fitting
    md_hessian = multi_hessian_fit(job.logger, config.terms, mol, qm_hessian_out, qm_energy_out, qm_gradient_out, fit_flexible=False)

    # do the scans
    if fragments is not None:
        DihedralScan(fragments, mol, job, config)

    calc_qm_vs_md_frequencies(job, main_hessian, md_hessian)

    ff = ForceField(config.ff.output_software, job, config, mol, mol.topo.neighbors)
    ff.software.write(job.dir, main_hessian.coords)

    print_outcome(job.logger, job.dir, config.ff.output_software)

    return mol


def runjob_v2(config, job, ext_q=None, ext_lj=None):
    # setup qm calculation
    qm = QM(job, config.qm)
    # do the preoptimization if selected
    qm.preopt()
    # get hessian output
    qm_hessian_out, _, _ = qm.get_hessian()
    main_hessian = qm_hessian_out[0]

    structs = AdditionalStructures.from_config(config.addstructs)
    structs.create(qm)
    # add hessian
    structs.add_hessian(main_hessian)

    ffcls = ForceField.implemented_md_software.get(config.ff.output_software, None)
    if ffcls is None:
        raise ValueError(f"Forcefield '{config.ff.output_software}' unknown!")

    # check molecule
    mol = Molecule(config, job, main_hessian, ffcls, ext_q, ext_lj, fit_flexible=True)

    # if len(mol.terms['dihedral/flexible']) > 0:
    scans = do_nofrag_scanning(mol, qm, job, config)
    # add forces
    structs.add_dihedrals(scans)
    # normalize!
    structs.normalize()
    # hessian fitting
    md_hessian = multi_hessian_newfit(job.logger, config.terms, mol, structs, fit_flexible=True)

    calc_qm_vs_md_frequencies(job, main_hessian, md_hessian)

    if main_hessian.dipole_deriv is not None and 'charge_flux' in mol.terms and len(mol.terms['charge_flux']) > 0:
        fit_charge_flux(main_hessian, qm_energy_out, qm_gradient_out, mol)

    ff = ForceField(config.ff.output_software, job, config, mol, mol.topo.neighbors)
    ff.software.write(job.dir, main_hessian.coords)

    print_outcome(job.logger, job.dir, config.ff.output_software)

    return mol


def save_jobs(job):
    with open(job.pathways['calculations.json'], 'w') as fh:
        fh.write(job.calkeeper.as_json())


def runspjob(config, job, ext_q=None, ext_lj=None, v2=False):
    """Run a single round of Q-Force"""
    # print qforce logo
    job.logger.info(LOGO)
    #
    try:
        if v2:
            mol = runjob_v2(config, job, ext_q=ext_q, ext_lj=ext_lj)
        else:
            mol = runjob(config, job, ext_q=ext_q, ext_lj=ext_lj)
        save_jobs(job)
        return mol
    except CalculationIncompleteError:
        save_jobs(job)
    except LoggerExit as err:
        save_jobs(job)
        if not job.logger.isstdout:
            job.logger.info(str(err))
        raise err from None
    return None


def run_qforce(input_arg, ext_q=None, ext_lj=None, config=None, presets=None, err=False, v2=True):
    """Execute Qforce from python directly """
    config, job = initialize(input_arg, config, presets)
    #
    if err is True:
        return runspjob(config, job, ext_q=ext_q, ext_lj=ext_lj, v2=v2)
    else:
        try:
            return runspjob(config, job, ext_q=ext_q, ext_lj=ext_lj, v2=v2)
        except LoggerExit as err:
            print(str(err))


def run_hessian_fitting_for_external(job_dir, qm_data, ext_q=None, ext_lj=None,
                                     config=None, presets=None):
    config, job = initialize(job_dir, config, presets)

    qm_hessian_out = HessianOutput(config.qm.vib_scaling, **qm_data)

    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    md_hessian = fit_hessian(job.logger, config.terms, mol, qm_hessian_out)
    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)

    ff = ForceField(config.ff.output_software, job.name, config, mol, mol.topo.neighbors)
    ff.software.write(job.dir, qm_hessian_out.coords)

    print_outcome(job.logger, job.dir, config.ff.output_software)

    return mol.terms


def print_outcome(logger, job_dir, output_software):
    logger.info(f'Output files can be found in the directory: {job_dir}.')
    logger.info(f'- Q-Force force field parameters in {output_software.upper()} format.')
    logger.info('- QM vs MM vibrational frequencies, pre-dihedral fitting (frequencies.txt,'
                ' frequencies.pdf).')
    logger.info('- Vibrational modes which can be visualized in VMD (frequencies.nmd).')
    logger.info('- QM vs MM dihedral profiles (if any) in "fragments" folder as ".pdf" files.\n')

