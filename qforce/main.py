from ase.io import read, write
from calkeeper import CalculationKeeper, CalculationIncompleteError
#
from .initialize import initialize
from .qm.qm import QM
from .forcefield.forcefield import ForceField
from .molecule import Molecule
from .frequencies import calc_qm_vs_md_frequencies
from .fit import multi_fit
# from .charge_flux import fit_charge_flux
from .misc import LOGO
from .logger import LoggerExit
#
from .schemes import Computations, HessianCreator, CrestCreator
from .schemes import DihedralCreator


def runjob(config, job, ext_q=None, ext_lj=None):

    qm_interface = QM(job, config.qm)
    ff_interface = ForceField.implemented_md_software[config.ff.output_software]

    mol = Molecule(job, config)

    do_crest(job, qm_interface, mol)

    # This is ideally temporary if we can fix the global optimization
    hessian_out = do_hessian(qm_interface, mol)

    mol.setup(config, job, ff_interface, hessian_out, ext_q, ext_lj)

    structs = do_all_structs(job, config, qm_interface, mol)

    md_hessian = multi_fit(job.logger, config.terms, mol, structs)
    calc_qm_vs_md_frequencies(job, hessian_out, md_hessian)

    if (hessian_out.dipole_deriv is not None
       and 'charge_flux' in mol.terms
       and len(mol.terms['charge_flux']) > 0):
        raise NotImplementedError("Charge flux is not updated to new syntax")
        # fit_charge_flux(main_hessian, qm_energy_out, qm_gradient_out, mol)

    ff = ForceField(config.ff.output_software, job, config, mol, mol.topo.neighbors)
    ff.software.write(job.dir, mol.coords)

    print_outcome(job.logger, job.dir, config.ff.output_software)

    return mol


def do_hessian(qm_interface, mol):
    hessian = HessianCreator(mol)
    hessian.run(qm_interface)
    main_hessian = hessian.main_hessian()
    mol.update_coords(main_hessian.coords, 'Structure optimized for Hessian calculation')
    return main_hessian


def do_crest(job, qm_interface, mol):
    folder = job.pathways.getdir('preopt', create=True)
    crest = CrestCreator(folder, mol)
    crest.run(qm_interface)
    mol.update_coords(crest.get_most_stable(), 'CREST lowest energy structure')
    mol.all_coords = crest.get_structures()
    mol.bond_orders = crest.get_bond_orders()


def do_all_structs(job, config, qm_interface, mol):
    folder = job.pathways.jobdir

    structs = Computations(config.addstructs, folder)
    structs.register('dihedrals', DihedralCreator(mol, job, config))
    structs.register('hessian', HessianCreator(mol))
    #
    structs.activate('fromfile')
    structs.activate('xtbmd', mol.all_coords)
    # do all additional calculations
    structs.run(qm_interface)
    #  register hessian after structs were run!
    # structs.register('hessian', hessian)
    #
    mol.qm_minimum_energy, mol.qm_minimum_coords = structs.normalize()
    return structs


def load_keeper(job):
    file = job.pathways['calculations.json']
    if file.exists():
        with open(file, 'r') as fh:
            keeper = CalculationKeeper.from_json(fh.read())
        return keeper
    raise SystemExit(f"No calculation for '{job.dir}'")


def write_bashscript(filename, config, job):
    methods = {name: calculator.as_string for name, calculator in job.calculators.items()}
    ncores = config.qm.n_proc
    keeper = load_keeper(job)

    with open(filename, 'w') as fh:
        fh.write("current=$PWD\n")
        for calc in keeper.get_incomplete():
            call = methods.get(calc.software, None)
            if call is None:
                raise ValueError("Call unknown!")
            fh.write(call(calc, ncores))


def save_jobs(config, job):
    with open(job.pathways['calculations.json'], 'w') as fh:
        fh.write(job.calkeeper.as_json())

    if config.logging.write_bash is True:
        write_bashscript(f'run_{job.name}_qforce.sh', config, job)


def runspjob(config, job, ext_q=None, ext_lj=None):
    """Run a single round of Q-Force"""
    # print qforce logo
    job.logger.info(LOGO)
    #
    try:
        mol = runjob(config, job, ext_q=ext_q, ext_lj=ext_lj)
        save_jobs(config, job)
        return mol
    except CalculationIncompleteError:
        save_jobs(config, job)
    except LoggerExit as err:
        save_jobs(config, job)
        if not job.logger.isstdout:
            job.logger.info(str(err))
        raise err from None
    return None


def run_qforce(input_arg, ext_q=None, ext_lj=None, config=None, presets=None, err=False):
    """Execute Qforce from python directly """
    config, job = initialize(input_arg, config, presets)
    #
    if err is True:
        return runspjob(config, job, ext_q=ext_q, ext_lj=ext_lj)
    else:
        try:
            return runspjob(config, job, ext_q=ext_q, ext_lj=ext_lj)
        except LoggerExit as err:
            print(str(err))


def print_outcome(logger, job_dir, output_software):
    logger.info(f'Output files can be found in the directory: {job_dir}.')
    logger.info(f'- Q-Force force field parameters in {output_software.upper()} format.')
    logger.info('- QM vs MM vibrational frequencies, pre-dihedral fitting (frequencies.txt,'
                ' frequencies.pdf).')
    logger.info('- Vibrational modes which can be visualized in VMD (frequencies.nmd).')
    logger.info('- QM vs MM dihedral profiles (if any) in "fragments" folder as ".pdf" files.\n')
