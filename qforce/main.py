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
from .schemes.bondorder import BondOrderCreator
from .schemes import NoFragmentationDihedralsCreator


def save_molecule(filename, molecule, comment=None):
    write(filename, molecule, plain=True, comment=comment)


def runjob_(config, job, ext_q=None, ext_lj=None):
    # setup the qm interface
    qm = QM(job, config.qm)
    # read initial structure, ase molecule structure
    molecule = read(job.coord_file)
    # run preopt
    folder = job.pathways.getdir('preopt', create=True)
    crest = CrestCreator(folder, molecule)
    crest.run(qm)
    molecule.set_positions(crest.most_stable())
    # setup hessian calculation
    save_molecule(qm.pathways.get('init.xyz'), molecule,
                  comment=f'{job.name} - input geometry for hessian')
    # get the bondorder to setup molecule
    bo = BondOrderCreator(molecule)
    bo.run(qm)
    bondorder = bo.bondorder()
    #
    ffcls = ForceField.implemented_md_software.get(config.ff.output_software, None)
    if ffcls is None:
        raise ValueError(f"Forcefield '{config.ff.output_software}' unknown!")

    mol = Molecule(config, job, bondorder, ffcls, ext_q, ext_lj)
    #
    structs = Computations.from_config(config.addstructs)
    structs.register('dihedrals', NoFragmentationDihedralsCreator(mol, job, config))
    structs.register('hessian', HessianCreator(molecule))
    #
    structs.activate('fromfile')
    structs.activate('xtbmd', crest.structures())
    # do all additional calculations
    structs.run(qm)
    #  register hessian after structs were run!
    # structs.register('hessian', hessian)
    #
    mol.qm_minimum_energy, mol.qm_minimum_coords = structs.normalize()
    #
    md_hessian = multi_fit(job.logger, config.terms, mol, structs)
    #
    return bondorder, md_hessian, mol, structs


def runjob(config, job, ext_q=None, ext_lj=None):
    main_hessian, md_hessian, mol, structs = runjob_(config, job, ext_q=ext_q, ext_lj=ext_lj)

    calc_qm_vs_md_frequencies(job, main_hessian, md_hessian)

    if (main_hessian.dipole_deriv is not None
       and 'charge_flux' in mol.terms
       and len(mol.terms['charge_flux']) > 0):
        raise NotImplementedError("Charge flux is not updated to new syntax")
        # fit_charge_flux(main_hessian, qm_energy_out, qm_gradient_out, mol)

    ff = ForceField(config.ff.output_software, job, config, mol, mol.topo.neighbors)
    ff.software.write(job.dir, main_hessian.coords)

    print_outcome(job.logger, job.dir, config.ff.output_software)

    return mol


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
