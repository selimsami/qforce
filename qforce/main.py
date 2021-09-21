import sys

from .polarize import polarize
from .initialize import initialize
from .qm.qm import QM
from .qm.qm_base import HessianOutput
from .forcefield import ForceField
from .molecule import Molecule
from .fragment import fragment
from .dihedral_scan import DihedralScan
from .frequencies import calc_qm_vs_md_frequencies
from .hessian import fit_hessian, fit_hessian_nl


def run_qforce(input_arg, ext_q=None, ext_lj=None, config=None, presets=None):
    #### Initialization phase ####
    print('\n#### INITIALIZATION PHASE ####\n')
    config, job = initialize(input_arg, config, presets)
    print('Config:')
    print(config, '\n')
    print('Job:')
    print(job, '\n')

    check_wellposedness(config)

    check_continue(config)

    #### Polarization phase ####
    print('\n#### POLARIZATION PHASE ####\n')
    if config.ff._polarize:
        polarize(job, config.ff)

    check_continue(config)

    #### QM phase ####
    print('\n#### QM PHASE ####\n')
    qm = QM(job, config.qm)
    qm_hessian_out = qm.read_hessian()

    check_continue(config)

    #### Molecule phase ####
    print('\n#### MOLECULE PHASE ####\n')
    mol = Molecule(config, job, qm_hessian_out, ext_q, ext_lj)

    check_continue(config)

    #### Hessian fitting phase ####
    print('\n#### HESSIAN FITTING PHASE ####\n')
    md_hessian = None
    if config.general.opt_type == 'linear':
        md_hessian = fit_hessian(config.terms, mol, qm_hessian_out, config.general.opt_iter, config.general.opt_verbose)
    elif config.general.opt_type == 'non_linear':
        md_hessian = fit_hessian_nl(config.terms, mol, qm_hessian_out, config.general.opt_verbose)

    check_continue(config)

    #### Flexible dihedral scan phase ####
    if config.ff.scan_dihedrals:
        print('\n#### FLEXIBLE DIHEDRAL SCAN PHASE ####\n')
        if len(mol.terms['dihedral/flexible']) > 0 and config.scan.do_scan:
            fragments = fragment(mol, qm, job, config)
            DihedralScan(fragments, mol, job, config)

        check_continue(config)

    #### Calculate frequencies phase ####
    print('\n#### CALCULATE FREQUENCIES PHASE ####\n')
    calc_qm_vs_md_frequencies(job, qm_hessian_out, md_hessian)

    check_continue(config)

    #### Calculate Force Field phase ####
    if config.ff.compute_ff:
        print('\n#### CALCULATE FORCE FIELD PHASE ####\n')
        ff = ForceField(job.name, config, mol, mol.topo.neighbors)
        ff.write_gromacs(job.dir, mol, qm_hessian_out.coords)

        print_outcome(job.dir)
    else:
        print('\nNo Force Field output requested...\n')
        print('RUN COMPLETED')


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


def check_wellposedness(config):
    if config.general.opt_type == 'linear' and (config.terms.morse or config.terms.morse_mp):
        raise Exception('Linear optimization is not valid for Morse bond potential')
    else:
        print('Configuration is valid!')


def check_continue(config):
    if config.general.debug_mode:
        x = input('\nDo you want to continue y/n? ')
        if x not in ['yes', 'y']:
            print()
            sys.exit(0)