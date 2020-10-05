from .forcefield import ForceField
from .molecule import Molecule
from .fragment import fragment
from .dihedral_scan import DihedralScan
from .frequencies import calc_qm_vs_md_frequencies
from .hessian import fit_hessian


def fit(qm, config, job):
    qm_out = qm.read_hessian()
    mol = Molecule(config, job, qm_out)

    fit_results, md_hessian = fit_hessian(config.terms, mol, qm_out)

    if config.scan.do_scan:
        fragments = fragment(mol, qm, job, config.scan)
        DihedralScan(fragments, mol, job, config)

    calc_qm_vs_md_frequencies(job, qm_out, md_hessian)
    ff = ForceField(job.name, config, mol, mol.topo.neighbors)
    ff.write_gromacs(job.dir, mol, qm_out.coords)

    print(f'\nOutput files can be found in the directory: {job.dir}.')
    print('- Q-Force force field parameters in GROMACS format (gas.gro, gas.itp, gas.top).')
    print('- QM vs MM vibrational frequencies (frequencies.txt, frequencies.pdf).')
    print('- Vibrational modes which can be visualized in VMD (frequencies.nmd).')
    print('- QM vs MM dihedral profiles (if any) in "fragments" folder as ".pdf" files.')
