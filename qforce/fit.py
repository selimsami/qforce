from .read_qm_out import QM
from .forcefield import ForceField
from .molecule import Molecule
from .fragment import fragment
from .dihedral_scan import scan_dihedrals
from .frequencies import calc_qm_vs_md_frequencies
from .hessian import fit_hessian


def fit_forcefield(inp, qm=None, mol=None):
    """
    Scope:
    ------
    Fit MD hessian to the QM hessian.

    TO DO:
    ------
    - Move calc_energy_forces to forces and clean it up
    - Include LJ, Coulomb flex dihed forces in the fitting as numbers

    CHECK
    -----
    - Does having (0,inf) fitting bound cause problems? metpyrl lower accuracy
      for dihed! Having -inf, inf causes problems for PTEG-1 (super high FKs)
    - Fix acetone angle! bond-angle coupling?)
    - Charges from IR intensities - together with interacting polarizable FF?
    """

    qm = QM(inp, "freq", fchk_file=inp.fchk_file, out_file=inp.qm_freq_out)

    mol = Molecule(inp, qm)

    fit_results, md_hessian = fit_hessian(inp, mol, qm, ignore_flex=True)

    if inp.fragment:
        fragments = fragment(inp, mol)
        scan_dihedrals(fragments, inp, mol)

    calc_qm_vs_md_frequencies(inp, qm, md_hessian)
    ff = ForceField(inp, mol, mol.topo.neighbors)
    ff.write_gromacs(inp, mol, inp.job_dir, qm.coords)

    print(f'\nOutput files can be found in the directory: {inp.job_dir}.')
    print('- Q-Force force field parameters in GROMACS format (gas.gro, gas.itp, gas.top).')
    print('- QM vs MM vibrational frequencies (frequencies.txt, frequencies.pdf).')
    print('- Vibrational modes which can be visualized in VMD (frequencies.nmd).')
    print('- QM vs MM dihedral profiles (if any) in "fragments" folder as ".pdf" files.')
