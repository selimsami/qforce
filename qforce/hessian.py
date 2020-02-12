import scipy.optimize as optimize
# import scipy.linalg as la
# import scipy.optimize.nnls as nnls
import numpy as np
from .read_qm_out import QM
from .read_forcefield import Forcefield
from .write_forcefield import write_ff
from .dihedral_scan import scan_dihedral
from .molecule import Molecule
from .fragment import fragment
# , calc_g96angles
from .elements import elements
from .frequencies import calc_qm_vs_md_frequencies
# from .decorators import timeit, print_timelog
from ase.optimize import BFGS
from ase import Atoms
from ase.io import write
from .calculator import QForce


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

    # Fit - add dihedrals - fit again >> Is it enough? More iteration?
    if not inp.nofrag:
        fragment(inp, mol, qm)
        fit_results, md_hessian = fit_hessian(inp, mol, qm, ignore_flex=False)

    molecule = Atoms(qm.atomids, positions=qm.coords, calculator=QForce(mol.terms))

    dyn = BFGS(molecule)
    dyn.run(fmax=0.005, )

    # print(molecule.get_positions())
    # write('optimized.xyz', molecule, plain=True)

    #
    #
    calc_qm_vs_md_frequencies(inp, qm, md_hessian)
    average_unique_minima(mol.terms)
    make_ff_params_from_fit(mol, fit_results, inp, qm)
    #
    #
    # temporary
    # fit_dihedrals(inp, mol, qm)


def fit_hessian(inp, mol, qm, ignore_flex=True):
    hessian, full_md_hessian_1d = [], []
    non_fit = []
    qm_hessian = np.copy(qm.hessian)

    print("Calculating the MD hessian matrix elements...")
    full_md_hessian = calc_hessian(qm.coords, mol, inp, ignore_flex)

    count = 0
    print("Fitting the MD hessian parameters to QM hessian values")
    for i in range(mol.topo.n_atoms*3):
        for j in range(i+1):
            hes = (full_md_hessian[i, j] + full_md_hessian[j, i]) / 2
            if all([h == 0 for h in hes]) or np.abs(qm_hessian[count]) < 1e+1:
                qm_hessian = np.delete(qm_hessian, count)
                full_md_hessian_1d.append(np.zeros(mol.terms.n_fitted_terms))
            else:
                count += 1
                hessian.append(hes[:-1])
                full_md_hessian_1d.append(hes[:-1])
                non_fit.append(hes[-1])
    print("Done!\n")

    difference = qm_hessian - np.array(non_fit)
    # la.lstsq or nnls could also be used:
    fit = optimize.lsq_linear(hessian, difference, bounds=(0, np.inf)).x

    for term in mol.terms:
        if term.idx < len(fit):
            term.fconst = fit[term.idx]

    full_md_hessian_1d = np.sum(full_md_hessian_1d * fit, axis=1)

    return fit, full_md_hessian_1d


def fit_dihedrals(inp, mol, qm):
    """
    Temporary - to be removed
    """

    from .fragment import check_one_fragment
    for term in mol.terms['dihedral/flexible']:
        frag_name, _, _, _ = check_one_fragment(inp, mol.topo, term.atomids)
        scan_dihedral(inp, term.atomids, frag_name)


def calc_hessian(coords, mol, inp, ignore_flex):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms,
                             mol.terms.n_fitted_terms+1))

    for a in range(mol.topo.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += 0.003
            f_plus = calc_forces(coords, mol, inp, ignore_flex)
            coords[a][xyz] -= 0.006
            f_minus = calc_forces(coords, mol, inp, ignore_flex)
            coords[a][xyz] += 0.003
            diff = - (f_plus - f_minus) / 0.006
            full_hessian[a*3+xyz] = diff.reshape(mol.terms.n_fitted_terms+1, 3*mol.topo.n_atoms).T
    return full_hessian


def calc_forces(coords, mol, inp, ignore_flex):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.

    """
    if ignore_flex:
        ignores = ['dihedral/flexible', 'dihedral/constr']
    else:
        ignores = []

    force = np.zeros((mol.terms.n_fitted_terms+1, mol.topo.n_atoms, 3))

    with mol.terms.add_ignore(ignores):
        for term in mol.terms:
            term.do_fitting(coords, force)

    return force


def average_unique_minima(terms):
    unique_terms = {}
    averaged_terms = ['bond', 'angle', 'urey']
    for name in [term_name for term_name in averaged_terms if term_name in terms.term_names]:
        for term in terms[name]:
            if str(term) in unique_terms.keys():
                term.equ = unique_terms[str(term)]
            else:
                eq = np.where(np.array(list(oterm.idx for oterm in terms[name])) == term.idx)
                minimum = np.array(list(oterm.equ for oterm in terms[name]))[eq].mean()
                term.equ = minimum
                unique_terms[str(term)] = minimum


def make_ff_params_from_fit(mol, fit, inp, qm, polar=False):
    """
    Scope:
    -----
    Convert units, average over equivalent minima and prepare everything
    to be written as a forcefield file.
    """
    ff = Forcefield()
    e = elements()
    bohr2nm = 0.052917721067
    ff.mol_type = inp.job_name
    ff.natom = mol.topo.n_atoms
    ff.box = [10., 10., 10.]
    ff.n_mol = 1
    ff.coords = list(qm.coords/10)
    mass = [round(e.mass[i], 5) for i in qm.atomids]
    atom_no = range(1, mol.topo.n_atoms + 1)
    atoms = []
    atom_dict = {}

    for i, a in enumerate(qm.atomids):
        sym = e.sym[a]
        if sym not in atom_dict:
            atom_dict[sym] = 1
        else:
            atom_dict[sym] += 1
        atoms.append(f'{sym}{atom_dict[sym]}')

    for i, (sigma, epsilon) in enumerate(zip(mol.topo.sigma, mol.topo.epsilon)):
        unique = mol.topo.types[mol.topo.list[i][0]]
        ff.atom_types.append([unique, 0, 0, "A", sigma*0.1, epsilon])

    for n, at, a_uniq, a, m in zip(atom_no, mol.topo.types, mol.topo.atoms, atoms, mass):
        ff.atoms.append([n, at, 1, "MOL", a, n, mol.topo.q[a_uniq], m])

    if polar:
        alphas = qm.alpha*bohr2nm**3
        drude = {}
        n_drude = 1
        ff.atom_types.append(["DP", 0, 0, "S", 0, 0])

        for i, alpha in enumerate(alphas):
            if alpha > 0:
                drude[i] = mol.topo.n_atoms+n_drude
                ff.atoms[i][6] += 8
                # drude atoms
                ff.atoms.append([drude[i], 'DP', 2, 'MOL', f'D{atoms[i]}',
                                 i+1, -8., 0.])
                ff.coords.append(ff.coords[i])
                # polarizability
                ff.polar.append([i+1, drude[i], 1, alpha])
                n_drude += 1
        ff.exclu = [[] for _ in ff.atoms]
        ff.natom = len(ff.atoms)
        for i, alpha in enumerate(alphas):
            if alpha > 0:
                # exclusions for balancing the drude particles
                for j in mol.topo.neighbors[inp.nrexcl-2][i]+mol.topo.neighbors[inp.nrexcl-1][i]:
                    if alphas[j] > 0:
                        ff.exclu[drude[i]-1].extend([drude[j]])
                for j in mol.topo.neighbors[inp.nrexcl-1][i]:
                    ff.exclu[drude[i]-1].extend([j+1])
                ff.exclu[drude[i]-1].sort()
                # thole polarizability
                for neigh in [mol.topo.neighbors[n][i] for n in range(inp.nrexcl)]:
                    for j in neigh:
                        if i < j and alphas[j] > 0:
                            ff.thole.append([i+1, drude[i], j+1, drude[j], "2", 2.6, alpha,
                                             alphas[j]])

    for term in mol.terms['bond']:
        atoms = [a+1 for a in term.atomids]
        ff.bonds.append(atoms + [1, term.equ*0.1, term.fconst*100])

    for term in mol.terms['angle']:
        atoms = [a+1 for a in term.atomids]
        ff.angles.append(atoms + [1, np.degrees(term.equ), term.fconst])

    if inp.urey:
        corresp_angles = np.array([angle[0:3:2] for angle in ff.angles])
        for term in mol.terms['urey']:
            match = np.all(corresp_angles == term.atomids+1, axis=1)
            match = np.nonzero(match)[0][0]
            ff.angles[match][3] = 5
            ff.angles[match].extend([term.equ*0.1, term.fconst*100])

    for term in mol.terms['dihedral/rigid']:
        atoms = [a+1 for a in term.atomids]
        minimum = np.degrees(term.equ)
        ff.dihedrals.append(atoms + [2, minimum, term.fconst])

    for term in mol.terms['dihedral/improper']:
        atoms = [a+1 for a in term.atomids]
        minimum = np.degrees(term.equ)
        ff.impropers.append(atoms + [2, minimum, term.fconst])

    for term in mol.terms['dihedral/flexible']:
        atoms = [a+1 for a in term.atomids]
        if inp.nofrag:
            ff.flexible.append(atoms + [3, term.idx+1])
        else:
            ff.flexible.append(atoms + [3] + list(term.equ))

    for term in mol.terms['dihedral/constr']:
        atoms = [a+1 for a in term.atomids]
        ff.constrained.append(atoms + [3, term.idx+1])

    write_ff(ff, inp, polar)
    print("Q-Force force field parameters (.itp, .top) can be found in the "
          f"directory: {inp.job_dir}/")
