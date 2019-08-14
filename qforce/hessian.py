import scipy.optimize as optimize
#import scipy.linalg as la
#import scipy.optimize.nnls as nnls
from scipy.linalg import eigh
import numpy as np
from .molecule import Molecule
from .read_qm_out import QM
from .read_forcefield import Forcefield
from .write_forcefield import write_itp
from .forces import calc_bonds, calc_angles, calc_dihedrals #, calc_g96angles
from .elements import elements

def fit_hessian(inp):
    """
    Scope:
    ------
    Fit MD hessian to the QM hessian.

    TO DO:
    ------
    - Move calc_energy_forces to forces and clean it up
    - Include LJ, Coulomb flex dihed forces in the fitting as numbers
    """
    hessian, hes_for_freq = [], []
    qm = QM(fchk_file = inp.fchk_file)
    mol = Molecule(qm.coords, qm.atomids, inp)  
#    print(qm.esp, sum(qm.esp))
    qm.esp = round_average_charges(mol, qm.esp) # BUG IN ROUNDING?
#    print(qm.esp, sum(qm.esp))
    job_name = inp.fchk_file.split(".")[0]

    print("Calculating the MD hessian matrix elements...")
    full_hessian = calc_hessian(qm.coords, mol)
    qm_freq, qm_vec = calc_vibrational_frequencies(qm.hessian, qm)

    count = 0
    print("Fitting the MD hessian parameters to QM hessian values")
    for i in range(mol.n_atoms*3):
        for j in range(i+1):
            hes = (full_hessian[i,j] + full_hessian[j,i]) / 2
            if all([i == 0 for i in hes]) or np.abs(qm.hessian[count]) < 1e-1:
                qm.hessian = np.delete(qm.hessian, count)
                hes_for_freq.append(np.zeros(mol.n_terms))
                full_hessian[j,i] = full_hessian[i,j]
            else:
                count += 1
                hessian.append(hes)
                hes_for_freq.append(hes)
    print("Done!")

    fit = optimize.lsq_linear(hessian, qm.hessian, bounds = (0, np.inf)) #la.lstsq nnls
    fit = np.array(fit.x)

    hes_for_freq = np.sum(hes_for_freq * fit, axis=1)
    md_freq, md_vec = calc_vibrational_frequencies(hes_for_freq, qm)
    write_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, job_name)
    make_ff_params_from_fit(mol, fit, job_name, qm.atomids, qm.esp, inp.urey)

def calc_hessian(coords, mol):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.n_atoms, 3*mol.n_atoms, mol.n_terms))

    for a in range(mol.n_atoms):
        for xyz in range(3):
            coords[a][xyz] += 0.003
            _, f_plus = calc_energy_forces(coords, mol)
            coords[a][xyz] -= 0.006
            _, f_minus = calc_energy_forces(coords, mol)
            coords[a][xyz] += 0.003
            diff = - (f_plus - f_minus) / 0.006
            full_hessian[:,a*3 + xyz, :] = diff.reshape(3*mol.n_atoms, mol.n_terms)
    return full_hessian

def calc_energy_forces(coords, mol):
    """
    Scope:
    ------
    For each displacement, calculate the forces from all terms.
    
    Notes:
    -----
    Bit ugly, probably can move the the forces file more neatly
    Technically don't need energies at the moment.
    Should add pair interactions at some point.
    """
    energy = np.zeros(mol.n_terms)
    force = np.zeros((mol.n_atoms, 3, mol.n_terms))
    energy, force = calc_bonds(coords, mol.bonds, energy, force)
    energy, force = calc_bonds(coords, mol.angles.urey, energy, force)
    energy, force = calc_angles(coords, mol.angles, energy, force) #g96
    energy, force = calc_dihedrals(coords, mol.dihedrals.stiff, 
                                   mol.dihedrals.improper, energy, force)

#    energy, force = calc_pairs(coords, mol.pairs, energy, force)
    return energy, force

def calc_vibrational_frequencies(upper, qm):
    """
    Calculate the MD vibrational frequencies by diagonalizing its Hessian
    """
    const_amu = 1.6605389210e-27
    const_avogadro = 6.0221412900e+23
    const_speedoflight = 299792.458
    kj2j = 1e3
    ang2meter = 1e-10
    to_omega2 = kj2j/ang2meter**2/(const_avogadro*const_amu) # 1/s**2
    to_waveno = 1e-5/(2.0*np.pi*const_speedoflight) #cm-1

    e = elements()
    matrix = np.zeros((3*qm.n_atoms, 3*qm.n_atoms))
    count = 0

    for i in range(3*qm.n_atoms):
        for j in range(i+1):
            mass_i = e.mass[qm.atomids[int(np.floor(i/3))]]
            mass_j = e.mass[qm.atomids[int(np.floor(j/3))]]
            matrix[i,j] = upper[count]/np.sqrt(mass_i*mass_j)
            matrix[j,i] = matrix[i,j]
            count += 1
    val, vec = eigh(matrix)
    vec = np.reshape(np.transpose(vec),(3*qm.n_atoms,qm.n_atoms,3))[6:]

    for i in range(qm.n_atoms):
        vec[:,i,:] = vec[:,i,:] / np.sqrt(e.mass[qm.atomids[i]])

    freq = np.sqrt(val[6:] * to_omega2) * to_waveno
    return freq, vec

def write_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, job_name):
    """
    Scope:
    ------
    Create the following files for comparing QM reference to the generated
    MD frequencies/eigenvalues.
    
    Output:
    ------
    JOBNAME_qforce.freq : QM vs MD vibrational frequencies and eigenvectors
    JOBNAME_qforce.nmd : MD eigenvectors that can be played in VMD with:
                                vmd -e filename
    """
    e = elements()
    freq_file, nmd_file = f"{job_name}_qforce.freq", f"{job_name}_qforce.nmd"
    with open(freq_file, "w") as freq:
        freq.write(" mode  QM-Freq   MD-Freq     Diff.  %Error\n")
        for i, (q, m) in enumerate(zip(qm_freq, md_freq)):
            diff = q - m
            err = diff / q * 100
            freq.write(f"{i+7:>4}{q:>10.1f}{m:>10.1f}{diff:>10.1f}{err:>8.2f}\n")
        freq.write("\n\n         QM vectors              MD Vectors\n")
        freq.write(50*"=")
        for i, (qm1, md1) in enumerate(zip(qm_vec, md_vec)):
            freq.write(f"\nMode {i+7}\n")
            for qm2, md2 in zip(qm1, md1):
                freq.write("{:>8.3f}{:>8.3f}{:>8.3f}{:>10.3f}{:>8.3f}{:>8.3f}\n"
                           .format(*qm2, *md2))
    with open(nmd_file, "w") as nmd:
        nmd.write(f"nmwiz_load {nmd_file}\n")
        nmd.write(f"title {job_name}\n")
        nmd.write("names")
        for ids in qm.atomids:
            nmd.write(f" {e.sym[ids]}")
        nmd.write("\nresnames")
        for i in range(qm.n_atoms):
            nmd.write(" RES")
        nmd.write("\nresnums")
        for i in range(qm.n_atoms):
            nmd.write(" 1")
        nmd.write("\ncoordinates")
        for c in qm.coords:
            nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
        for i, m in  enumerate(md_vec):
            nmd.write(f"\nmode {i+7}")
            for c in m:
                nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
    print(f"QM vs MD vibrational frequencies can be found in: {freq_file}")
    print(f"MD vibrational modes (can be run in VMD) is located in: {nmd_file}")

def make_ff_params_from_fit(mol, fit, job_name, atomids, esp, urey):
    """
    Scope:
    -----
    Convert units, average over equivalent minima and prepare everything
    to be written as a forcefield file.
    """
    ff = Forcefield()
    e = elements()
    ff.mol_type = job_name
    ff.charges = esp
    mass = [round(e.mass[i],5) for i in atomids]
    atom_no = range(1, mol.n_atoms + 1)
    atoms = [e.sym[i] for i in atomids]

    for n, at, a, q, m in zip(atom_no, mol.types, atoms, ff.charges, mass):
        ff.atoms.append([n, at, 1, "MOL", a, n, q, m])
        
    for i, term in enumerate(mol.bonds.term_ids):
        atoms = [a+1 for a in mol.bonds.atoms[i]]
        param = fit[term] * 100
        equiv_terms = np.where(np.array(mol.bonds.term_ids)==term)
        minimum = np.array(mol.bonds.minima)[equiv_terms].mean() * 0.1
        ff.bonds.append(atoms + [1, minimum, param])
        
    for i, term in enumerate(mol.angles.term_ids):
        atoms = [a+1 for a in mol.angles.atoms[i]]
        param = fit[term]
        eq = np.where(np.array(mol.angles.term_ids)==term)
        minimum = np.degrees(np.array(mol.angles.minima)[eq].mean())
        ff.angles.append(atoms + [1, minimum, param])
        
    if urey:
        for i, term in enumerate(mol.angles.urey.term_ids):
            param = fit[term] * 100
            eq = np.where(np.array(mol.angles.urey.term_ids)==term)
            minimum = np.array(mol.angles.urey.minima)[eq].mean() * 0.1
            ff.angles[i][3] = 5
            ff.angles[i].extend([minimum, param])
    
    for i, term in enumerate(mol.dihedrals.stiff.term_ids):
        atoms = [a+1 for a in mol.dihedrals.stiff.atoms[i]]
        param = fit[term]
        eq = np.where(np.array(mol.dihedrals.stiff.term_ids)==term)
        minimum = np.degrees(np.array(mol.dihedrals.stiff.minima)[eq].mean())
        ff.dihedrals.append(atoms + [2, minimum, param])

    for i, term in enumerate(mol.dihedrals.improper.term_ids):
        atoms = [a+1 for a in mol.dihedrals.improper.atoms[i]]
        param = fit[term]
        eq = np.where(np.array(mol.dihedrals.improper.term_ids)==term)
        minimum = np.degrees(np.array(mol.dihedrals.improper.minima)[eq].mean())
        ff.impropers.append(atoms + [2, minimum, param])

    for i, term in enumerate(mol.dihedrals.flexible.term_ids):
        atoms = [a+1 for a in mol.dihedrals.flexible.atoms[i]]
        
        eq = np.where(np.array(mol.dihedrals.flexible.term_ids)==term)
        minimum = np.degrees(np.array(mol.dihedrals.flexible.minima)[eq].mean())
        ff.flexible.append(atoms + [3, minimum])

    out_file = f"{job_name}_qforce.itp"
    write_itp(ff, out_file, urey)
    print(f"Q-Force parameters can be found in: {out_file}")

def round_average_charges(mol, esp):
    charges, n_eq = [], []
    for l in mol.list:
        c = 0
        n_eq.append(len(l))
        for a in l:
            c += esp[a]
        charges.append(round(c/n_eq[-1], 5))

    sum_c = sum([charges[i]*n_eq[i] for i in range(mol.n_types)])
    extra = 100000 * round(sum_c - round(sum_c,0),5)
    min_eq = min(n_eq)
    min_ind = n_eq.index(min_eq)
    div, rem = divmod(extra, min_eq)
    charges[min_ind] -= div / 100000

    rounded = [charges[i] for i in mol.atoms]
    rounded -= rem / 100000
    return rounded
