import os
import numpy as np
from .read_qm_out import QM
from .elements import elements
from .molecule import Molecule
from symfit import Fit, parameters, variables, Equality
import matplotlib.pyplot as plt

def fit_dipoles(inp):
    """
    Scope:
    -----
    - Read coordinates and dipoles from multiple single point QM calculations
    - Fit MD charges to best match the multi-configuration QM dipoles
    - Write the charges to an ITP file
    - Plot the QM vs MD dipole correlation as a PDF
    """
    #get a list of files that end with ".log" or ".out"
    job_name = inp.traj_dir.split("_")[0]
    files = os.listdir(inp.traj_dir)    
    files = [f"{inp.traj_dir}/{file}" for file in files if (
             ".log" in file or ".out" in file)]
    
    #read all outputs and check if any failed
    qm = QM(out_files = files, job_type = "traj")   
    
    mol = Molecule(qm.coords[0], qm.atomids, inp)
    
    print(f"There are {len(mol.list)} unique atoms.")
    
    qm_x, qm_y, qm_z, x, y, z= variables('qm_x, qm_y, qm_z, x, y, z')
    xcoord = variables(f', '.join(f'x_{i}' for i in range(qm.natoms)))
    ycoord = variables(f', '.join(f'y_{i}' for i in range(qm.natoms)))
    zcoord = variables(f', '.join(f'z_{i}' for i in range(qm.natoms)))
    charge = parameters(f', '.join(f'q_{i}' for i in mol.atoms))

    for c, init in zip(charge, qm.init_charges):
        c.value = init
        if inp.fit_percent >= 0:
            c.min = init - inp.fit_percent * abs(init)
            c.max = init + inp.fit_percent * abs(init)

    sum_charge, x, y, z = 0, 0, 0, 0
    
    dipoles = qm.dipoles * 0.208194342

    for i in range(qm.natoms):
        sum_charge = sum_charge + charge[i]

    for xc, yc, zc, q in zip(xcoord, ycoord, zcoord, charge):
        x = x + xc * q
        y = y + yc * q
        z = z + zc * q

    model = {qm_x: x, qm_y: y, qm_z: z}
    print("Fitting started...")
    fit = Fit(model, 
              qm_x = dipoles[:,0], qm_y = dipoles[:,1], qm_z = dipoles[:,2], 
            **{f'x_{i}': qm.coords[:,i,0] for i in range(qm.natoms)},
            **{f'y_{i}': qm.coords[:,i,1] for i in range(qm.natoms)},
            **{f'z_{i}': qm.coords[:,i,2] for i in range(qm.natoms)},
              constraints = [Equality(sum_charge, qm.charge)])

    fit_result = fit.execute()
    print(f"Done! R-squared for the fitting is: {fit_result.r_squared:<10.4f}")
#    print(fit_result)
    q_fitted = round_to_integer(fit_result.params, sum_charge, mol, qm.charge)
#    print(q_fitted)

    fitted_dipole = fit.model(
            **{f'x_{i}': qm.coords[:,i,0] for i in range(qm.natoms)},
            **{f'y_{i}': qm.coords[:,i,1] for i in range(qm.natoms)},
            **{f'z_{i}': qm.coords[:,i,2] for i in range(qm.natoms)},
            **fit_result.params)

    fitted_dipole = np.array(fitted_dipole) / 0.208194342
    fitted_dipole = np.append(fitted_dipole, [np.sqrt(fitted_dipole[0]**2 + 
                                                      fitted_dipole[1]**2 + 
                                                      fitted_dipole[2]**2)],
                                                      axis = 0)
    write_gmx_atoms(qm, q_fitted, job_name, mol)
    plot_dipole_correlation(qm.dipoles, fitted_dipole, job_name)

def round_to_integer(params, sum_charge, mol, qm_charge):
    n_equiv = [len(i) for i in mol.list]
    charges = [round(i,4) for i in list(params.values())]
    extra = round(sum_charge(*charges),4) - qm_charge
    abs_c = [abs(i) for i in charges]
    sum_c = sum_charge(*charges)
    test_c = charges
    while abs(sum_c) > 1e-5:
        largest = abs_c.index(max(abs_c))
        if abs_c[largest] == 0:
            print("Could not round the total charge to integer. "
                  "Check manually")
            break
        test_c = charges.copy()
        test_c[largest] = round(test_c[largest] - extra / n_equiv[largest],4)
        sum_c = sum_charge(*test_c)
        abs_c[largest] = 0
    else:
        charges = test_c
    return charges

def write_gmx_atoms(qm, q_fitted, job_name, mol):
    all_q = [q_fitted[i] for i in mol.atoms]
    e = elements()
    atom_no = range(1, qm.natoms + 1)
    mass = [round(e.mass[i],4) for i in qm.atomids]
    atoms = [e.sym[i] for i in qm.atomids]
    with open(f"{job_name}_atoms.itp", "w") as gmx:
        gmx.write("[atoms]\n")
        gmx.write(";   nr  type resnr   res  atom"
                  "  cgnr      charge        mass\n")
        for n, at, a, q, m in zip(atom_no, mol.types, atoms, all_q, mass):
            gmx.write("{:>6}{:>6}{:>6}{:>6}{:>6}{:>6}{:>12.4f}{:>12.4f}\n"
                      .format(n, at, 1, "LIG", a, n, q, m))
    print(f"Fitted charges are written to: {job_name}_atoms.itp")

def plot_dipole_correlation(qm, fitted, file):
    titles = ["x dipole", "y dipole", "z dipole", "total dipole"]
    f, axs = plt.subplots(2, 2)
    
    for i in range(4):
        x, y = "{0:02b}".format(i)
        x, y = int(x), int(y)
        y_min = min(qm[:,i])
        x_min = min(fitted[i])
        minim = min(y_min, x_min)
        y_max = max(qm[:,i])
        x_max = max(fitted[i])
        maxim = max(y_max, x_max)
        axs[x, y].plot(fitted[i], qm[:,i], 'o', markersize = 2)
        axs[x, y].plot([minim, maxim], [minim, maxim])
        axs[x, y].axis([minim, maxim, minim, maxim])
        axs[x, y].set_aspect('equal', 'box')
        axs[x, y].set_title(titles[i], fontsize = 14)

    f.text(0.5, 0.04, 'Fitted Dipole Moment (Debye)', ha='center', va='center',
           fontsize = 16)
    f.text(0.06, 0.5, 'QM Dipole Moment (Debye)', ha='center', va='center',
           rotation='vertical', fontsize = 16)
    f.set_size_inches(10.0, 10.0)
    f.savefig(f"{file}_dipolefit.pdf", dpi = 300, bbox_inches='tight')
    print(f"QM vs MD dipole plots are written to: {file}_dipolefit.pdf")
