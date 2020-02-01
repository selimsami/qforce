import subprocess
import numpy as np
from ase.io import write
from ase import Atoms
from scipy.optimize import curve_fit


def run_dftd4(inp, qm):
    n_more = 0
    c6, c8, alpha, r_rel = [], [], [], []
    write_optimized_xyz(inp, qm)
    dftd4 = subprocess.Popen(['dftd4', '-c', str(inp.charge), inp.xyz_file],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dftd4.wait()
    check_termination(dftd4)
    out = dftd4.communicate()[0].decode("utf-8")

    with open(f'{inp.job_dir}/dftd4_results', 'w') as dftd4_file:
        dftd4_file.write(out)

    for line in out.split('\n'):
        if 'number of atoms' in line:
            n_atoms = int(line.split()[4])
        elif 'covCN                  q              C6A' in line:
            n_more = n_atoms
        elif n_more > 0:
            line = line.split()
            qm.q.append(float(line[4]))
            c6.append(float(line[5]))
            c8.append(float(line[6]))
            alpha.append(float(line[7]))
            r_rel.append(float(line[8])**(1/3))
            n_more -= 1

    calc_c6_c12(qm, c6, c8, r_rel, inp.param)


def write_optimized_xyz(inp, qm):
    inp.xyz_file = f'{inp.job_dir}/opt.xyz'
    mol = Atoms(numbers=qm.atomids, positions=qm.coords)
    write(inp.xyz_file, mol, plain=True, comment=f'{inp.job_name} - optimized '
          'geometry')


def check_termination(process):
    if process.returncode != 0:
        print(process.communicate()[0].decode("utf-8"))
        raise RuntimeError({"DFTD4 run has terminated unsuccessfully"})


def calc_c6_c12(qm, c6s, c8s, r_rels, param):
    hartree2kjmol = 2625.499638
    bohr2ang = 0.52917721067
    new_ljs = []

    order2elem = [6, 1, 8, 7]
    r_ref = {1: 1.986, 6: 2.083, 7: 1.641, 8: 1.452, 16: 1.5}
    s8_scale = {1: 0.133, 6: 0.133, 7: 0.683, 8: 0.683, 16: 0.5}

    for i, s8 in enumerate(param[::2]):
        s8_scale[order2elem[i]] = s8
    for i, r in enumerate(param[1::2]):
        r_ref[order2elem[i]] = r

    for i, (c6, c8, r_rel) in enumerate(zip(c6s, c8s, r_rels)):
        elem = qm.atomids[i]
        c8 *= s8_scale[elem]
        c10 = 40/49*(c8**2)/c6

        r_vdw = 2*r_ref[elem]*r_rel/bohr2ang
        r = np.arange(r_vdw*0.5, 20, 0.01)
        c12 = (c6 + c8/r_vdw**2 + c10/r_vdw**4) * r_vdw**6 / 2
        lj = c12/r**12 - c6/r**6 - c8/r**8 - c10/r**10
        weight = 10*(1-lj / min(lj))+1
        popt, _ = curve_fit(calc_lj, r, lj, absolute_sigma=False, sigma=weight)
        new_ljs.append(popt)

    new_ljs = np.array(new_ljs)*hartree2kjmol
    qm.c6 = new_ljs[:, 0]*bohr2ang**6
    qm.c12 = new_ljs[:, 1]*bohr2ang**12

#    for i, atom in enumerate(mol.atoms):
#        mol.node(i)['c6'] = qm.c6[atom]
#        mol.node(i)['c12'] = qm.c12[atom]


def calc_lj(r, c6, c12):
    return c12/r**12 - c6/r**6


# def move_polarizability_from_hydrogens(alpha, mol):
#     new_alpha = np.zeros(mol.n_atoms)
#     for i, a_id in enumerate(mol.atomids):
#         if a_id == 1:
#             new_alpha[mol.neighbors[0][i][0]] += alpha[mol.atoms[i]]
#         else:
#             new_alpha[i] += alpha[mol.atoms[i]]
#     return new_alpha
