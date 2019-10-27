import subprocess
import numpy as np
import pulp
from scipy.optimize import curve_fit


def run_dftd4(inp, mol):
    n_more = 0
    q, c6, alpha = [], [], []
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
        elif ' covCN         q      C6AA' in line:
            n_more = n_atoms
        elif n_more > 0:
            line = line.split()
            q.append(float(line[4]))
            c6.append(float(line[5]))
            alpha.append(float(line[6]))
            n_more -= 1

    q = average_equivalent_terms(q, mol.list)
    q = sum_charges_to_qtotal(mol, q, inp.charge)
    alpha = average_equivalent_terms(alpha, mol.list)
    c6 = average_equivalent_terms(c6, mol.list)
#    calc_c6_c12(c6[0])

    return q, alpha, c6


def check_termination(process):
    if process.returncode != 0:
        print(process.communicate()[0].decode("utf-8"))
        raise RuntimeError({"DFTD4 run has terminated unsuccessfully"})


def average_equivalent_terms(term, eq):
    avg_term = []
    term = np.array(term)
    for l in eq:
        total = 0
        for a in l:
            total += term[a]
        avg_term.append(round(total/len(l), 5))
    for i, e in enumerate(eq):
        term[e] = avg_term[i]
    return term


def sum_charges_to_qtotal(mol, charges, q_total):
    extra = int(100000 * round(charges.round(5).sum() - q_total, 5))
    if extra != 0:
        if extra > 0:
            sign = 1
        else:
            sign = -1
            extra = - extra

        n_eq = [len(l) for l in mol.list]
        no = [f"{i:05d}" for i, _ in enumerate(n_eq)]

        var = pulp.LpVariable.dicts("x", no, lowBound=0, cat='Integer')
        prob = pulp.LpProblem('prob', pulp.LpMinimize)
        prob += pulp.lpSum([var[n] for n in no])
        prob += pulp.lpSum([eq * var[no[i]] for i, eq
                            in enumerate(n_eq)]) == extra
        prob.solve()

        if prob.status == 1:
            for i, v in enumerate(prob.variables()):
                charges[mol.list[i]] -= sign * v.varValue / 100000
        else:
            print('Failed to equate total of charges to the total charge of '
                  'the system. Do so manually')
    return charges


def calc_c6_c12(c6):
    rvdw = 2.0  # to be parameterized from MD simulations
    c8 = 0  # to be read from DFTD4
    c10 = 40/49*(c8**2)/c6
    r = np.arange(rvdw, 20, 0.01)

    c12 = (c6/rvdw**6 + c8/rvdw**8 + c10/rvdw**10) * rvdw**12
    lj = c12/r**12 - (c6/r**6 + c8/r**8 + c10/r**10)

    weight = 1/np.exp(-1000000*lj)
    popt, pcov = curve_fit(calc_lj, r, lj, absolute_sigma=False, sigma=weight)
    print(popt)


def calc_lj(r, c6, c12):
    return c12/r**12 - c6/r**6
