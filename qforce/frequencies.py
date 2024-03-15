from .elements import ATOMMASS, ATOM_SYM
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
#


def calc_qm_vs_md_frequencies(job, qm, md_hessian):
    qm_freq, qm_vec = calc_vibrational_frequencies(qm.hessian, qm)
    md_freq, md_vec = calc_vibrational_frequencies(md_hessian, qm)
    mean_percent_error = write_vibrational_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, job)
    plot_frequencies(job, qm_freq, md_freq, mean_percent_error)


def plot_frequencies(job, qm_freq, md_freq, mean_percent_error):
    n_freqs = np.arange(len(qm_freq))+1
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width, height), dpi=300)
    sns.set(font_scale=1.3)
    plt.title(f'Mean Percent Error = {round(mean_percent_error, 2)}%', loc='left')
    plt.xlabel('Vibrational Mode #')
    plt.ylabel(r'Frequencies (cm$^{-1}$)')
    plt.plot(n_freqs, qm_freq, linewidth=3, label='QM')
    plt.plot(n_freqs, md_freq, linewidth=3, label='Q-Force')
    plt.tight_layout()
    plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
    f.savefig(f"{job.dir}/frequencies.pdf", bbox_inches='tight')
    plt.close()


def calc_vibrational_frequencies(upper, qm):
    """
    Calculate the MD vibrational frequencies by diagonalizing its Hessian
    """
    molar_mass_const = 0.00099999999965  # CODATA 2018, kg/mol
    const_speedoflight = 299792458  # CODATA 2018, m/s
    kj2j = 1e3
    ang2meter = 1e-10
    m2cm = 1e-2

    to_omega2 = kj2j/ang2meter**2/molar_mass_const  # 1/s**2
    to_waveno = m2cm/(2.0*np.pi*const_speedoflight)  # cm-1

    matrix = np.zeros((3*qm.n_atoms, 3*qm.n_atoms))

    # convert to matrix
    count = 0
    for i in range(3*qm.n_atoms):
        for j in range(i+1):
            mass_i = ATOMMASS[qm.elements[int(np.floor(i/3))]]
            mass_j = ATOMMASS[qm.elements[int(np.floor(j/3))]]
            matrix[i, j] = upper[count]/np.sqrt(mass_i*mass_j)
            matrix[j, i] = matrix[i, j]
            count += 1
    val, vec = eigh(matrix)

    freq = np.sqrt(val.clip(min=0) * to_omega2) * to_waveno

    vec = np.reshape(np.transpose(vec), (3*qm.n_atoms, qm.n_atoms, 3))

    # mass weight
    masses = [ATOMMASS[qm.elements[i]] for i in range(qm.n_atoms)]
    vec /= np.sqrt(masses)[np.newaxis, :, np.newaxis]

    # normalize
    #norm = (vec**2).sum(axis=(1, 2))**0.5
    #vec /= norm[:, np.newaxis, np.newaxis]

    return freq, vec


def write_vibrational_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, job):
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
    freq_file = f"{job.dir}/frequencies.txt"
    nmd_file = f"{job.dir}/frequencies.nmd"
    errors = []

    if qm_freq[5] > 300:
        transrot = 5
    else:
        transrot = 6
    mask = np.arange(qm_freq.size) >= transrot

    qm_freq = qm_freq[mask]
    md_freq = md_freq[mask]
    qm_vec = qm_vec[mask]
    md_vec = md_vec[mask]

    with open(freq_file, "w") as f:
        f.write(" mode    QM-Freq     MD-Freq       Diff.  %Error\n")
        for i, (q, m) in enumerate(zip(qm_freq, md_freq)):
            diff = q - m
            err = diff / q * 100
            if q > 100:
                errors.append(err)
            f.write(f"{i+transrot+1:>4}{q:>12.3f}{m:>12.3f}{diff:>12.3f}{err:>8.2f}\n")
        f.write("\n\n         QM vectors              MD Vectors\n")
        f.write(62*"=")
        for i, (qm1, md1) in enumerate(zip(qm_vec, md_vec)):
            f.write(f"\nMode {i+transrot+1}\n")
            for qm2, md2 in zip(qm1, md1):
                f.write("{:>10.5f}{:>10.5f}{:>10.5f}  {:>10.5f}{:>10.5f}{:>10.5f}\n".format(*qm2, *md2))

    mean_percent_error = np.abs(np.array(errors)).mean()

    with open(nmd_file, "w") as nmd:
        nmd.write(f"nmwiz_load {job.name}_qforce.nmd\n")
        nmd.write(f"title {job.name}\n")
        nmd.write("names")
        for ids in qm.elements:
            nmd.write(f" {ATOM_SYM[ids]}")
        nmd.write("\nresnames")
        for i in range(qm.n_atoms):
            nmd.write(" RES")
        nmd.write("\nresnums")
        for i in range(qm.n_atoms):
            nmd.write(" 1")
        nmd.write("\ncoordinates")
        for c in qm.coords:
            nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
        for i, m in enumerate(md_vec):
            nmd.write(f"\nmode {i+7}")
            for c in m:
                nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
    return mean_percent_error
