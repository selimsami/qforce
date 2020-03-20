import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
#
from .elements import ATOMMASS, ATOM_SYM


def calc_qm_vs_md_frequencies(inp, qm, md_hessian):
    qm_freq, qm_vec = calc_vibrational_frequencies(qm.hessian, qm)
    md_freq, md_vec = calc_vibrational_frequencies(md_hessian, qm)
    mean_percent_error = write_vibrational_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, inp)
    plot_frequencies(inp, qm_freq, md_freq, mean_percent_error)


def plot_frequencies(inp, qm_freq, md_freq, mean_percent_error):
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
    f.savefig(f"{inp.job_dir}/frequencies.pdf", bbox_inches='tight')


def calc_vibrational_frequencies(upper, qm):
    """
    Calculate the MD vibrational frequencies by diagonalizing its Hessian
    """
    const_amu = 1.6605389210e-27
    const_avogadro = 6.0221412900e+23
    const_speedoflight = 299792.458
    kj2j = 1e3
    ang2meter = 1e-10
    to_omega2 = kj2j/ang2meter**2/(const_avogadro*const_amu)  # 1/s**2
    to_waveno = 1e-5/(2.0*np.pi*const_speedoflight)  # cm-1

    matrix = np.zeros((3*qm.n_atoms, 3*qm.n_atoms))
    count = 0

    for i in range(3*qm.n_atoms):
        for j in range(i+1):
            mass_i = ATOMMASS[qm.atomids[int(np.floor(i/3))]]
            mass_j = ATOMMASS[qm.atomids[int(np.floor(j/3))]]
            matrix[i, j] = upper[count]/np.sqrt(mass_i*mass_j)
            matrix[j, i] = matrix[i, j]
            count += 1
    val, vec = eigh(matrix)
    vec = np.reshape(np.transpose(vec), (3*qm.n_atoms, qm.n_atoms, 3))[6:]

    for i in range(qm.n_atoms):
        vec[:, i, :] = vec[:, i, :] / np.sqrt(ATOMMASS[qm.atomids[i]])

    freq = np.sqrt(val.clip(min=0)[6:] * to_omega2) * to_waveno
    return freq, vec


def write_vibrational_frequencies(qm_freq, qm_vec, md_freq, md_vec, qm, inp):
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
    freq_file = f"{inp.job_dir}/{inp.job_name}_qforce.freq"
    nmd_file = f"{inp.job_dir}/{inp.job_name}_qforce.nmd"
    errors = []

    with open(freq_file, "w") as f:
        f.write(" mode  QM-Freq   MD-Freq     Diff.  %Error\n")
        for i, (q, m) in enumerate(zip(qm_freq, md_freq)):
            diff = q - m
            err = diff / q * 100
            if q > 100:
                errors.append(err)
            f.write(f"{i+7:>4}{q:>10.1f}{m:>10.1f}{diff:>10.1f}{err:>8.2f}\n")
        f.write("\n\n         QM vectors              MD Vectors\n")
        f.write(50*"=")
        for i, (qm1, md1) in enumerate(zip(qm_vec, md_vec)):
            f.write(f"\nMode {i+7}\n")
            for qm2, md2 in zip(qm1, md1):
                f.write("{:>8.3f}{:>8.3f}{:>8.3f}{:>10.3f}{:>8.3f}{:>8.3f}\n".format(*qm2, *md2))

    mean_percent_error = np.abs(np.array(errors)).mean()

    with open(nmd_file, "w") as nmd:
        nmd.write(f"nmwiz_load {inp.job_name}_qforce.nmd\n")
        nmd.write(f"title {inp.job_name}\n")
        nmd.write("names")
        for ids in qm.atomids:
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
    print(f"QM vs MD vibrational frequencies can be found in: {freq_file}")
    print(f"Vibrational modes (can be run in VMD) is located in: {nmd_file}\n")
    return mean_percent_error
