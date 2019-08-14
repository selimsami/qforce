# Q-Force: **Q**M-derived **force**field parameters

## Installation
Install with:

    pip install git+https://github.com/selimsami/qforce.git
- Make sure you have Python 3.5 or newer (`python --version`)
- If you can't call the qforce executable afterwards, make sure you have the python bin in your PATH.
- If you work in a shared environment (cluster), install it locally with `pip install --user`

## Features
Currently qforce has several features:
- **hessianfitting** : Fit the MD Hessian to the QM Hessian to obtain bonded stiff force field parameters
- **dipolefitting** : Fit atomistic charges to reproduce QM dipoles from multiple snapshots
- **dihedralfitting** : Fit dihedral functions to reproduce QM torsional scan profiles

**Gaussian** is not directly used by this package but its output is necessary for all of the above options. To this end, **qforce** can create the necessary **Gaussian** input files starting from a .gro/.pdb file for each of the above features:

- **input_bondangle** : Creates an input for a frequency calculation with pre-optimization (for bondangle)
- **input_traj** : For a GRO/PDB file with multiple snapshots (can be created by gmx trjconv) create a QM single point calculation input for each of the snapshots (for dipolefitting)
-  **input_dihedral** : Creates a dihedral scan input for each of the dihedrals entered (for dihedralfitting)


## Scope

### hessianfitting
- Input: Gaussian frequency calculation output and the formatted checkpoint file
- Compute QM-based bond lengths, angles and stiff dihedrals and the corresponding force constants
- Output: Bonds, angles, stiff dihedrals, charges and the corresponding force constants in GROMACS format
- If equivalence option is turned on, parameters are averaged over equivalent atoms (read more on equivalence below in Usage)

### dipolefitting

- Input: Multiple QM single-point calculation outputs
- Fit atomistic charges to QM dipole moments in x, y, z directions
- If equivalence option is turned on, same charges are fitted to equivalent atoms (read more on equivalence below in Usage)
- If desired, charges can be constrained to only change by a percentage compared to ESP or Mulliken charges (this can help to prevent sign flip of very small charges). Turned off by default.
- Output: 1 - Plot of the total and x, y, z dipole moments  - QM vs with fitted charges. 2 - Charges written in GROMACS format.
### dihedralfitting
- Input: QM dihedral scan output(s)
- Run GROMACS energy minimization at each scan angle starting with the QM optimized structures without a dihedral function at the scanned dihedral
- If multiple files for a single dihedral is provided (f.e. forward and reverse scan) and if both scans go over same angles, the one with the lower energy is taken (this helps get rid of a possible hysteresis problem)
- Compute the difference between QM and MD energies to obtain the profile to be fitted. 
- Fit the profile to a choice of three functions (GROMACS dihedral type): Ryckaert-Bellemans (3)(default), periodic (1), improper (2)  
- Rerun GROMACS with the fitted dihedrals
- Output: 1- Plot of QM vs MD dihedral profiles for each dihedral. 2- QM-fitted dihedrals in GROMACS format.

**Requirements:** GROMACS software

### input_bondangle / input_traj / input_dihedral
- Create the QM input files whose output is necessary for the previously mentioned features.
- Can either use the default QM calculation options or alter them
- Can also add pre/post input commands to turn QM inputs directly into job scripts

**Requirements:** OpenBabel software

### input_traj
- This option requires .gro/.pdb file with multiple snapshots. This file can easily be created by `gmx trjconv -f trj.xtc -s topol.tpr -o frames.gro`. If you have a single molecule, make sure that it is whole with `-pbc whole` option. And if you have multiple molecules, make sure that the cluster is together (aka not seeing each other through pbc) with `-pbc cluster`.
- For each snapshot in the .gro/.pdb file, a QM input is created in a directory.


## Usage

Q-Force can be run by providing the name of the job type and the input file:

    qforce <job_type> <input_file>
    
Currently, the available job types are the ones described in the **Features** section.

For the input file, there are two type of keywords, one-line and \[multi-line]:

    [multi_line]
    value1
    value2
    ...
    
    one_line = value

Each feature has different **compulsory** (in bold) and optional keywords. All keywords can be put together in the same input file if desired.

### hessianfitting
- **`fchk_file`: Name of the QM formatted checkpoint file**
- `n_equivalence`: For each atom, checks up to the **n**th neighbor and if multiple atoms have same number of neighbors with same elements, consider them equivalent and average parameters over equivalent properties (optional, default: 4, can be turned off with -1)

### dipolefitting
- **`traj_dir`: Name of directory in which all QM outputs are** (this is not necessary if `coord_file` is given instead)
- `n_equivalence`: For each atom, checks up to the **n**th neighbor and if multiple atoms have same number of neighbors with same elements, consider them equivalent and fit same charges for equivalent atoms (optional, default: 4, can be turned off with -1)
- `fit_percent`:  Constrain the charges to only change by a percentage compared to ESP or Mulliken charges (optional, default: -1 [off])
- `guess_charges`: Initial guess charges used during fitting and also the choice of charges for the optional constrained fitting with `fit_percent` (optional, default: ESP, alternative: mulliken)

### dihedralfitting
- **`[qm_scan_out]`: Name(s) of the QM dihedral scan output file(s). Files belonging to the same dihedral (f.e. a forward and reverse scan) should be given in the same line. Files belonging to different dihedrals should be given in different lines.**
- **`itp_file`: Name of the .itp file containing the molecule with the scanned dihedral**
- **`top_file`: Name of the .top file describing the system**
- mdp_file: Name of the energy minimization .mdp file containing the run options (optional, default: an .mdp file is provided by qforce)
- `[extra_files]`: If your .top file is linked to other local file(s) that is required to run a simulation, you should give their names here. (optional)
- `fitting_function`: Type of dihedral function to be used for fitting. Options are: bellemans (default), periodic, improper (optional)

### input_bondangle / input_traj / input_dihedral
- **`coord_file`: Name of the coordinate file (.pdb or .gro)**
- `charge`: Charge of the molecule (optional, default: 0)
- `multiplicity`: Multiplicity of the molecule (optional, default: 1)
- `method`: QM method to be used (optional, default: wB97XD)
- `dispersion`: Dispersion correction method (optional, default: off)
- `basis_set`: Basis set to be used (optional, default: 6-31G(d,p))
- `n_procs`: Number of cores to be used (optional, default: unset (1))
- `memory`: Size of the memory to be used/asked (optional, default: unset)
- `[pre_input_commands]`: Lines to appear before the QM input. Can be used to change QM inputs into job scripts. Here, `<outfile>` pattern can be used within any line insert the name of the output file into the job script (optional, default: none)
- `[post_input_commands]`: Lines to appear after the QM input (optional, default: none)

**Few options only relevant for input_dihedral:**
- `[dihedrals_scanned]`: Atom numbers of each dihedral in a different line (optional, default: d1 d2 d3 d4 - as a template)
- `scan_no`: Number of scan steps (optional, default: 35)
- `scan_step`: Angle difference between each scan step (optional, default: 10.0)