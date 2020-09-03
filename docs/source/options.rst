Options
======================

.. code-block:: text

    # ...
    a = b :: c :: d

    # = Description
    a = command
    b = default value
    c = type
    d = available options


# Directory where the fragments are saved

frag_lib = ~/qforce_fragments :: folder

----------------

# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent

n_equiv = 4 :: int

----------------

# Number of first n neighbors to exclude in the forcefield

n_excl = 2 :: int :: [2, 3]

----------------

# Point charges used in the forcefield

point_charges = cm5 :: str :: [cm5, esp, ext]

----------------

# Lennard jones method for the forcefield

lennard_jones = gromos_auto :: str :: [gromos_auto, gromos, opls, gaff]

----------------

# Use Urey-Bradley angle terms

urey = yes :: bool

----------------

# Ignore non-bonded interactions during fitting

non_bonded = yes :: bool

----------------

# Make fragments and calculate flexible dihedrals
# Use 'available' option to skip dihedrals with missing scan data

fragment = yes :: str :: [yes, no, available]

----------------

# Number of neighbors after bonds can be fragmented (0 or smaller means no fragmentation)

frag_n_neighbor = 3 :: int

----------------

# Set all dihedrals as rigid

all_rigid = no :: bool

----------------

# Method for doing the MM relaxed dihedral scan

scan_method = qforce :: str :: [qforce, gromacs]

----------------

# Number of iterations of dihedral fitting

n_dihed_scans = 3 :: int

----------------

# To turn the QM input files into job scripts

job_script = :: literal

----------------

# Additional exclusions (GROMACS format)

exclusions = :: literal

----------------

# Number of dihedral scan steps to perform

scan_no = 23 :: int

----------------

# Step size for the dihedral scan

scan_step = 15.0 :: float

----------------

# Total charge of the system

charge = 0 :: int

----------------

# Multiplicity of the system

multi = 1 :: int

----------------

# QM method to be used

method = PBEPBE :: str

----------------

# QM basis set to be used

basis = 6-31+G(D) :: str

----------------

# Dispersion correction to include

disp = GD3BJ :: str

----------------

# Number of processors for the QM calculation

nproc = 1 :: int

----------------

# Memory setting for the QM calculation

mem = 4GB :: str
