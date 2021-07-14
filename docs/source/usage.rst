Usage
=====

Q-Force is run in multiple stages. These stages are explained below.
At each stage, an options file can be provided to change the default settings
with :code:`-o file_name`. Possible options are listed in :doc:`options`.

.. colt_commandline:: qforce.main run

    main_order = logo, comment, usage, pos_args, opt_args, subparser_args, space, space, space
    alias = qforce

    [arg_format]
    name = 25
    comment = 60

    [subparser_format]
    name = 25
    comment = 60

    [logo]
          ____         ______
         / __ \       |  ____|
        | |  | |______| |__ ___  _ __ ___ ___
        | |  | |______|  __/ _ \| '__/ __/ _ \
        | |__| |      | | | (_) | | | (_|  __/
         \___\_\      |_|  \___/|_|  \___\___|

                     Selim Sami
            University of Groningen - 2020
            ==============================


1) Creating the initial QM input
--------------------------------

et's assume that we have a coordinate file called **mol.ext** for a molecule named **mol**.
The extension (**ext**) can be anything that is supported by 
`ASE <https://gitlab.com/ase/ase>`_ (xyz, pdb, gro, ...).
reate the initial QM input (choosing the QM Software is described in :doc:`options`) by running the following command
:code:`qforce mol.ext`

This creates a directory called *mol_qforce*. In it, you can find **mol_hessian.inp**.
Run this calculation on a cluster or locally, and place the output(s) in the same directory.


2) Treating the flexible dihedrals
-----------------------------------

If your molecule contains flexible dihedrals and if the treatment of flexible dihedrals are
not turned off, then fragments and the corresponding QM inputs are created for all unique flexible
dihedrals inside the subdirectory *fragments* with:

:code:`qforce mol` (or :code:`qforce mol_qforce`, or :code:`qforce mol.ext`)

Run these calculations on a cluster or locally, and place the output in the same subdirectory.


3) Creating the force field
----------------------------

Now that all necessary QM results are available, the fitting of the force field is done with:

:code:`qforce mol` (or :code:`qforce mol_qforce`, or :code:`qforce mol.ext`)


4) Output
----------------------------

Done! Q-Force generates several outputs:

-   Force field files in GROMACS format (.gro, .itp, .top)
-   Force field validation:

    *   QM vs MM vibrational frequencies (frequencies.txt, frequencies.pdf)
    *   QM vs MM dihedral profile(s) in the *fragments* subdirectory (.pdf)

-   MM vibrational modes (frequencies.nmd) that can be visualized in VMD
