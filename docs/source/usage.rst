Usage
======

Q-Force is run in multiple stages. These stages are explained below.
At each stage, an options file can be provided to change the default settings
with :code:`-o file_name`. These options are discussed in the next page.


1) Creating the initial QM input
---------------------------------

Let's assume that we have a coordinate file called mol.ext for a molecule named mol.
The extension (ext) can be anything that is supported by ASE (xyz, pdb, gro, ...).
Create the initial QM input by running the following command:

:code:`qforce mol.ext`

This creates a folder called mol_qforce. In it, you can find mol_hessian.inp
(inp extension varies based on QM software and settings).
Run this calculation on a cluster or locally, and place the output in the same directory.


2) Treating the flexible dihedrals
-----------------------------------

If your molecule contains flexible dihedrals and if the treatment of flexible dihedrals are
not turned off, then fragments and the corresponding QM inputs are created for all unique flexible
dihedrals inside the subdirectory 'fragments' with one of the commands:

:code:`qforce mol.ext` or :code:`qforce mol` or :code:`qforce mol_qforce`


Run these calculations on a cluster or locally, and place the output in the same subdirectory.


3) Creating the force field
----------------------------

Now that all necessary QM results are available, the force field can be created with one of the
following commands:

:code:`qforce mol.ext` or :code:`qforce mol` or :code:`qforce mol_qforce`
