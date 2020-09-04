Examples
======================

Here are two examples of how Q-Force can be used: In default settings and with some customization.
For the purposes of these examples, whenever you need an additional file, QM outputs or otherwise,
they are provided in the directory *necessary_files*.

First, please get the example files by:

:code:`git clone https://github.com/selimsami/qforce_examples.git`

|

Default settings
-------------------

Creating the initial QM input
++++++++++++++++++++++++++++++++

Find in *examples/default_settings* a coordinate file (propane.xyz) for the propane molecule.

Let's first create the QM input file:

:code:`qforce propane.xyz`

This will create a *propane_qforce* directory, and in it, you will find 'propane_hessian.com'.
Now run this QM calculation and put the output file (.out) and the formatted checkpoint file
(.fchk) in the same directory.

Treating the flexible dihedrals
++++++++++++++++++++++++++++++++

Now we can run Q-Force again from the same folder to create fragments and the corresponding 
QM dihedral scan input files by:

:code:`qforce propane`

This will create all the necessary input files in the directory *propane_qforce/fragments*.
Then, run these calculations and put the output file(s) (.out) in the same directory.

Creating the force field
++++++++++++++++++++++++++++++++

Now that all necessary QM data is available, let's create our force field:

:code:`qforce propane`

You can now find the necessary force field files in the *propane_qforce* directory.

|

Custom settings
------------------
Find in *examples/custom_settings* a coordinate file (benzene.pdb) for the benzene molecule.
In this example, we look at some of the custom settings available with Q-Force and how they
can be executed.
The custom settings are provided with an external file with:

:code:`qforce benzene.pdb -o settings_file_name`.


Custom Lennard-Jones and Charges
++++++++++++++++++++++++++++++++

By default, Q-Force determines the atom types for Lennard-Jones interactions automatically.
Alternatively, the user can also provide atom types manually, for a force field of their choice.
Here, we choose to use the GAFF force field by:

.. code-block:: text

    lennard_jones = gaff

With this command, the user also has to provide the atom types manually in the 'benzene_qforce'
directory in a file called "ext_lj". In this file, every line should contain the atom type of one
atom in the same order as the coordinate file.

Similarly, for the point charges, by default CM5 charges are used. In this example, we want to use
ESP charges instead. This is done by:

.. code-block:: text

    point_charges = esp

Conversion to job script
++++++++++++++++++++++++

Often the QM calculations are needed to be submitted as jobs in supercomputers.
For large molecules Q-Force can have a large number of QM dihedral scans that needs to be
performed and therefore it may be convenient to have input files converted to job scripts.
This can be done with the **[job_script]** block setting as shown in the following example:

.. code-block:: text

    [job_script]
    #!/bin/bash
    #SBATCH --time=1-00:00:00
    #SBATCH -o <outfile>.out

    g16<<EOF
    <input>
    EOF

Here we make a SLURM job script. Two placeholders can be used: **<outfile>** and **<input>**.
**<outfile>** gets replaced by the name of the calculation, for example in the case of the
'benzene_hessian.inp', it will be 'benzene_hessian.out'.
**<input>** is where the content of the QM input file will be placed.



Creating the initial QM input
++++++++++++++++++++++++++++++++

Now that we know what these settings do, let's supply them to Q-Force:

:code:`qforce benzene.pdb -o settings`


Again, this will create a *benzene_qforce* directory, and in it, you will find
'benzene_hessian.inp', this time as a job script instead of an input file. Now run this QM
calculation and put the output file (.out) and the formatted checkpoint file (.fchk) in
the same directory.



Creating the force field
++++++++++++++++++++++++++++++++

As benzene does not have any flexible dihedrals, the second step is skipped in this case.
Make sure you have also added this time the **ext_lj** file in *benzene_qforce* and then we can
already create the force field with:

:code:`qforce benzene -o settings`

You can now find the necessary force field files in the *benzene_qforce* directory.
As you will notice, in this case GAFF atom types with ESP charges are used.
