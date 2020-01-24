def write_itp(ff, itp_file, inp):
    """
    Scope:
    ------
    Write forcefield files: ITP and/or GRO

    Notes:
    -----
    "if str" corresponds to reading an existing FF - related to polarize
    "else" corresponds to hessian fitting
    This is terrible. Should be improved at some point.
    Perhaps it can wait until polarize is combined with FF creation.
    """
    form_atypes = "{:>5} {:>8.4f} {:>8.4f} {:>2} {:>12.5e} {:>12.5e}\n"
    form_atoms = "{:>5}{:>5}{:>6}{:>6}{:>7}{:>5}{:>11.5f}{:>10.5f}\n"
    form_bonds = "{:>6}{:>6}{:>6}{:>10.5f}{:>10.0f}\n"
    form_angles = "{:>6}{:>6}{:>6}{:>6}{:>10.2f}{:>13.3f}\n"
    form_urey = "{:>6}{:>6}{:>6}{:>6}{:>10.2f}{:>13.3f}{:>10.5f}{:>13.3f}\n"
    form_diheds = "{:>6}{:>6}{:>6}{:>6}{:>6}{:>10.2f}{:>13.3f}\n"
    form_thole = "{:>6}{:>6}{:>6}{:>6}{:>4}{:>7.2f}{:>14.8f}{:>14.8f}\n"
    form_unfit = ";{:>6}{:>6}{:>6}{:>6}{:>6}   -   Type: {:>6}\n"

    with open(itp_file, "w") as itp:

        # atom types
        if ff.atom_types != []:
            itp.write("[ atomtypes ]\n")
            itp.write(";name     mass   charge  t           c6          c12\n")
        for at in ff.atom_types:
            itp.write(form_atypes.format(*at))

        # molecule type
        space = " "*(len(ff.mol_type)-5)
        itp.write("\n[ moleculetype ]\n")
        itp.write(f";{space}name nrexcl\n")
        itp.write(f"{ff.mol_type}{3:>7}\n")

        # atoms
        itp.write("\n[ atoms ]\n")
        itp.write(";  nr type resnr resnm   atom cgrp     charge      mass\n")
        for atom in ff.atoms:
            itp.write(form_atoms.format(*atom))

        # polarization
        if ff.polar != []:
            itp.write("\n[ polarization ]\n")
            itp.write(";    i     j     f         alpha\n")
        for pol in ff.polar:
            itp.write("{:>6}{:>6}{:>6}{:>14.8f}\n".format(*pol))

        # thole polarization
        if ff.thole != []:
            itp.write("\n[ thole_polarization ]\n")
            itp.write(";   ai    di    aj    dj   f      a      alpha(i)      "
                      "alpha(j)\n")
        for tho in ff.thole:
            itp.write(form_thole.format(*tho))

        # pairs
        if ff.pairs != []:
            itp.write("\n[ pairs ]\n")
            itp.write(";    i     j  func\n")
        for pair in ff.pairs:
            itp.write(f"{pair[0]:>6}{pair[1]:>6}{1:>6}\n")

        # bonds
        itp.write("\n[ bonds ]\n")
        itp.write(";   ai    aj     f        r0        kb\n")
        for bond in ff.bonds:
            if type(bond) is str:
                itp.write(bond)
            else:
                itp.write(form_bonds.format(*bond))

        # angles
        itp.write("\n[ angles ]\n")
        itp.write(";   ai    aj    ak     f       th0          kth\n")
        for angle in ff.angles:
            if type(angle) is str:
                itp.write(angle)
            elif inp.urey:
                itp.write(form_urey.format(*angle))
            else:
                itp.write(form_angles.format(*angle))

        # dihedrals
        itp.write("\n[ dihedrals ]\n")
        itp.write(";   ai    aj    ak    al     f       th0          kth\n")
        itp.write("; proper dihedrals \n")
        for dihedral in ff.dihedrals:
            if type(dihedral) is str:
                itp.write(dihedral)
            else:
                itp.write(form_diheds.format(*dihedral))

        # improper dihedrals
        itp.write("\n[ dihedrals ]\n")
        itp.write(";   ai    aj    ak    al   f     th0         kth\n")
        itp.write("; improper dihedrals \n")
        for improper in ff.impropers:
            itp.write(form_diheds.format(*improper))

        # flexible dihedrals
        itp.write("\n;[ dihedrals ]\n")
        itp.write(";   ai    aj    ak    al   f     th0         kth\n")
        itp.write("; flexible dihedrals - fit manually for now\n")
        for flexible in ff.flexible:
            itp.write(form_unfit.format(*flexible))

        # flexible dihedrals
        itp.write("\n;[ dihedrals ]\n")
        itp.write(";   ai    aj    ak    al   f     th0         kth\n")
        itp.write("; constrained dihedrals - fit manually for now\n")
        for constrained in ff.constrained:
            itp.write(form_unfit.format(*constrained))

        # exclusions
        if ff.exclu != []:
            itp.write("\n[ exclusions ]\n")
        for i, ex in enumerate(ff.exclu):
            if len(ex) > 0:
                itp.write("{} ".format(i+1))
                itp.write(("{} "*len(ex)).format(*ex))
                itp.write("\n")


def write_gro(ff, gro_file):
    with open(gro_file, "w") as gro:
        gro.write("{}".format(ff.title))
        gro.write("{}\n".format(ff.gro_natom*2))
        for m in range(ff.n_mol):
            for i, a in enumerate(ff.atoms):
                gro.write("{:>9}".format((str(a[2])+a[3])))
                gro.write("{:>6}{:>5}".format(a[4], m*ff.natom*2+a[0]))
                gro.write("{:>8.3f}{:>8.3f}{:>8.3f}\n"
                          .format(*ff.coords[m*ff.natom*2+i]))
        gro.write("{}".format(ff.box))
