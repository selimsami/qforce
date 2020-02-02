def write_ff(ff, inp, is_polar):
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

    if is_polar:
        polar = "_polar"
    else:
        polar = ""

    itp_file = f"{inp.job_dir}/{inp.job_name}_qforce{polar}.itp"
    top_file = f"{inp.job_dir}/gas{polar}.top"
    gro_file = f"{inp.job_dir}/gas{polar}.gro"

    write_itp(ff, inp, itp_file)
    write_top(inp, top_file, polar)
    write_gro(ff, inp, gro_file)


def write_itp(ff, inp, itp_file):
    form_atypes = "{:>5} {:>8.4f} {:>8.4f} {:>2} {:>12.5e} {:>12.5e}\n"
    form_atoms = "{:>5}{:>5}{:>6}{:>6}{:>7}{:>5}{:>11.5f}{:>10.5f}\n"
    form_bonds = "{:>6}{:>6}{:>6}{:>10.5f}{:>10.0f}\n"
    form_angles = "{:>6}{:>6}{:>6}{:>6}{:>11.3f}{:>13.3f}\n"
    form_urey = "{:>6}{:>6}{:>6}{:>6}{:>11.3f}{:>13.3f}{:>10.5f}{:>13.3f}\n"
    form_diheds = "{:>6}{:>6}{:>6}{:>6}{:>6}{:>11.3f}{:>13.3f}\n"
    form_thole = "{:>6}{:>6}{:>6}{:>6}{:>4}{:>7.2f}{:>14.8f}{:>14.8f}\n"
    form_flex = ("{:>6}{:>6}{:>6}{:>6}{:>6}{:>11.3f}{:>11.3f}{:>11.3f}"
                 "{:>11.3f}{:>11.3f}{:>11.3f}\n")
    form_unfit = ";{:>6}{:>6}{:>6}{:>6}{:>6} ; Term ID: {:>6}\n"

    with open(itp_file, "w") as itp:
        # fitting parameters - temporary
        if inp.param != []:
            itp.write('; fitting parameters are (C, H, O, N):\n')
            itp.write(f'; S8: ')
            for s8 in inp.param[::2]:
                itp.write(f'{s8} ')
            itp.write(f'\n; R_ref: ')
            for r in inp.param[1::2]:
                itp.write(f'{r} ')
            itp.write('\n\n\n')

        # atom types
        if ff.atom_types != []:
            itp.write("[ atomtypes ]\n")
            itp.write(";name     mass   charge  t        sigma      epsilon\n")
        for at in ff.atom_types:
            itp.write(form_atypes.format(*at))

        # molecule type
        space = " "*(len(ff.mol_type)-5)
        itp.write("\n[ moleculetype ]\n")
        itp.write(f";{space}name nrexcl\n")
        itp.write(f"{ff.mol_type}{inp.n_excl:>7}\n")

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
            itp.write(form_bonds.format(*bond))

        # angles
        itp.write("\n[ angles ]\n")
        itp.write(";   ai    aj    ak     f        th0          kth\n")
        for angle in ff.angles:
            if inp.urey:
                itp.write(form_urey.format(*angle))
            else:
                itp.write(form_angles.format(*angle))

        # dihedrals
        itp.write("\n[ dihedrals ]\n")
        itp.write(";   ai    aj    ak    al     f        th0          kth\n")
        itp.write("; proper dihedrals \n")
        for dihedral in ff.dihedrals:
            itp.write(form_diheds.format(*dihedral))

        # improper dihedrals
        itp.write("\n[ dihedrals ]\n")
        itp.write(";   ai    aj    ak    al   f      th0         kth\n")
        itp.write("; improper dihedrals \n")
        for improper in ff.impropers:
            itp.write(form_diheds.format(*improper))

        # flexible dihedrals
        itp.write("\n;[ dihedrals ]\n")
        itp.write(";   ai    aj    ak    al   f     th0         kth\n")
        itp.write("; flexible dihedrals\n")
        if inp.nofrag:
            for flexible in ff.flexible:
                itp.write(form_unfit.format(*flexible))
        else:
            for flexible in ff.flexible:
                itp.write(form_flex.format(*flexible))

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


def write_top(inp, top_file, polar):
    with open(top_file, "w") as top:
        # defaults
        top.write("\n[ defaults ]\n")
        top.write(";nbfunc   comb-rule   gen-pairs   fudgeLJ   fudgeQQ\n")
        top.write("      1           2          no       1.0       1.0\n\n\n")

        top.write("; Include the molecule ITP\n")
        top.write(f'#include "./{inp.job_name}_qforce{polar}.itp"\n\n\n')

        size = len(inp.job_name)
        top.write("[ system ]\n")
        top.write(f"; {' '*(size-6)}name\n")
        top.write(f"{' '*(6-size)}{inp.job_name}\n\n\n")

        top.write("[ molecules ]\n")
        top.write(f"; {' '*(size-10)}compound    n_mol\n")
        top.write(f"{' '*(10-size)}{inp.job_name}        1\n\n")


def write_gro(ff, inp, gro_file):
    with open(gro_file, "w") as gro:
        gro.write(f"{inp.job_name}\n")
        gro.write(f"{ff.natom*ff.n_mol:>6}\n")
        for m in range(ff.n_mol):
            for i, a in enumerate(ff.atoms):
                gro.write(f"{a[2]:>5}{a[3]:<5}")
                gro.write(f"{a[4]:>5}{m*ff.natom+a[0]:>5}")
                gro.write("{:>8.3f}{:>8.3f}{:>8.3f}\n"
                          .format(*ff.coords[m*ff.natom+i]))
        gro.write("{:>12.5f}{:>12.5f}{:>12.5f}\n".format(*ff.box))
