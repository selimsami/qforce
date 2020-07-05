
def polarize(inp):
    """
    Generate the polarizable versions of the input forcefield
    for both GRO and ITP files
    """
    polar_coords = []

    atoms, mol_natoms, max_resnr = read_itp(inp.itp_file)
    coords, gro_natoms, box_dim = read_gro(inp.coord_file)

    # add coords
    n_mols = int(gro_natoms/mol_natoms)

    for i in range(n_mols):
        polar_coords.extend(coords[i*mol_natoms:(i+1)*mol_natoms]*2)

    for i in range(mol_natoms):
        # add atoms
        atoms.append({'nr': i+mol_natoms+1, 'resnr': max_resnr+atoms[i]['resnr'],
                      'resname': atoms[i]['resname'], 'atom_name': 'D'})

    polar_gro_file = f"{inp.job_name}_polar.gro"
    write_gro(inp, atoms, mol_natoms, gro_natoms, n_mols, polar_coords, box_dim, polar_gro_file)

    print("Done!")
    print(f"Polarizable coordinate file in: {polar_gro_file}\n\n")


def read_gro(gro_file):
    coords = []
    with open(gro_file, "r") as gro:
        gro.readline()
        gro_natoms = int(gro.readline())
        for i in range(gro_natoms):
            line = gro.readline()
            x = float(line[21:29].strip())
            y = float(line[29:37].strip())
            z = float(line[37:45].strip())
            coords.append([x, y, z])
        box_dim = gro.readline()
    return coords, gro_natoms, box_dim


def read_itp(itp_file):
    atoms = []
    max_resnr = 0

    with open(itp_file, "r") as itp:
        in_section = []
        for line in itp:
            low_line = line.lower().strip().replace(" ", "")
            line = line.split()
            if low_line == "" or low_line[0] == ";":
                continue
            elif "[" in low_line and "]" in low_line:
                open_bra = low_line.index("[") + 1
                close_bra = low_line.index("]")
                in_section = low_line[open_bra:close_bra]

            elif in_section == "atoms":
                if line[4] == 'D' or line[2] == 'DP':
                    continue
                atoms.append({'nr': int(line[0]), 'resnr': int(line[2]), 'resname': line[3],
                              'atom_name': line[4]})
                if atoms[-1]['resnr'] > max_resnr:
                    max_resnr = atoms[-1]['resnr']
    return atoms, len(atoms), max_resnr


def write_gro(inp, atoms, mol_natoms, gro_natoms, n_mols, coords, box_dim, gro_file):
    with open(gro_file, "w") as gro:
        gro.write(f"{inp.job_name} - polarized\n")
        gro.write("{}\n".format(gro_natoms*2))
        for m in range(n_mols):
            for i, atom in enumerate(atoms):
                gro.write(f"{(str(atom['resnr'])+atom['resname']):>9}")
                gro.write(f"{atom['atom_name']:>6}{m*mol_natoms*2+atom['nr']:>5}")
                coord = coords[m*mol_natoms*2+i]
                gro.write(f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}\n")
        gro.write(f"{box_dim}")
