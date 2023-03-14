import numpy as np
import sys


def polarize(inp, path):
    """
    Generate the polarizable versions of the input forcefield
    for both GRO and ITP files
    """
    polar_coords = []
    polar_vel = []

    atoms, mol_natoms, max_resnr, polar_atoms = read_itp(inp.itp_file)
    coords, velocities, gro_natoms, box_dim = read_gro(inp.coord_file)

    # add coords
    n_mols = int(gro_natoms/mol_natoms)
    n_polar_atoms = len(polar_atoms)

    for i in range(n_mols):
        polar_coords.extend(coords[i*mol_natoms:(i+1)*mol_natoms])
        polar_coords.extend(coords[i*mol_natoms:(i+1)*mol_natoms][sorted(polar_atoms.values())])

        if velocities != []:
            polar_vel.extend(velocities[i*mol_natoms:(i+1)*mol_natoms])
            polar_vel.extend([[0., 0., 0.]]*n_polar_atoms)

    for i in range(n_polar_atoms):
        # add atoms
        ia = i+mol_natoms
        atoms.append({'nr': ia+1, 'resnr': max_resnr+atoms[polar_atoms[ia]]['resnr'],
                      'resname': 'DRU', 'atom_name': f'D{i+1}'})

    new_mol_natoms = mol_natoms + n_polar_atoms

    polar_gro_file = f"{inp.job_name}_polar.gro"
    write_gro(inp, atoms, new_mol_natoms, n_mols, polar_coords, polar_vel, box_dim,
              polar_gro_file)

    print("Done!")
    print(f"Polarizable coordinate file in: {polar_gro_file}\n\n")
    raise SystemExit


def read_gro(gro_file):
    coords, velocities = [], []
    with open(gro_file, "r") as gro:
        gro.readline()
        gro_natoms = int(gro.readline())
        for i in range(gro_natoms):
            line = gro.readline()
            x = float(line[21:29].strip())
            y = float(line[29:37].strip())
            z = float(line[37:45].strip())
            coords.append([x, y, z])
            if len(line) > 68:
                vx = float(line[45:53].strip())
                vy = float(line[53:61].strip())
                vz = float(line[61:69].strip())
                velocities.append([vx, vy, vz])
        box_dim = gro.readline()
    return np.array(coords), np.array(velocities), gro_natoms, box_dim


def read_itp(itp_file):
    atoms = []
    polar_atoms = {}
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
                if line[4].startswith('D') or line[1] == 'DP' or line[3] == 'DRU':
                    continue
                atoms.append({'nr': int(line[0]), 'resnr': int(line[2]), 'resname': line[3],
                              'atom_name': line[4]})
                if atoms[-1]['resnr'] > max_resnr:
                    max_resnr = atoms[-1]['resnr']

            elif in_section == "polarization":
                p_atoms = [int(l)-1 for l in line[0:2]]
                polar_atoms[max(p_atoms)] = min(p_atoms)

    return atoms, len(atoms), max_resnr, polar_atoms


def write_gro(inp, atoms, mol_natoms, n_mols, coords, velocities, box_dim, gro_file):
    with open(gro_file, "w") as gro:
        gro.write(f"{inp.job_name} - polarized\n")
        gro.write(f"{int(mol_natoms*n_mols)}\n")
        for m in range(n_mols):
            for i, atom in enumerate(atoms):
                gro.write(f"{(str(atom['resnr'])+atom['resname']):>8}")
                gro.write(f"{atom['atom_name']:>7}{m*mol_natoms+atom['nr']:>5}")
                coord = coords[m*mol_natoms+i]
                gro.write(f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}")
                if velocities != []:
                    vel = velocities[m*mol_natoms+i]
                    gro.write(f"{vel[0]:>8.3f}{vel[1]:>8.3f}{vel[2]:>8.3f}")
                gro.write('\n')
        gro.write(f"{box_dim}")
