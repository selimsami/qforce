def bonds_calculated_printed(vibrational_scaling_squared, bond_list,
                             bond_lengths, atom_names, eigenvalues, 
                             eigenvectors, coords, out_file):
    
    #This function uses the Seminario method to find the bond
    #parameters and print them to file

    import numpy as np
    from .force_constants import force_constant_bond

    k_b = np.zeros(len(bond_list))
    bond_length_list = np.zeros(len(bond_list))

    bonds = {}

    for i, bond in enumerate(bond_list):
        AB = force_constant_bond(*bond, eigenvalues, eigenvectors, coords)
        BA = force_constant_bond(*bond, eigenvalues, eigenvectors, coords)
        
        # Order of bonds sometimes causes slight differences, find the mean
        k_b[i] = np.real(( AB + BA ) /2); 

        # Vibrational_scaling takes into account DFT deficities/ anharmocity    
        k_b[i] = k_b[i] * vibrational_scaling_squared
        bond_length_list[i] =  bond_lengths[bond[0]][bond[1]]

        b_type = sorted([atom_names[bond[0]], atom_names[bond[1]]])
        b_type = '-'.join(b_type)
    
        if b_type not in bonds.keys():
            bonds[b_type] = [[i], bond_length_list[i], k_b[i], 1]
        else:
            bonds[b_type][0].append(i)
            bonds[b_type][1] += bond_length_list[i]
            bonds[b_type][2] += k_b[i]
            bonds[b_type][3] += 1
    
    for b_type in bonds.keys():
        bonds[b_type][1] = bonds[b_type][1] * 0.1 / bonds[b_type][3]
        bonds[b_type][2] = bonds[b_type][2] * 4.184 * 200 / bonds[b_type][3]
        for i in bonds[b_type][0]:
            bond_list[i] = [atom + 1 for atom in bond_list[i]]
            bond_list[i].extend([bonds[b_type][1], bonds[b_type][2]])
    
    with open(out_file, 'w') as gmx:    
        gmx.write('[bonds]\n')
        gmx.write(';   ai    aj   f        r0           kb\n')
        for bond in bond_list:
            gmx.write("{:>6}{:>6}   1{:>10.4f}{:>13.1f}\n".format(*bond))