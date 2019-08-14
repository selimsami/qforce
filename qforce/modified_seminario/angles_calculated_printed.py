def angles_calculated_printed(vibrational_scaling_squared, angle_list, 
                              bond_lengths, atom_names, eigenvalues, 
                              eigenvectors, coords, out_file):
    #This function uses the modified Seminario method to find the angle
    #parameters and print them to file

    from operator import itemgetter
    import numpy as np
    from .force_constants import u_PA_from_angles, force_angle_constant

    angles = {}
    k_theta = np.zeros(len(angle_list))
    theta_0 = np.zeros(len(angle_list))

    #Modified Seminario part goes here ...
    #Connectivity information for Modified Seminario Method
    central_atoms_angles = []

    #A structure is created with the index giving the central atom of the
    #angle, an array then lists the angles with that central atom. 
    #ie. central_atoms_angles{3} contains an array of angles with central atom
    #3
    for i in range(0, len(coords)):
        central_atoms_angles.append([])
        for j in range(0, len(angle_list)):
            if i == angle_list[j][1]:
                #For angle ABC, atoms A C are written to array
                AC_array = [angle_list[j][0],  angle_list[j][2], j]
                central_atoms_angles[i].append(AC_array)

                #For angle ABC, atoms C A are written to array
                CA_array = [angle_list[j][2],  angle_list[j][0], j]
                central_atoms_angles[i].append(CA_array)

    #Sort rows by atom number
    for i in range(0, len(coords)):
        central_atoms_angles[i] = sorted(central_atoms_angles[i],
                                         key=itemgetter(0))

    #Find normals u_PA for each angle
    unit_PA_all_angles = []

    for i in range(0,len(central_atoms_angles)):
        unit_PA_all_angles.append([])
        for j in range(0, len(central_atoms_angles[i])):
        #For the angle at central_atoms_angles[i][j,:] the corresponding
        #u_PA value is found for the plane ABC and bond AB, where ABC
        #corresponds to the order of the arguements
        #This is why the reverse order was also added
            unit_PA_all_angles[i].append(u_PA_from_angles(central_atoms_angles[i][j][0], i, central_atoms_angles[i][j][1], coords))

    #Finds the contributing factors from the other angle terms
    #scaling_factor_all_angles = cell(max(max(angle_list))); %This array will contain scaling factor and angle list position
    scaling_factor_all_angles = []

    for i in range(0,len(central_atoms_angles)):
        scaling_factor_all_angles.append([])
        for j in range(0,len(central_atoms_angles[i])):
            n = 1
            m = 1
            angles_around = 0 
            additional_contributions = 0 
            scaling_factor_all_angles[i].append([0,0]) 
        
            #Position in angle list
            scaling_factor_all_angles[i][j][1] =  central_atoms_angles[i][j][2] 
        
            #Goes through the list of angles with the same central atom
            #And computes the term need for the modified Seminario method    

            #Forwards directions, finds the same bonds with the central atom i  
            while( ( (j + n ) < len(central_atoms_angles[i]) ) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j+n][0] ):
                additional_contributions = additional_contributions + (abs(np.dot(unit_PA_all_angles[i][j][:], unit_PA_all_angles[i][j + n][:])))**2 
                n = n + 1
                angles_around = angles_around + 1
        
            #Backwards direction, finds the same bonds with the central atom i   
            while( ( (j - m ) >= 0 ) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j-m][0] ):
                additional_contributions = additional_contributions + (abs(np.dot(unit_PA_all_angles[i][j][:], unit_PA_all_angles[i][j - m][:] ) ) )**2
                m = m + 1
                angles_around =  angles_around + 1
        
            if (n != 1 or m != 1):
                #Finds the mean value of the additional contribution
                #To change to normal Seminario method comment out + part 
                scaling_factor_all_angles[i][j][0] = 1 + ( additional_contributions / (m  + n - 2) )  
            else:
                scaling_factor_all_angles[i][j][0] = 1

    scaling_factors_angles_list = []
    for i in range(0,len(angle_list) ):
        scaling_factors_angles_list.append([]) 

    #Orders the scaling factors according to the angle list
    for i in range(0,len(central_atoms_angles)):
        for j in range(0,len(central_atoms_angles[i]) ):
            scaling_factors_angles_list[scaling_factor_all_angles[i][j][1]].append(scaling_factor_all_angles[i][j][0]) 

    #Finds the angle force constants with the scaling factors included for each angle
    for i, angle in enumerate(angle_list):
        #Ensures that there is no difference when the ordering is changed 
        [AB_k_theta, AB_theta_0] = force_angle_constant( angle[0], angle[1], angle[2], bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][0], scaling_factors_angles_list[i][1] ) 
        [BA_k_theta, BA_theta_0] = force_angle_constant( angle[2], angle[1], angle[0], bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][1], scaling_factors_angles_list[i][0] ) 
        k_theta[i] = ( AB_k_theta + BA_k_theta ) / 2
        theta_0[i] = ( AB_theta_0 +  BA_theta_0 ) / 2
    
        #Vibrational_scaling takes into account DFT deficities/ anharmocity 
        k_theta[i] =  k_theta[i] * vibrational_scaling_squared

        a_type = sorted([atom_names[angle[0]], atom_names[angle[2]]])
        a_type = '-'.join([a_type[0], atom_names[angle[1]], a_type[1]])

        if a_type not in angles.keys():
            angles[a_type] = [[i], theta_0[i], k_theta[i], 1]
        else:
            angles[a_type][0].append(i)
            angles[a_type][1] += theta_0[i]
            angles[a_type][2] += k_theta[i]
            angles[a_type][3] += 1

    for a_type in angles.keys():
        angles[a_type][1] = angles[a_type][1] / angles[a_type][3]
        angles[a_type][2] = angles[a_type][2] * 4.184 * 2 / angles[a_type][3]
        for i in angles[a_type][0]:
            angle_list[i] = [atom + 1 for atom in angle_list[i]]
            angle_list[i].extend([angles[a_type][1], angles[a_type][2]])

    with open(out_file, 'a') as gmx:    
        gmx.write("\n[angles]\n")
        gmx.write(";   ai    aj    ak   f       th0         kth\n")
        for angle in angle_list:
            gmx.write("{:>6}{:>6}{:>6}   1{:>10.2f}{:>12.3f}\n".format(*angle))
