def modified_Seminario_method(inp):
    #   Program to implement the Modified Seminario Method
    #   Written by Alice E. A. Allen, TCM, University of Cambridge
    #   Reference: AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018)
    #   doi:10.1021/acs.jctc.7b00785

    # Edited by Selim Sami to implement into the qforce package

    import numpy as np
    import os.path 

    from .input_data_processing import input_data_processing
    from .bonds_calculated_printed import bonds_calculated_printed
    from .angles_calculated_printed import  angles_calculated_printed

    print('\n----- Modified Seminario Method -----')
    print('This module is originally written by Alice E. A. Allen')
    print('If you use this module please cite: 10.1021/acs.jctc.7b00785\n')

    #Square the vibrational scaling used for frequencies
    vibrational_scaling_squared = inp.vibr_coef**2; 

    #Import all input data
    (bond_list, angle_list, coords, N, hessian, atom_names, 
     equiv_list) = input_data_processing(inp)

    #Find bond lengths
    bond_lengths = np.zeros((N, N))

    for i in range (0,N):
        for j in range(0,N):
            diff_i_j = np.array(coords[i,:]) - np.array(coords[j,:])
            bond_lengths[i][j] =  np.linalg.norm(diff_i_j)

    eigenvectors = np.empty((3, 3, N, N), dtype=complex)
    eigenvalues = np.empty((N, N, 3), dtype=complex)
    partial_hessian = np.zeros((3, 3))

    for i in range(0,N):
        for j in range(0,N):
            partial_hessian = hessian[(i * 3):((i + 1)*3),(j * 3):((j + 1)*3)]
            [a, b] = np.linalg.eig(partial_hessian)
            eigenvalues[i,j,:] = (a)
            eigenvectors[:,:,i,j] = (b)

    out_file = "{}_bondangle.itp".format(os.path.splitext(inp.fchk_file)[0])

    # The bond values are calculated and written to file
    bonds_calculated_printed(vibrational_scaling_squared, bond_list, 
                             bond_lengths, atom_names, eigenvalues, 
                             eigenvectors, coords, out_file)
    # The angle values are calculated and written to file
    angles_calculated_printed(vibrational_scaling_squared, angle_list, 
                              bond_lengths, atom_names, eigenvalues, 
                              eigenvectors, coords, out_file)

    print ("\nDone! QM optimized bonds and angles can be found in:\n{}"
           .format(out_file))