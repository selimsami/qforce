import numpy as np
import os
from .molecule import Molecule
from .read_forcefield import Forcefield
from .write_forcefield import write_itp
from .write_forcefield import write_gro

def polarize(inp):
    """
    Generate the polarizable versions of the input forcefield 
    for both GRO and ITP files
    """
    polar_coords = []
    ff = Forcefield(itp_file = inp.itp_file, gro_file = inp.coord_file)
    mol = Molecule(np.array(ff.coords)*10, ff.atomids[0:ff.natom], inp)
    ff.exclu = [[] for i in range(ff.natom)]

#    polar_dict = { 1: 0.000413835,  6: 0.00145,  7: 0.000971573, 
#                   8: 0.000851973,  9: 0.000444747, 16: 0.002474448, 
#                  17: 0.002400281, 35: 0.003492921, 53: 0.005481056}
#    polar_dict = { 1: 0.000413835,  6: 0.001288599,  7: 0.000971573, 
#                   8: 0.000851973,  9: 0.000444747, 16: 0.002474448, 
#                  17: 0.002400281, 35: 0.003492921, 53: 0.005481056}
#    polar_dict = { 1: 0.000205221,  6: 0.000974759,  7: 0.000442405, 
#                   8: 0.000343551,  9: 0.000220884, 16: 0.001610042, 
#                  17: 0.000994749, 35: 0.001828362, 53: 0.002964895}
    polar_dict = { 1: 0.00045330, 6: 0.00130300, 7: 0.00098850, 8: 0.00083690} # current PTEGs

    # add drude atom type
    ff.atom_types.append(["DP", 0, 0, "S", 0, 0])

    #add coords
    ff.n_mol = int(ff.gro_natom/ff.natom)
    for i in range(ff.n_mol):
        polar_coords.extend(ff.coords[i*ff.natom:(i+1)*ff.natom]*2)
    ff.coords = polar_coords
    
    for i in range(ff.natom):

        # add exclusions for nrexcl=3
        for n3 in mol.neighbors[1][i]:
            ff.exclu[i].extend([n3+ff.natom+1])
        for n4 in mol.neighbors[2][i]:
            if sorted([i+1, n4+1]) not in ff.pairs:
                ff.exclu[i].extend([n4+ff.natom+1, n4+1])
                
        # add exclusions for nrexcl=2
#        for n2 in mol.neighbors[0][i]:
#            ff.exclu[i].extend([n2+ff.natom+1])
#        for n3 in mol.neighbors[1][i]:
#            if sorted([i+1, n3+1]) not in ff.pairs:
#                ff.exclu[i].extend([n3+ff.natom+1, n3+1])            
    
    
        # add atoms
        ff.atoms.append([i+ff.natom+1, "DP", ff.atoms[i][2]+ff.maxresnr, 
                         ff.atoms[i][3], "D{}".format(i+1), ff.atoms[i][5],
                         -8, 0])
        ff.atoms[i][6] = ff.atoms[i][6]+8
    
        # add polarization
        ff.polar.append([i+1, i+1+ff.natom, 1, polar_dict[ff.atomids[i]]])
        
        # add thole
        for a in mol.neighbors[0][i]+mol.neighbors[1][i]+mol.neighbors[2][i]:
            if i < a:
                ff.thole.append([i+1, i+ff.natom+1, a+1, a+ff.natom+1, "2", 
                                 "2.6", polar_dict[ff.atomids[i]],
                                 polar_dict[ff.atomids[a]]]) #2.1304 2.899

    polar_itp = "{}_polar.itp".format(os.path.splitext(inp.itp_file)[0])
    polar_gro = "{}_polar.gro".format(os.path.splitext(inp.coord_file)[0])
    write_itp(ff, polar_itp, False)
    write_gro(ff, polar_gro)
    
    print("Done!")
    print(f"Polarizable coordinate file in: {polar_gro}")
    print(f"Polarizable force field file in: {polar_itp}")