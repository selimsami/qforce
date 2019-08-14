import math
import numpy as np

def input_data_processing(inp):
    #This function takes input data that is need from the files supplied
    #Function extracts input coords and hessian from .fchk file, bond and angle
    #lists from .log file and atom names if a z-matrix is supplied

    from ..read_qm_out import QM
    from ..molecule import Molecule
    
    qm = QM(out_files = [inp.qm_freq_out], job_type = "freq")
    bond_list = [bond for bond in qm.bond_atoms]
    angle_list = [angle for angle in qm.angle_atoms]
    
    mol = Molecule(qm.coords[0], qm.atomids, inp)
    
    #Gets Hessian in unprocessed form and writes .xyz file too 
    unprocessed_Hessian, coords = read_qm_fchk(inp.fchk_file)
    length_hessian = 3 * qm.natoms
    hessian = np.zeros((length_hessian, length_hessian))
    m = 0

    #Write the hessian in a 2D array format 
    for i in range(length_hessian):
        for j in range (0,(i + 1)):
            hessian[i][j] = unprocessed_Hessian[m]
            hessian[j][i] = unprocessed_Hessian[m]
            m = m + 1
        
    #Change from Hartree/bohr to kcal/mol /ang
    hessian = (hessian * (627.509391) )/ (0.529**2) ; 

    return (bond_list, angle_list, coords, qm.natoms, hessian, 
            mol.types, mol.list)

def read_qm_fchk(fchk_file):
    coords, hessian = [], [] 
    bohr2ang = 0.52917721067
    
    with open(fchk_file, "r", encoding='utf-8') as fchk:
        for line in fchk:
            if "Number of atoms  " in line:
                n_atoms = int(line.split()[4])
                
            if "Current cartesian coordinates   " in line:
                n_line = math.ceil(3*n_atoms/5)
                for i in range(n_line):
                    line = fchk.readline()
                    coords.extend(line.split())

            if "Cartesian Force Constants  " in line:
                n_line = math.ceil(3*n_atoms*(3*n_atoms + 1) / 10)
                for i in range(n_line):                                         
                    line = fchk.readline()
                    hessian.extend(line.split())

    coords, hessian = np.asfarray(coords,float), np.asfarray(hessian,float)
    coords  = np.reshape(coords, (-1,3)) * bohr2ang
    return hessian, coords
