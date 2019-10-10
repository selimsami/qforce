import numpy as np

def relax_drude(coords, mol, q_esp):
    """
    Drude energy minimization using the same procedure as GROMACS.
    Work in progress - to be implemented with polarizable FF derivataions.
    """
    EPS0 = 1389.35458 # kJ*ang/mol/e2
    gro2cgs = 0.376730313488167
    cgs2au = 18.8972612545**3 / 4 / np.pi

    k = []
    pair_list = []
    mu = []
    q_a = q_esp + 8.0
    print(sum(q_esp))
    for i in q_a:
        print(i)

    for p in mol.polar:
        k.append(64.0 * EPS0/p) #units???

    for i in range(mol.n_atoms):
        for j in range(i+1, mol.n_atoms):
            if j not in mol.neighbors[0][i]+mol.neighbors[1][i]+mol.neighbors[2][i]:
                pair_list.append([i,j])

    print("No Field")
    xyz = minimize_drude(coords, k, mol, q_a, pair_list, -1)
    print("Field-X")
    x = minimize_drude(coords, k, mol, q_a, pair_list, 0)
    print("Field-Y")
    y = minimize_drude(coords, k, mol, q_a, pair_list, 1)
    print("Field-Z")
    z = minimize_drude(coords, k, mol, q_a, pair_list, 2)
    
    mu.append(sum(xyz[:,0] * q_a - 8 * x[:,0]))
    mu.append(sum(xyz[:,1] * q_a - 8 * y[:,1]))
    mu.append(sum(xyz[:,2] * q_a - 8 * z[:,2]))
    mu = np.array(mu)
    
    mu = mu / 0.20819434 / 0.2 * gro2cgs * cgs2au
    
    print("Polarizabilities: ", mu)
    
def minimize_drude(coords, k, mol, q_a, pair_list, field):
    xyz = np.array(coords)# + np.array([0, 0, 0.1]) #
    ###
    f_tol = 0.01 #kj/mol/ang
    s_scale_min = 0.8
    s_scale_incr = 0.2
    s_scale_max = 1.2
    s_scale_mult= (s_scale_max - s_scale_min) / s_scale_incr
    n_max_step = 30
    step = 0
    ###
    force = calc_force(xyz, coords, k, mol, q_a, pair_list, field)
    norm = np.linalg.norm(force, axis=1)
    f_rms = np.sqrt(np.mean(norm**2))
    f_rms_min = f_rms

    print(f"step_no: {step}, f_rms: {f_rms}")
    
    while f_rms > f_tol and step < n_max_step:
        step += 1
        if step == 1:
            step_size = 1 / np.tile(np.array([k]).T, 3)
        else:
            df = force - f_old
            k_est = np.divide(-dx, df, out=np.full((mol.n_atoms,3), np.inf), where=df!=0)
            
            for i, s_atom in enumerate(step_size):
                for d in range(3):
                    if k_est[i][d] != np.inf:
                        s_atom[d] = s_scale_min*s_atom[d] + (s_scale_incr * 
                              min(s_scale_mult*s_atom[d], max(k_est[i][d], 0)))
                    else:
                        s_atom[d] *= s_scale_max            
        f_old = force
        xyz_old = xyz
        dx = f_old * step_size
        xyz = xyz_old + dx
        
        force = calc_force(xyz, coords, k, mol, q_a, pair_list, field)
        norm = np.linalg.norm(force, axis=1)
        f_rms = np.sqrt(np.mean(norm**2))
        
        if f_rms < f_rms_min:
            f_rms_min = f_rms
        else:
            print("scaling step size")
            step_size *= 0.8
            
        print(f"step_no: {step}, f_rms: {f_rms}")
    
    return xyz

def calc_force(d_xyz, xyz, k, mol, q_a, pair_list, field):
    jaco = np.zeros((len(xyz),3))
    field_strength = 0.2 # V/Ang
    ev2kjmol = 96.48533645956869
    #drude
    for i, (d,a) in enumerate(zip(d_xyz,xyz)):
        vec12, _ = get_dist(a, d)
        jaco[i] += k[i] * vec12
    #thole
    for i, j, afac in mol.thole:
        f = calc_thole(d_xyz[i], d_xyz[j], afac, 64) #drude-drude
        jaco[i] += f
        jaco[j] -= f 
        jaco[i] += calc_thole(d_xyz[i], xyz[j], afac, -64) #drude-atom
        jaco[j] += calc_thole(d_xyz[j], xyz[i], afac, -64) #atom-drude
    #Coulomb
    for i, j in pair_list:
        f = calc_coulomb(d_xyz[i], d_xyz[j], 64) #drude-drude
        jaco[i] += f
        jaco[j] -= f
        jaco[i] += calc_coulomb(d_xyz[i], xyz[j], -8*q_a[j]) #drude-atom
        jaco[j] += calc_coulomb(d_xyz[j], xyz[i], -8*q_a[i]) #atom-drude
    #Applied field
    if field >= 0:
        for i in range(mol.n_atoms):
            jaco[i,field] += -8 * ev2kjmol * field_strength / 10
    return jaco

def calc_thole(coord1, coord2, afac, qq):
    EPS0 = 1389.35458 # kJ*ang/mol/e2
    vec12, r12 = get_dist(coord1, coord2)
    r12_bar = r12 * afac
    ebar = np.exp(-r12_bar)
    v0 = qq * EPS0 / r12 
    v1 = 1 - (1 + 0.5*r12_bar) * ebar
    f = ((v0/r12)*v1 - v0*0.5*afac*ebar*(r12_bar+1)) / r12
    return vec12 * f

def calc_coulomb(coord1, coord2, qq):
    EPS0 = 1389.35458
    vec12, r = get_dist(coord1, coord2)
    f = qq * EPS0 / r**3
    return vec12 * f

def get_dist(coord1, coord2):
    vec = coord1 - coord2
    r = (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5
    return vec, r