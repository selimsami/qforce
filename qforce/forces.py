import numpy as np
"""
    Calculation of forces on x, y, z directions and also seperately on
    term_ids. So forces are grouped seperately for each unique FF
    parameter.
"""
def calc_bonds(coords, bonds, energy, force):
    for a, r0, t in zip(bonds.atoms, bonds.minima, bonds.term_ids):
        vec12, r12 = get_dist(coords[a[0]], coords[a[1]])
        
        energy[t] += 0.5 * (r12-r0)**2
        
        f =  - vec12 * (r12-r0) / r12
        force[a[0], :, t] += f
        force[a[1], :, t] -= f
    return energy, force

def calc_angles(coords, angles, energy, force):
    for a, theta0, t in zip(angles.atoms, angles.minima, angles.term_ids):
        vec12, r12 = get_dist(coords[a[0]], coords[a[1]])
        vec32, r32 = get_dist(coords[a[2]], coords[a[1]])
        theta = get_angle(vec12, vec32)
        cos_theta = np.cos(theta)
        dtheta = theta - theta0
        
        energy[t] += 0.5 * dtheta**2
        
        st = - dtheta / np.sqrt(1 - cos_theta**2)
        sth = st * cos_theta
        c13 = st / r12 / r32
        c11 = sth / r12 / r12
        c33 = sth / r32 / r32

        f1 = c11 * vec12 - c13 * vec32
        f3 = c33 * vec32 - c13 * vec12
        force[a[0], :, t] += f1
        force[a[2], :, t] += f3
        force[a[1], :, t] += - f1 - f3
    return energy, force

def calc_g96angles(coords, angles, energy, force):
    for a, theta0, t in zip(angles.atoms, angles.minima, angles.term_ids):
        r_ij, r12 = get_dist(coords[a[0]], coords[a[1]])
        r_kj, r32 = get_dist(coords[a[2]], coords[a[1]])
        theta = get_angle(r_ij, r_kj)
        cos_theta = np.cos(theta)
        dtheta = theta - theta0
        energy[t] += 0.5 * dtheta**2
        
        rij_1    = 1 / np.sqrt(np.inner(r_ij, r_ij))
        rkj_1    = 1 / np.sqrt(np.inner(r_kj, r_kj))
        rij_2    = rij_1*rij_1
        rkj_2    = rkj_1*rkj_1
        rijrkj_1 = rij_1*rkj_1

        f1    = dtheta*(r_kj*rijrkj_1 - r_ij*rij_2*cos_theta);
        f3    = dtheta*(r_ij*rijrkj_1 - r_kj*rkj_2*cos_theta);
        f2    = - f1 - f3;
        
        force[a[0], :, t] += f1;
        force[a[1], :, t] += f2;
        force[a[2], :, t] += f3;
    return energy, force

#def calc_quartic_angles(coords, angles, energy, force):
#    for a, theta0, t in zip(angles.atoms, angles.minima, angles.term_ids):

        # QUARTIC : SAME AS ANGLES BUT 5 TIMES. LOOP OVER!
        
        
#    return energy, force

def calc_dihedrals(coords, stiff, improper, energy, force): 
    for a, phi0, n in zip(stiff.atoms + improper.atoms , stiff.minima + 
                          improper.minima, stiff.term_ids + improper.term_ids):

        phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(*coords[a])
        energy, force = calc_imp_dih(energy, force, a, phi, phi0, vec_ij,
                                    vec_kj, vec_kl, cross1, cross2, n)
    return energy, force

#def calc_pairs(coords, natoms, mol.connect, mol.types, energy, force, pair, ff):
#    eps = 1389.35458
#
#    for i in range(natoms): 
#        for j in range(i+1, natoms):
#            if all([j not in mol.connect[c][i] for c in range(2)]):
#                types = [mol.types[i], mol.types[j]]
#                pair = parameters(', '.join('{}_{}'.format(t, z) for t in 
#                                            ["q", "c6", "c12"] for z in types))
#                qi, qj, c6i, c6j, c12i, c12j = pair
#                
#                vec, r = get_dist(coords[i], coords[j])
#                q12er = qi*qj*eps/r
#                r2 = 1/r**2
#                r6 = r2**3
#                c6r6 = sqrt(c6i*c6j) * r6
#                c12r12 = sqrt(c12i*c12j) * r6**2
#    
#                energy += q12er + c12r12 - c6r6
#                f = q12er * r2 + (12*c12r12 - 6*c6r6) * r2
#                
#                #small numerical disagreement with gromacs for lj forces
#                for k in range(3):
#                    fk = f * vec[k]
#                    force[i,k] += fk
#                    force[j,k] -= fk
#    return energy, force

    
def calc_imp_dih(energy, force, a, phi, phi0, vec_ij, vec_kj, vec_kl, cross1,
                 cross2, n):
    dphi = phi - phi0
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi # dphi between -pi to pi
    
    energy[n] += 0.5 * dphi**2
    
    force = calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2,
                           dphi, n)
    return energy, force

def calc_per_dih(energy, force, a, phi, phi0, vec_ij, vec_kj, vec_kl, cross1,
                 cross2, n):
    mult = 2
    mdphi = mult*phi - phi0
    ddphi = mult * np.sin(mdphi)
    
    energy[n] = 1 + np.cos(mdphi)

    force = calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2,
                           ddphi, n)
    return energy, force

#def calc_rb_dih(energy, force, a, phi, vec_ij, vec_kj, vec_kl, cross1, cross2,
#                n):
#    kd = [1, 0, -1, 0, 0, 0]
#    phi += np.pi
#    cos_phi = np.cos(phi)
#    sin_phi = np.sin(phi)
#
#    for i, c in enumerate(kd):
#        if i == 0:
#            cos_factor = 1
#            ddphi = 0
#            energy += kd[0]
#            continue
#        ddphi += i * cos_factor * c 
#        cos_factor *= cos_phi
#        energy += c * cos_factor
#        
#    ddphi *= - sin_phi
#    force = calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2, 
#                           ddphi)
#    return energy, force

def calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi, n):
    inner1  = np.inner(cross1, cross1)
    inner2  = np.inner(cross2, cross2)
    nrkj2 = np.inner(vec_kj, vec_kj)
    
    nrkj_1 = 1 / np.sqrt(nrkj2)
    nrkj_2 = nrkj_1 * nrkj_1
    nrkj = nrkj2 * nrkj_1
    aa = -ddphi * nrkj / inner1
    f_i = aa * cross1
    bb = ddphi * nrkj / inner2
    f_l = bb * cross2
    p = np.inner(vec_ij, vec_kj) * nrkj_2
    q = np.inner(vec_kl, vec_kj) * nrkj_2
    uvec = p * f_i
    vvec = q * f_l
    svec = uvec - vvec
    
    f_j = f_i - svec
    f_k = f_l + svec
    
    force[a[0],:,n] += f_i
    force[a[1],:,n] -= f_j
    force[a[2],:,n] -= f_k
    force[a[3],:,n] += f_l
    return force

def get_dist(coord1, coord2):
    vec = coord1 - coord2
    r = np.linalg.norm(vec)
    return vec, r

def get_angle(vec1, vec2):
    cross = np.cross(vec1, vec2)
    norm = np.linalg.norm(cross)
    inner = np.inner(vec1, vec2)
    theta = np.arctan2(norm, inner)
    return theta

def get_dihed(coord1, coord2, coord3, coord4):
    vec12, r12 = get_dist(coord1, coord2)
    vec32, r32 = get_dist(coord3, coord2)
    vec34, r34 = get_dist(coord3, coord4)
    cross1 = np.cross(vec12, vec32)
    cross2 = np.cross(vec32, vec34)
    phi = get_angle(cross1, cross2)
    if np.inner(vec12, cross2) < 0:
        phi = - phi
    return phi, vec12, vec32, vec34, cross1, cross2