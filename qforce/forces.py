import numpy as np
import math
from numba import jit
"""
    Calculation of forces on x, y, z directions and also seperately on
    term_ids. So forces are grouped seperately for each unique FF
    parameter.
"""

@jit(nopython=True)
def calc_bonds(coords, atoms, r0, term, force):
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]]) 
#   energy[t] += 0.5 * (r12-r0)**2 
    f =  - vec12 * (r12-r0) / r12
    force[atoms[0], term] += f
    force[atoms[1], term] -= f
    return force

@jit(nopython=True)
def calc_angles(coords, atoms, theta0, term, force):
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    vec32, r32 = get_dist(coords[atoms[2]], coords[atoms[1]])
    theta = get_angle(vec12, vec32)
    cos_theta = np.cos(theta)
    dtheta = theta - theta0  
#   energy[t] += 0.5 * dtheta**2
    st = - dtheta / np.sqrt(1. - cos_theta**2)
    sth = st * cos_theta
    c13 = st / r12 / r32
    c11 = sth / r12 / r12
    c33 = sth / r32 / r32

    f1 = c11 * vec12 - c13 * vec32
    f3 = c33 * vec32 - c13 * vec12
    force[atoms[0], term] += f1
    force[atoms[2], term] += f3
    force[atoms[1], term] += - f1 - f3
    return force

#def calc_g96angles(coords, angles, force):
#    for a, theta0, t in zip(angles.atoms, angles.minima, angles.term_ids):
#        r_ij, r12 = get_dist(coords[a[0]], coords[a[1]])
#        r_kj, r32 = get_dist(coords[a[2]], coords[a[1]])
#        theta = get_angle(r_ij, r_kj)
#        cos_theta = np.cos(theta)
#        dtheta = theta - theta0
##        energy[t] += 0.5 * dtheta**2
#        
#        rij_1    = 1 / np.sqrt(np.inner(r_ij, r_ij))
#        rkj_1    = 1 / np.sqrt(np.inner(r_kj, r_kj))
#        rij_2    = rij_1*rij_1
#        rkj_2    = rkj_1*rkj_1
#        rijrkj_1 = rij_1*rkj_1
#
#        f1    = dtheta*(r_kj*rijrkj_1 - r_ij*rij_2*cos_theta);
#        f3    = dtheta*(r_ij*rijrkj_1 - r_kj*rkj_2*cos_theta);
#        f2    = - f1 - f3;
#        
#        force[a[0], :, t] += f1;
#        force[a[1], :, t] += f2;
#        force[a[2], :, t] += f3;
#    return force

#def calc_quartic_angles(coords, angles, force):
#    for a, theta0, t in zip(angles.atoms, angles.minima, angles.term_ids):

        # QUARTIC : SAME AS ANGLES BUT 5 TIMES. LOOP OVER!
        
        
#    return force

#@jit(nopython=True)

def calc_dihedrals(coords, atoms, phi0, term, force): 
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(*coords[atoms])
    force = calc_imp_dih(force, atoms, phi, phi0, vec_ij, vec_kj, vec_kl, 
                         cross1, cross2, term)
    return force
#@jit(nopython=True) 
def calc_imp_dih(force, a, phi, phi0, vec_ij, vec_kj, vec_kl, cross1, cross2, n):
    dphi = phi - phi0
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi # dphi between -pi to pi
#    energy[n] += 0.5 * dphi**2
    force = calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2,
                           dphi, n)
    return force
#@jit(nopython=True)
def calc_per_dih(force, a, phi, phi0, vec_ij, vec_kj, vec_kl, cross1, cross2, n):
    mult = 3.
    mdphi = mult * phi - phi0
    ddphi =  mult * np.sin(mdphi)
#    energy[n] = 1 + np.cos(mdphi)
    force = calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2,
                           ddphi, n)
    return force

#def calc_rb_dih(force, a, phi, vec_ij, vec_kj, vec_kl, cross1, cross2,
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
##            energy += kd[0]
#            continue
#        ddphi += i * cos_factor * c 
#        cos_factor *= cos_phi
##        energy += c * cos_factor
#        
#    ddphi *= - sin_phi
#    force = calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2, 
#                           ddphi)
#    return force

@jit("f8(f8[:], f8[:])", nopython=True)
def dot_prod(a, b):
    x = a[0]*b[0]
    y = a[1]*b[1]
    z = a[2]*b[2]
    return x+y+z

@jit("f8[:,:,:](f8[:,:,:], i8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4)", 
     nopython=True)
def calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi, n):
    inner1  = dot_prod(cross1, cross1)
    inner2  = dot_prod(cross2, cross2)
    nrkj2 = dot_prod(vec_kj, vec_kj)
    
    nrkj_1 = 1 / np.sqrt(nrkj2)
    nrkj_2 = nrkj_1 * nrkj_1
    nrkj = nrkj2 * nrkj_1
    aa = -ddphi * nrkj / inner1
    f_i = aa * cross1
    bb = ddphi * nrkj / inner2
    f_l = bb * cross2
    p = dot_prod(vec_ij, vec_kj) * nrkj_2
    q = dot_prod(vec_kl, vec_kj) * nrkj_2
    uvec = p * f_i
    vvec = q * f_l
    svec = uvec - vvec
    
    f_j = f_i - svec
    f_k = f_l + svec
    
    force[a[0],n] += f_i
    force[a[1],n] -= f_j
    force[a[2],n] -= f_k
    force[a[3],n] += f_l
    return force

#@jit(nopython=True)
def calc_pairs(coords, natoms, neighbors, force):
    eps = 1389.35458

    q =  [0.041, -0.066, 0.041, -0.066, 0.041, 0.041, -0.017, 0.063, 0.063, 
          -0.279, -0.017, 0.063, 0.063, -0.388, 0.139, 0.139, 0.139]
    
    c6 = [ 8.464e-05, 0.0023406244, 8.464e-05, 0.0023406244, 8.464e-05, 
          8.464e-05, 0.0023406244, 8.464e-05, 8.464e-05, 0.00863041,
          0.0023406244, 8.464e-05, 8.464e-05, 0.002025, 8.464e-05, 8.464e-05,
          8.464e-05]
    c12 = [1.5129e-08, 4.937284e-06, 1.5129e-08, 4.937284e-06, 1.5129e-08, 
           1.5129e-08, 4.937284e-06, 1.5129e-08, 1.5129e-08, 2.025e-05,
           4.937284e-06, 1.5129e-08, 1.5129e-08, 1e-06, 1.5129e-08, 1.5129e-08, 
           1.5129e-08]
    
    c6 = np.array(c6) * 1e6
    c12 = np.array(c12) * 1e12
    for i in range(natoms): 
        for j in range(i+1, natoms):
            if all([j not in neighbors[c][i] for c in range(2)]):
                vec, r = get_dist(coords[i], coords[j])
                
                q12er = q[i]*q[j]*eps/r
                r_2 = 1/r**2
                
                r_6 = r_2**3
                c6_r6 = np.sqrt(c6[i]*c6[j]) * r_6
                c12_r12 = np.sqrt(c12[i]*c12[j]) * r_6**2
#                energy += q12er + c12_r12 - c6_r6
                f = q12er * r_2 + (12*c12_r12 - 6*c6_r6) * r_2
                #tiny numerical disagreement with gromacs for lj forces??
                # double check!
                for k in range(3):
                    fk = f * vec[k]
                    force[i,k,-1] -= fk
                    force[j,k,-1] += fk
    return force

@jit(nopython=True)
def get_dist(coord1, coord2):
    vec = coord1 - coord2
    r = norm(vec)
    return vec, r

@jit(nopython=True)
def get_angle(vec1, vec2):
    dot = np.dot(vec1/norm(vec1), vec2/norm(vec2))
    return math.acos(dot)


@jit(nopython=True)
def get_dihed(coord1, coord2, coord3, coord4):
    vec12, r12 = get_dist(coord1, coord2)
    vec32, r32 = get_dist(coord3, coord2)
    vec34, r34 = get_dist(coord3, coord4)
    cross1 = cross_prod(vec12, vec32)
    cross2 = cross_prod(vec32, vec34)
    phi = get_angle(cross1, cross2)
    if dot_prod(vec12, cross2) < 0:
        phi = - phi
    return phi, vec12, vec32, vec34, cross1, cross2

@jit("f8[:](f8[:], f8[:])", nopython=True)
def cross_prod(a, b):
    c = np.empty(3, dtype=np.double)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

@jit("f8(f8[:])", nopython=True)
def norm(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    