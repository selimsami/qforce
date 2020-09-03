import numpy as np
import math
from numba import jit
"""
    Calculation of forces on x, y, z directions and also seperately on
    term_ids. So forces are grouped seperately for each unique FF
    parameter.
"""


@jit(nopython=True)
def calc_bonds(coords, atoms, r0, fconst, force):
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    energy = 0.5 * fconst * (r12-r0)**2
    f = - fconst * vec12 * (r12-r0) / r12
    force[atoms[0]] += f
    force[atoms[1]] -= f
    return energy


@jit(nopython=True)
def calc_angles(coords, atoms, theta0, fconst, force):
    theta, vec12, vec32, r12, r32 = get_angle(coords[atoms])
    cos_theta = math.cos(theta)
    cos_theta_sq = cos_theta**2
    dtheta = theta - theta0
    energy = 0.5 * fconst * dtheta**2
    if cos_theta_sq < 1:
        st = - fconst * dtheta / np.sqrt(1. - cos_theta_sq)
        sth = st * cos_theta
        c13 = st / r12 / r32
        c11 = sth / r12 / r12
        c33 = sth / r32 / r32

        f1 = c11 * vec12 - c13 * vec32
        f3 = c33 * vec32 - c13 * vec12
        force[atoms[0]] += f1
        force[atoms[2]] += f3
        force[atoms[1]] -= f1 + f3
    return energy


def calc_cross_bond_angle(coords, atoms, r0s, fconst, force):
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    vec32, r32 = get_dist(coords[atoms[2]], coords[atoms[1]])
    vec13, r13 = get_dist(coords[atoms[0]], coords[atoms[1]])

    s1 = r12 - r0s[0]
    s2 = r32 - r0s[1]
    s3 = r13 - r0s[2]

    energy = fconst * s3 * (s1+s2)

    k1 = - fconst * s3/r12
    k2 = - fconst * s3/r32
    k3 = - fconst * (s1+s2)/r13

    f1 = k1*vec12 + k3*vec13
    f3 = k2*vec32 + k3*vec13

    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3
    return energy


def calc_imp_diheds(coords, atoms, phi0, fconst, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    dphi = phi - phi0
    dphi = np.pi - (dphi + np.pi) % (2 * np.pi)  # dphi between -pi to pi
    energy = 0.5 * fconst * dphi**2
    ddphi = - fconst * dphi
    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def calc_rb_diheds(coords, atoms, params, fconst, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    phi += np.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    energy = params[0]
    ddphi = 0
    cos_factor = 1

    for i in range(1, 6):
        ddphi += i * cos_factor * params[i]
        cos_factor *= cos_phi
        energy += cos_factor * params[i]

    ddphi *= - sin_phi
    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def calc_inversion(coords, atoms, phi0, fconst, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    phi += np.pi

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    c0, c1, c2 = convert_to_inversion_rb(fconst, phi0)

    energy = c0

    ddphi = c1
    energy += cos_phi * c1

    ddphi += 2 * c2 * cos_phi
    energy += cos_phi**2 * c2

    ddphi *= - sin_phi
    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def calc_periodic_dihed(coords, atoms, phi0, fconst, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    mult = 3
    phi0 = 0

    mdphi = mult * phi - phi0
    ddphi = fconst * mult * np.sin(mdphi)

    energy = fconst * (1 + np.cos(mdphi))

    force = calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@jit(nopython=True)
def convert_to_inversion_rb(fconst, phi0):
    cos_phi0 = np.cos(phi0)
    c0 = fconst * cos_phi0**2
    c1 = 2 * fconst * cos_phi0
    c2 = fconst
    return c0, c1, c2


@jit("f8(f8[:], f8[:])", nopython=True)
def dot_prod(a, b):
    x = a[0]*b[0]
    y = a[1]*b[1]
    z = a[2]*b[2]
    return x+y+z


@jit("void(f8[:,:], i8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8)",
     nopython=True)
def calc_dih_force(force, a, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi):
    inner1 = dot_prod(cross1, cross1)
    inner2 = dot_prod(cross2, cross2)
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

    force[a[0]] += f_i
    force[a[1]] -= f_j
    force[a[2]] -= f_k
    force[a[3]] += f_l


@jit(nopython=True)
def calc_pairs(coords, atoms, params, force):
    c6, c12, qq = params
    vec, r = get_dist(coords[atoms[0]], coords[atoms[1]])
    qq_r = qq/r
    r_2 = 1/r**2
    r_6 = r_2**3
    c6_r6 = c6 * r_6
    c12_r12 = c12 * r_6**2
    energy = qq_r + c12_r12 - c6_r6
    f = (qq_r + 12*c12_r12 - 6*c6_r6) * r_2
    fk = f * vec
    force[atoms[0]] += fk
    force[atoms[1]] -= fk
    return energy


@jit(nopython=True)
def get_dist(coord1, coord2):
    vec = coord1 - coord2
    r = norm(vec)
    return vec, r


@jit(nopython=True)
def get_angle(coords):
    vec12, r12 = get_dist(coords[0], coords[1])
    vec32, r32 = get_dist(coords[2], coords[1])
    dot = np.dot(vec12/r12, vec32/r32)
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.acos(dot), vec12, vec32, r12, r32


@jit(nopython=True)
def get_angle_from_vectors(vec1, vec2):
    dot = np.dot(vec1/norm(vec1), vec2/norm(vec2))
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.acos(dot)


@jit(nopython=True)
def get_dihed(coords):
    vec12, r12 = get_dist(coords[0], coords[1])
    vec32, r32 = get_dist(coords[2], coords[1])
    vec34, r34 = get_dist(coords[2], coords[3])
    cross1 = cross_prod(vec12, vec32)
    cross2 = cross_prod(vec32, vec34)
    phi = get_angle_from_vectors(cross1, cross2)
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


# @jit(nopython=True)
# def calc_quartic_angles(oords, atoms, theta0, fconst, force):
#    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
#    vec32, r32 = get_dist(coords[atoms[2]], coords[atoms[1]])
#    theta = get_angle(vec12, vec32)
#    cos_theta = np.cos(theta)
#    dtheta = theta - theta0
#
#    coefs = fconst * np.array([1, -0.014, 5.6e-5, -7e-7, 2.2e08])
#
#    dtp = dtheta
#    dvdt = 0
#    energy = coefs[0]
#    for i in range(1, 5):
#        dvdt += i*dtp*coefs[i]
#        dtp *= dtheta
#        energy += dtp * coefs[i]
#
#    st = - dvdt / np.sqrt(1. - cos_theta**2)
#    sth = st * cos_theta
#    c13 = st / r12 / r32
#    c11 = sth / r12 / r12
#    c33 = sth / r32 / r32
#
#    f1 = c11 * vec12 - c13 * vec32
#    f3 = c33 * vec32 - c13 * vec12
#    force[atoms[0]] += f1
#    force[atoms[2]] += f3
#    force[atoms[1]] -= f1 + f3
#    return energy
#
# def calc_g96angles(coords, angles, force):
#    for a, theta0, t in zip(angles.atoms, angles.minima, angles.term_ids):
#        r_ij, r12 = get_dist(coords[a[0]], coords[a[1]])
#        r_kj, r32 = get_dist(coords[a[2]], coords[a[1]])
#        theta = get_angle(r_ij, r_kj)
#        cos_theta = np.cos(theta)
#        dtheta = theta - theta0
#        energy[t] += 0.5 * dtheta**2
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
