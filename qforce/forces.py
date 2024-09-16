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
def calc_morse_bonds(coords, atoms, equ, fconst, force):
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    r0, beta = equ[0], equ[1]
    D = 0.5 * fconst / beta**2
    exp = np.exp(-beta*(r12-r0))
    energy = D * (1-exp)**2
    f = - 2 * beta * D * exp * (1-exp) * vec12 / r12
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


@jit(nopython=True)
def calc_cosine_angles(coords, atoms, theta0, fconst, force):
    theta, vec12, vec32, r12, r32 = get_angle(coords[atoms])

    cos_theta = math.cos(theta)
    dtheta = (cos_theta - math.cos(theta0))
    dtheta_sq = dtheta**2
    energy = 0.5 * fconst * dtheta_sq

    dvdt = - fconst * dtheta
    r12r32 = r12*r32

    f1 = dvdt * (vec32 / r12r32 - vec12 / r12**2 * cos_theta)
    f3 = dvdt * (vec12 / r12r32 - vec32 / r32**2 * cos_theta)
    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3

    return energy


@jit(nopython=True)
def calc_cross_bond_bond(coords, atoms, r0s, fconst, force):
    vec12, r12 = get_dist(coords[atoms[0]], coords[atoms[1]])
    vec43, r43 = get_dist(coords[atoms[2]], coords[atoms[3]])

    s1 = r12 - r0s[0]
    s2 = r43 - r0s[1]
    energy = fconst * s1 * s2

    f1 = fconst * vec12 / r12 * s2
    f2 = fconst * vec43 / r43 * s1

    force[atoms[0]] -= f1
    force[atoms[1]] += f1
    force[atoms[2]] -= f2
    force[atoms[3]] += f2

    return energy


@jit(nopython=True)
def calc_cross_bond_angle(coords, atoms, equ, fconst, force):
    theta, vec12, vec32, r12, r32 = get_angle(coords[atoms[:3]])
    vec45, r45 = get_dist(coords[atoms[3]], coords[atoms[4]])

    dtheta = theta - equ[0]
    dr = r45 - equ[1]

    energy = fconst * dtheta * dr

    cos_theta = math.cos(theta)
    cos_theta_sq = cos_theta**2

    st = - fconst * dr / np.sqrt(1. - cos_theta_sq)
    sth = st * cos_theta
    c13 = st / r12 / r32
    c11 = sth / r12 / r12
    c33 = sth / r32 / r32

    f1 = c11 * vec12 - c13 * vec32
    f3 = c33 * vec32 - c13 * vec12
    f4 = vec45 * dtheta / r45

    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3
    force[atoms[3]] -= f4
    force[atoms[4]] += f4

    return energy


@jit(nopython=True)
def calc_cross_bond_cos_angle(coords, atoms, equ, fconst, force):
    theta, vec12, vec32, r12, r32 = get_angle(coords[atoms[:3]])
    vec45, r45 = get_dist(coords[atoms[3]], coords[atoms[4]])

    cos_theta = math.cos(theta)
    dtheta = cos_theta - math.cos(equ[0])
    dr = r45 - equ[1]
    energy = fconst * dtheta * dr

    st = - fconst * dr
    c13 = st / r12 / r32
    c11 = st / r12 / r12
    c33 = st / r32 / r32
    f1 = c13 * vec32 - c11 * vec12 * cos_theta
    f3 = c13 * vec12 - c33 * vec32 * cos_theta
    f4 = vec45 * dtheta / r45

    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3
    force[atoms[3]] -= f4
    force[atoms[4]] += f4

    return energy


@jit(nopython=True)
def calc_cross_bond_urey(coords, atoms, r0s, fconst, force):
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


@jit(nopython=True)
def calc_cross_angle_angle(coords, atoms, equ, fconst, force):
    theta1, vec12, vec32, r12, r32 = get_angle(coords[atoms[:3]])
    theta2, vec45, vec65, r45, r65 = get_angle(coords[atoms[3:]])

    dtheta1 = theta1 - equ[0]
    dtheta2 = theta2 - equ[1]

    energy = fconst * dtheta1 * dtheta2

    cos_theta = math.cos(theta1)
    cos_theta_sq = cos_theta**2
    st = - fconst * dtheta2 / np.sqrt(1. - cos_theta_sq)
    sth = st * cos_theta
    c13 = st / r12 / r32
    c11 = sth / r12 / r12
    c33 = sth / r32 / r32
    f1 = c11 * vec12 - c13 * vec32
    f3 = c33 * vec32 - c13 * vec12

    cos_theta = math.cos(theta2)
    cos_theta_sq = cos_theta**2
    st = - fconst * dtheta1 / np.sqrt(1. - cos_theta_sq)
    sth = st * cos_theta
    c46 = st / r45 / r65
    c44 = sth / r45 / r45
    c66 = sth / r65 / r65
    f4 = c44 * vec45 - c46 * vec65
    f6 = c66 * vec65 - c46 * vec45

    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3

    force[atoms[3]] += f4
    force[atoms[5]] += f6
    force[atoms[4]] -= f4 + f6

    return energy


@jit(nopython=True)
def calc_cross_cos_angle_angle(coords, atoms, equ, fconst, force):
    theta1, vec12, vec32, r12, r32 = get_angle(coords[atoms[:3]])
    theta2, vec45, vec65, r45, r65 = get_angle(coords[atoms[3:]])

    cos_theta1 = math.cos(theta1)
    cos_theta2 = math.cos(theta2)

    dtheta1 = cos_theta1 - math.cos(equ[0])
    dtheta2 = cos_theta2 - math.cos(equ[1])

    energy = fconst * dtheta1 * dtheta2

    st = - fconst * dtheta2
    c13 = st / r12 / r32
    c11 = st / r12 / r12
    c33 = st / r32 / r32
    f1 = c13 * vec32 - c11 * vec12 * cos_theta1
    f3 = c13 * vec12 - c33 * vec32 * cos_theta1

    st = - fconst * dtheta1
    c46 = st / r45 / r65
    c44 = st / r45 / r45
    c66 = st / r65 / r65
    f4 = c46 * vec65 - c44 * vec45 * cos_theta2
    f6 = c46 * vec45 - c66 * vec65 * cos_theta2

    force[atoms[0]] += f1
    force[atoms[2]] += f3
    force[atoms[1]] -= f1 + f3

    force[atoms[3]] += f4
    force[atoms[5]] += f6
    force[atoms[4]] -= f4 + f6

    return energy


def calc_cross_dihed_bond(coords, atoms, equ, fconst, force):
    phi0 = np.radians(equ[2])
    phi = get_dihed(coords[atoms[:4]])[0]
    _, r = get_dist(coords[atoms[4]], coords[atoms[5]])
    v_dihed = 1 + np.cos(equ[1]*phi-phi0)
    v_bond = r - equ[0]
    energy = fconst * v_dihed * v_bond

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi = get_dihed(c_new[atoms[:4]])[0]
            _, r = get_dist(c_new[atoms[4]], c_new[atoms[5]])
            v_dihed = 1 + np.cos(equ[1]*phi-phi0)
            v_bond = r - equ[0]
            e_plus = fconst * v_dihed * v_bond

            c_new[a, j] -= 2*move
            phi = get_dihed(c_new[atoms[:4]])[0]
            _, r = get_dist(c_new[atoms[4]], c_new[atoms[5]])
            v_dihed = 1 + np.cos(equ[1]*phi-phi0)
            v_bond = r - equ[0]
            e_minus = fconst * v_dihed * v_bond

            force[a, j] += (e_minus-e_plus) / (2*move)

    return energy


def calc_cross_cos_cube_dihed_bond(coords, atoms, equ, fconst, force):
    phi = get_dihed(coords[atoms[:4]])[0]
    _, r = get_dist(coords[atoms[4]], coords[atoms[5]])

    v_dihed = (np.cos(phi)+1)**4
    v_bond = r - equ[0]
    energy = fconst * v_dihed * v_bond

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi = get_dihed(c_new[atoms[:4]])[0]
            _, r = get_dist(c_new[atoms[4]], c_new[atoms[5]])
            v_dihed = (np.cos(phi)+1)**4
            v_bond = r - equ[0]
            e_plus = fconst * v_dihed * v_bond

            c_new[a, j] -= 2*move
            phi = get_dihed(c_new[atoms[:4]])[0]
            _, r = get_dist(c_new[atoms[4]], c_new[atoms[5]])
            v_dihed = (np.cos(phi)+1)**4
            v_bond = r - equ[0]
            e_minus = fconst * v_dihed * v_bond

            force[a, j] += (e_minus-e_plus) / (2*move)

    return energy


def calc_cross_dihed_angle(coords, atoms, equ, fconst, force):
    phi0 = np.radians(equ[2])
    phi = get_dihed(coords[atoms[:4]])[0]
    theta = get_angle(coords[atoms[4:]])[0]
    v_angle = theta - equ[0]
    v_dihed = 1 + np.cos(equ[1]*phi-phi0)
    energy = fconst * v_dihed * v_angle

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi = get_dihed(c_new[atoms[:4]])[0]
            theta = get_angle(c_new[atoms[4:]])[0]
            v_angle = theta - equ[0]
            v_dihed = 1 + np.cos(equ[1]*phi-phi0)
            e_plus = fconst * v_dihed * v_angle

            c_new[a, j] -= 2*move
            phi = get_dihed(c_new[atoms[:4]])[0]
            theta = get_angle(c_new[atoms[4:]])[0]
            v_angle = theta - equ[0]
            v_dihed = 1 + np.cos(equ[1]*phi-phi0)
            e_minus = fconst * v_dihed * v_angle

            force[a, j] += (e_minus-e_plus)/(2*move)

    return energy


def calc_cross_cos_cube_dihed_angle(coords, atoms, equ, fconst, force):
    phi = get_dihed(coords[atoms[:4]])[0]
    theta = get_angle(coords[atoms[4:]])[0]
    v_angle = theta - equ[0]
    v_dihed = (np.cos(phi)+1)**4
    energy = fconst * v_dihed * v_angle

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi = get_dihed(c_new[atoms[:4]])[0]
            theta = get_angle(c_new[atoms[4:]])[0]
            v_angle = theta - equ[0]
            v_dihed = (np.cos(phi)+1)**4
            e_plus = fconst * v_dihed * v_angle

            c_new[a, j] -= 2*move
            phi = get_dihed(c_new[atoms[:4]])[0]
            theta = get_angle(c_new[atoms[4:]])[0]
            v_angle = theta - equ[0]
            v_dihed = (np.cos(phi)+1)**4
            e_minus = fconst * v_dihed * v_angle

            force[a, j] += (e_minus-e_plus)/(2*move)

    return energy

def calc_cross_dihed_angle_angle(coords, atoms, equ, fconst, force):
    phi = get_dihed(coords[atoms])[0]
    theta1 = get_angle(coords[atoms[:3]])[0]
    theta2 = get_angle(coords[atoms[1:]])[0]

    v_angle1 = theta1 - equ[0]
    v_angle2 = theta2 - equ[1]
    v_dihed = 1 + np.cos(equ[2]*phi-equ[3])
    energy = fconst * v_dihed * v_angle1 * v_angle2

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi = get_dihed(c_new[atoms])[0]
            theta1 = get_angle(c_new[atoms[:3]])[0]
            theta2 = get_angle(c_new[atoms[1:]])[0]
            v_angle1 = theta1 - equ[0]
            v_angle2 = theta2 - equ[1]
            v_dihed = 1 + np.cos(equ[2]*phi-equ[3])
            e_plus = fconst * v_dihed * v_angle1 * v_angle2

            c_new[a, j] -= 2*move
            phi = get_dihed(c_new[atoms])[0]
            theta1 = get_angle(c_new[atoms[:3]])[0]
            theta2 = get_angle(c_new[atoms[1:]])[0]
            v_angle1 = theta1 - equ[0]
            v_angle2 = theta2 - equ[1]
            v_dihed = 1 + np.cos(equ[2]*phi-equ[3])
            e_minus = fconst * v_dihed * v_angle1 * v_angle2

            force[a, j] += (e_minus-e_plus)/(2*move)

    return energy


def calc_cross_cos_cube_dihed_angle_angle(coords, atoms, equ, fconst, force):
    phi = get_dihed(coords[atoms])[0]
    theta1 = get_angle(coords[atoms[:3]])[0]
    theta2 = get_angle(coords[atoms[1:]])[0]

    v_angle1 = theta1 - equ[0]
    v_angle2 = theta2 - equ[1]
    v_dihed = (np.cos(phi)+1)**4
    energy = fconst * v_dihed * v_angle1 * v_angle2

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi = get_dihed(c_new[atoms])[0]
            theta1 = get_angle(c_new[atoms[:3]])[0]
            theta2 = get_angle(c_new[atoms[1:]])[0]
            v_angle1 = theta1 - equ[0]
            v_angle2 = theta2 - equ[1]
            v_dihed = (np.cos(phi)+1)**4
            e_plus = fconst * v_dihed * v_angle1 * v_angle2

            c_new[a, j] -= 2*move
            phi = get_dihed(c_new[atoms])[0]
            theta1 = get_angle(c_new[atoms[:3]])[0]
            theta2 = get_angle(c_new[atoms[1:]])[0]
            v_angle1 = theta1 - equ[0]
            v_angle2 = theta2 - equ[1]
            v_dihed = (np.cos(phi)+1)**4
            e_minus = fconst * v_dihed * v_angle1 * v_angle2

            force[a, j] += (e_minus-e_plus)/(2*move)

    return energy


def calc_oop_angle(coords, atoms, phi0, fconst, force):
    phi = get_oop_angle(coords[atoms])
    dphi = phi - phi0
    energy = 0.5 * fconst * dphi**2

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi = get_oop_angle(c_new[atoms])
            dphi = phi - phi0
            e_plus = 0.5 * fconst * dphi**2

            c_new[a, j] -= 2*move
            phi = get_oop_angle(c_new[atoms])
            dphi = phi - phi0
            e_minus = 0.5 * fconst * dphi**2

            force[a, j] += (e_minus-e_plus) / (2*move)
    return energy


@ jit(nopython=True)
def calc_harmonic_diheds(coords, atoms, phi0, fconst, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    dphi = phi - phi0
    dphi = np.pi - (dphi + np.pi) % (2 * np.pi)  # dphi between -pi to pi
    energy = 0.5 * fconst * dphi**2
    ddphi = - fconst * dphi

    calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@ jit(nopython=True)
def calc_periodic_dihed(coords, atoms, equ, fconst, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    mdphi = equ[0] * phi - equ[1]
    ddphi = - fconst * equ[0] * np.sin(mdphi)
    energy = fconst * (1 + np.cos(mdphi))
    calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@ jit(nopython=True)
def calc_rb_diheds(coords, atoms, params, force):
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
    calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@ jit(nopython=True)
def lsq_rb_diheds(coords, atoms, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])
    phi += np.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    energy = np.array([0.0 for _ in range(6)])
    energy[0] = 1.0

    cos_factor = 1.0

    for i in range(1, 6):
        cos_factor *= cos_phi
        energy[i] += cos_factor

    # Term
    # \sum_n C_n * cos(phi)^n
    #
    # d/dphi \sum_n C_n * cos(phi)^n = \sum_n C_n * (-n*sin(phi)) * cos(phi)^(n-1)
    #
    # d/dri \sum_n C_n * cos(phi)^n =  \sum_n C_n * (-n*sin(phi)) * cos(phi)^(n-1) * d/dri phi
    #

    tmp_force = np.zeros((4, 3), dtype=coords.dtype)
    calc_dih_force(tmp_force, np.array([0, 1, 2, 3], dtype=atoms.dtype), vec_ij, vec_kj, vec_kl, cross1, cross2, 1.0)
    # ddphi = d/dphi
    ddphi = 0
    for n in range(1, 6):
        ddphi = -n * sin_phi * cos_phi**n
        for i, ia in enumerate(atoms):
            force[n][ia] += tmp_force[i]*ddphi
    # Return the energy as an array
    return energy


@ jit(nopython=True)
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
    calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    return energy


@ jit(nopython=True)
def convert_to_inversion_rb(fconst, phi0):
    cos_phi0 = np.cos(phi0)
    c0 = fconst * cos_phi0**2
    c1 = 2 * fconst * cos_phi0
    c2 = fconst
    return c0, c1, c2


def calc_cos_cube_diheds(coords, atoms, fconst, force):
    phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(coords[atoms])

    # cos_phi1 = np.cos(phi)+1
    # cos_phi2 = cos_phi1 * cos_phi1
    # cos_phi3 = cos_phi2 * cos_phi1
    #
    # # energy = fconst * cos_phi3
    # # ddphi = - 3 * fconst * np.sin(phi) * cos_phi2
    #
    # cos_phi4 = cos_phi3 * cos_phi1
    # cos_phi5 = cos_phi4 * cos_phi1
    # cos_phi6 = cos_phi5 * cos_phi1
    # energy = fconst * cos_phi6
    # ddphi = - 6 * fconst * np.sin(phi) * cos_phi5

    energy = fconst * (np.cos(phi)+1)**4

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(c_new[atoms])
            e_plus = fconst * (np.cos(phi)+1)**4

            c_new[a, j] -= 2*move
            phi, vec_ij, vec_kj, vec_kl, cross1, cross2 = get_dihed(c_new[atoms])
            e_minus = fconst * (np.cos(phi)+1)**4

            force[a, j] += (e_minus-e_plus) / (2*move)

    # f2 = np.zeros(force.shape)
    # calc_dih_force(force, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    # calc_dih_force(f2, atoms, vec_ij, vec_kj, vec_kl, cross1, cross2, ddphi)
    # print('NUM\n', f1)
    # print('ANAL\n', f2)
    # print('----')

    return energy


def calc_pitorsion_diheds(coords, atoms, phi0, fconst, force):
    phi, vec12, vec13, vec45, vec46, cross1, cross2 = get_pitorsion(coords[atoms])
    dphi = phi - phi0
    dphi = np.pi - (dphi + np.pi) % (2 * np.pi)  # dphi between -pi to pi
    energy = fconst * np.sin(dphi)**2

    unique_atoms = np.unique(atoms)
    move = 1e-8
    for a in unique_atoms:
        for j in range(3):
            c_new = np.copy(coords)

            c_new[a, j] += move
            phi, vec12, vec13, vec45, vec46, cross1, cross2 = get_pitorsion(c_new[atoms])
            dphi = phi - phi0
            dphi = np.pi - (dphi + np.pi) % (2 * np.pi)  # dphi between -pi to pi
            e_plus = fconst * np.sin(dphi)**2

            c_new[a, j] -= 2*move
            phi, vec12, vec13, vec45, vec46, cross1, cross2 = get_pitorsion(c_new[atoms])
            dphi = phi - phi0
            dphi = np.pi - (dphi + np.pi) % (2 * np.pi)  # dphi between -pi to pi
            e_minus = fconst * np.sin(dphi)**2

            force[a, j] += (e_minus-e_plus) / (2*move)

    return energy


@ jit("f8(f8[:], f8[:])", nopython=True)
def dot_prod(a, b):
    x = a[0]*b[0]
    y = a[1]*b[1]
    z = a[2]*b[2]
    return x+y+z


@ jit("void(f8[:,:], i8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8)",
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


@ jit(nopython=True)
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


@ jit(nopython=True)
def get_dist(coord1, coord2):
    vec = coord1 - coord2
    r = norm(vec)
    return vec, r


@ jit(nopython=True)
def get_angle(coords):
    vec12, r12 = get_dist(coords[0], coords[1])
    vec32, r32 = get_dist(coords[2], coords[1])
    dot = np.dot(vec12/r12, vec32/r32)
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.acos(dot), vec12, vec32, r12, r32


@ jit(nopython=True)
def get_angle_from_vectors(vec1, vec2):
    dot = np.dot(vec1/norm(vec1), vec2/norm(vec2))
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.acos(dot)


@ jit(nopython=True)
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


@ jit(nopython=True)
def get_pitorsion(coords):
    vec12, _ = get_dist(coords[1], coords[0])
    vec13, _ = get_dist(coords[2], coords[0])
    vec45, _ = get_dist(coords[4], coords[3])
    vec46, _ = get_dist(coords[5], coords[3])

    cross1 = cross_prod(vec12, vec13)
    cross2 = cross_prod(vec45, vec46)
    phi = get_angle_from_vectors(cross1, cross2)

    return phi, vec12, vec13, vec45, vec46, cross1, cross2

# @ jit(nopython=True)
# def get_pitorsion(coords):
#     vec15, _ = get_dist(coords[4], coords[0])
#     vec16, _ = get_dist(coords[5], coords[0])
#     vec42, _ = get_dist(coords[1], coords[3])
#     vec43, _ = get_dist(coords[2], coords[3])
#
#     cross1 = cross_prod(vec15, vec16)
#     cross2 = cross_prod(vec42, vec43)
#
#     phi = get_dihed([coords[0]+cross2, coords[0], coords[3], coords[3]+cross1])[0]
#
#     return phi, vec15, vec16, vec42, vec43, cross1, cross2
#

def get_oop_angle(coords):
    vec24, _ = get_dist(coords[1], coords[3])
    vec34, _ = get_dist(coords[2], coords[3])
    vec14, _ = get_dist(coords[0], coords[3])

    cross = cross_prod(vec24, vec34)
    cross /= norm(cross)
    dot = dot_prod(cross, vec14)
    proj = coords[0] - cross*dot
    theta = get_angle([coords[0], coords[3], proj])[0]
    print(theta)
    return theta


@ jit("f8[:](f8[:], f8[:])", nopython=True)
def cross_prod(a, b):
    c = np.empty(3, dtype=np.double)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c


@ jit("f8(f8[:])", nopython=True)
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
