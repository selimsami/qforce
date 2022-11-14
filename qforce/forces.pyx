# cython: language_level=3
cimport cython
from cython.view cimport array as cvarray

from libc.math cimport sqrt, acos, cos, sin
from libc.stdio cimport printf
from libc.math cimport pi as PI
#
import numpy as np
cimport numpy as np


np.import_array()


@cython.boundscheck(False)
cpdef double get_distance(double [:] c1, double [:] c2):
    cdef double [3] r
    return c_get_distance(r, c1, c2)


@cython.boundscheck(False)
cpdef double py_get_dist(double [:] v, double [:] c1, double [:] c2):
    return c_get_distance(v, c1, c2)


@cython.boundscheck(False)
cpdef double py_get_dihed(double [:, :] coords):
    cdef double[:] c1, c2, c3, c4
    cdef double[:] cr1, cr2, v12, v32, v34
    cdef double[3] cross1, cross2, vec12, vec32, vec34

    v12 = vec12
    v32 = vec32
    v34 = vec34

    cr1 = cross1
    cr2 = cross2

    c1 = coords[0]
    c2 = coords[1]
    c3 = coords[2]
    c4 = coords[3]

    return get_dihed(cr1, cr2, v12, v32, v34, c1, c2, c3, c4)


@cython.boundscheck(False)
cpdef double calc_bonds(double [:,:] coords, long [:] atoms, double r0, double fconst, double [:, :] force):
    cdef double [3] vec12, f
    cdef double [:] v1, v2, v
    cdef double r12, energy
    cdef long ia, ib

    # get atomids
    ia = atoms[0]
    ib = atoms[1]

    # get coordinates
    v1 = coords[ia]
    v2 = coords[ib]
    v = vec12

    # 
    r12 = c_get_distance(v, v1, v2) 

    energy = 0.5 * fconst * (r12 - r0)**2
    
    #
    f[0] = - fconst * vec12[0] * (r12-r0) / r12
    f[1] = - fconst * vec12[1] * (r12-r0) / r12
    f[2] = - fconst * vec12[2] * (r12-r0) / r12
    
    #
    force[ia, 0] += f[0]
    force[ia, 1] += f[1]
    force[ia, 2] += f[2]
    #
    force[ib, 0] -= f[0]
    force[ib, 1] -= f[1]
    force[ib, 2] -= f[2]

    return energy


@cython.boundscheck(False)
cpdef double calc_angles(double [:, :] coords,
                         long [:] atoms,
                         double theta0,
                         double fconst,
                         double [:, :] force):

    cdef double[:] c1, c2, c3
    cdef double[:] v12, v32, res
    cdef double[3] vec12, vec32
    cdef double[2] vec_res
    cdef long a1, a2, a3

    cdef double theta, cos_theta, cos_theta_sq, dtheta
    cdef double energy, st, sth
    #
    cdef double c13, c11, c33
    cdef double r12, r32
    cdef double[3] f1, f3
    #
    a1 = atoms[0]
    a2 = atoms[1]
    a3 = atoms[2]
    #
    v12 = vec12
    v32 = vec32
    res = vec_res
    #
    c1 = coords[a1]
    c2 = coords[a2]
    c3 = coords[a3]
    # compute angle and get vectors
    theta = c_get_angle(res, v12, v32, c1, c2, c3)
    cos_theta = cos(theta)
    cos_theta_sq = cos_theta*cos_theta
    dtheta = theta - theta0
    #
    energy = 0.5 * fconst * dtheta**2
    # why???
    if cos_theta_sq < 1:
        r12 = res[0]
        r32 = res[1]
        #
        st = -fconst * dtheta / sqrt(1. - cos_theta_sq)
        sth = st * cos_theta
        #
        c13 = (st / r12) / r32
        c11 = (sth / r12) / r12
        c33 = (sth / r32) / r32
        #
        f1[0] = c11 * vec12[0] - c13 * vec32[0]
        f1[1] = c11 * vec12[1] - c13 * vec32[1]
        f1[2] = c11 * vec12[2] - c13 * vec32[2]
        #
        f3[0] = c33 * vec32[0] - c13 * vec12[0]
        f3[1] = c33 * vec32[1] - c13 * vec12[1]
        f3[2] = c33 * vec32[2] - c13 * vec12[2]

        force[a1, 0] += f1[0]
        force[a1, 1] += f1[1]
        force[a1, 2] += f1[2]
        #
        force[a3, 0] += f3[0]
        force[a3, 1] += f3[1]
        force[a3, 2] += f3[2]
        #
        force[a2, 0] -= f1[0] + f3[0]
        force[a2, 1] -= f1[1] + f3[1]
        force[a2, 2] -= f1[2] + f3[2]
    return energy        


@cython.boundscheck(False)
cpdef double calc_cross_bond_angle(double [:, :] coords, 
                                   long [:] atoms, 
                                   double [:] r0s, 
                                   double fconst, 
                                   double [:, :] force):

    cdef double[:] c1, c2, c3
    cdef double[:] v12, v32, v13
    cdef double[3] vec12, vec32, vec13
    cdef long a1, a2, a3
    #
    cdef double energy
    #
    cdef double r12, r32, r13
    #
    cdef double s1, s2, s3
    cdef double k1, k2, k3
    #
    cdef double[3] f1, f3
    #
    a1 = atoms[0]
    a2 = atoms[1]
    a3 = atoms[2]
    #
    v12 = vec12
    v32 = vec32
    v13 = vec13
    #
    c1 = coords[a1]
    c2 = coords[a2]
    c3 = coords[a3]
    #
    r12 = c_get_distance(v12, c1, c2)
    r32 = c_get_distance(v32, c3, c2)
    r13 = c_get_distance(v13, c1, c3)
    #
    s1 = r12 - r0s[0]
    s2 = r32 - r0s[1]
    s3 = r13 - r0s[2]

    energy = fconst * s3 * (s1 + s2)

    k1 = -fconst * s3/r12
    k2 = -fconst * s3/r32
    k3 = -fconst * (s1+s2)/r13
    #
    f1[0] = k1*vec12[0] + k3*vec13[0]
    f1[1] = k1*vec12[1] + k3*vec13[1]
    f1[2] = k1*vec12[2] + k3*vec13[2]
    #
    f3[0] = k2*vec32[0] + k3*vec13[0]
    f3[1] = k2*vec32[1] + k3*vec13[1]
    f3[2] = k2*vec32[2] + k3*vec13[2]

    force[a1, 0] += f1[0]
    force[a1, 1] += f1[1]
    force[a1, 2] += f1[2]

    force[a3, 0] += f3[0]
    force[a3, 1] += f3[1]
    force[a3, 2] += f3[2]

    force[a2, 0] -= (f1[0] + f3[0])
    force[a2, 1] -= (f1[1] + f3[1])
    force[a2, 2] -= (f1[2] + f3[2])

    return energy


@cython.boundscheck(False)
cpdef double calc_imp_diheds(double [:,:] coords, 
                             long [:] atoms, 
                             double phi0, 
                             double fconst, 
                             double [:, :] force):

    cdef double energy
    cdef long a1, a2, a3, a4
    cdef double[:] c1, c2, c3, c4

    cdef double[3] vec12, vec32, vec34, cross1, cross2
    cdef double[:] v12, v32, v34, cr1, cr2

    cdef double phi, dphi, ddphi

    cr1 = cross1
    cr2 = cross2

    v12 = vec12
    v32 = vec32
    v34 = vec34

    a1 = atoms[0]
    a2 = atoms[1]
    a3 = atoms[2]
    a4 = atoms[3]

    c1 = coords[a1]
    c2 = coords[a2]
    c3 = coords[a3]
    c4 = coords[a4]
    #
    phi = get_dihed(cr1, cr2, v12, v32, v34, c1, c2, c3, c4)

    dphi = phi - phi0
    dphi = PI - (dphi + PI) % (2 * PI)

    energy = 0.5 * fconst * dphi**2

    ddphi = -fconst * dphi

    print(force)

    calc_dih_force(force, atoms, v12, v32, v34, cr1, cr2, ddphi)

    print(force)

    return energy

@cython.boundscheck(False)
cpdef double calc_rb_diheds(double [:, :] coords,
                            long [:] atoms,
                            double [:] params,
                            double fconst,
                            double [:, :] force):
    cdef double energy
    cdef long a1, a2, a3, a4
    cdef double[:] c1, c2, c3, c4

    cdef double[3] vec12, vec32, vec34, cross1, cross2
    cdef double[:] v12, v32, v34, cr1, cr2

    cdef double phi, cos_phi, sin_phi, cos_factor, ddphi

    cdef long i

    cr1 = cross1
    cr2 = cross2

    v12 = vec12
    v32 = vec32
    v34 = vec34

    a1 = atoms[0]
    a2 = atoms[1]
    a3 = atoms[2]
    a4 = atoms[3]

    c1 = coords[a1]
    c2 = coords[a2]
    c3 = coords[a3]
    c4 = coords[a4]
    #
    phi = get_dihed(cr1, cr2, v12, v32, v34, c1, c2, c3, c4)

    phi += PI

    cos_phi = cos(phi)
    sin_phi = sin(phi)

    energy = params[0]
    cos_factor = 1.0
    ddphi = 0.0

    for i in range(1, 6):
        ddphi += i * cos_factor * params[i]
        cos_factor *= cos_phi
        energy += cos_factor * params[i]

    ddphi *= - sin_phi

    calc_dih_force(force, atoms, v12, v32, v34, cr1, cr2, ddphi)
    return energy


@cython.boundscheck(False)
cpdef double calc_inversion(double [:, :] coords,
                            long [:] atoms,
                            double phi0,
                            double fconst,
                            double [:, :] force):
    cdef double energy
    cdef long a1, a2, a3, a4
    cdef double[:] c1, c2, c3, c4

    cdef double[3] vec12, vec32, vec34, cross1, cross2, vecc
    cdef double[:] v12, v32, v34, cr1, cr2, vc

    cdef double phi, cos_phi, sin_phi, ddphi

    vc = vecc
    cr1 = cross1
    cr2 = cross2

    v12 = vec12
    v32 = vec32
    v34 = vec34

    a1 = atoms[0]
    a2 = atoms[1]
    a3 = atoms[2]
    a4 = atoms[3]

    c1 = coords[a1]
    c2 = coords[a2]
    c3 = coords[a3]
    c4 = coords[a4]
    #
    phi = get_dihed(cr1, cr2, v12, v32, v34, c1, c2, c3, c4)
    phi += PI

    cos_phi = cos(phi)
    sin_phi = sin(phi)

    convert_to_inversion_rb(vc, fconst, phi0)

    energy = vc[0]

    ddphi = vc[1]

    energy += cos_phi * vc[1]

    ddphi += 2 * vc[2] * cos_phi
    energy += cos_phi**2 * vc[2]

    ddphi *= -sin_phi

    calc_dih_force(force, atoms, v12, v32, v34, cr1, cr2, ddphi)
    return energy


@cython.boundscheck(False)
cpdef double calc_periodic_dihed(double [:, :] coords,
                                 long [:] atoms,
                                 double phi0,
                                 double fconst,
                                 double [:, :] force):
    cdef double energy
    cdef long a1, a2, a3, a4
    cdef double[:] c1, c2, c3, c4

    cdef double[3] vec12, vec32, vec34, cross1, cross2
    cdef double[:] v12, v32, v34, cr1, cr2

    cdef double phi, mult, ddphi

    cr1 = cross1
    cr2 = cross2

    v12 = vec12
    v32 = vec32
    v34 = vec34

    a1 = atoms[0]
    a2 = atoms[1]
    a3 = atoms[2]
    a4 = atoms[3]

    c1 = coords[a1]
    c2 = coords[a2]
    c3 = coords[a3]
    c4 = coords[a4]
    #
    phi = get_dihed(cr1, cr2, v12, v32, v34, c1, c2, c3, c4)

    mult = 3.0
    phi0 = 0.0 # why?

    mdphi = mult * phi - phi0
    ddphi = fconst * mult * sin(mdphi)

    energy = fconst * (1.0 + cos(mdphi))

    calc_dih_force(force, atoms, v12, v32, v34, cr1, cr2, ddphi)
    return energy


@cython.boundscheck(False)
cpdef double calc_pairs(double [:,:] coords,
                        long [:] atoms,
                        double [:] params,
                        double [:,:] force):
    cdef double c6, c12, qq
    cdef double r
    cdef double[3] vec
    cdef double[:] v, c1, c2
    cdef long a1, a2

    cdef double qq_r, r_2, r_6, c6_r6
    cdef double energy

    cdef double f

    v = vec
    a1 = atoms[0]
    a2 = atoms[1]
    c1 = coords[a1]
    c2 = coords[a2]


    c6 = params[0]
    c12 = params[1]
    qq = params[2]

    r = c_get_distance(v, c1, c2)

    qq_r = qq/r
    r_2 = 1./(r*r)
    r_6 = r_2**3

    c6_r6 = c6*r_6
    c12_r12 = c12 * r_6**2

    energy = qq_r + c12_r12 - c6_r6

    f = (qq_r + 12*c12_r12 - 6*c6_r6) * r_2

    force[a1, 0] += vec[0] * f
    force[a1, 1] += vec[1] * f
    force[a1, 2] += vec[2] * f

    force[a2, 0] -= vec[0] * f
    force[a2, 1] -= vec[1] * f
    force[a2, 2] -= vec[2] * f

    return energy

@cython.boundscheck(False)
cpdef void py_convert_to_inversion_rb(double [:] c, double fconst, double phi0) nogil:
    cdef double cos_phi0

    cos_phi0 = cos(phi0)
    c[0] = fconst * cos_phi0**2
    c[1] = 2 * fconst * cos_phi0
    c[2] = fconst

@cython.boundscheck(False)
cdef void convert_to_inversion_rb(double [:] c, double fconst, double phi0) nogil:
    cdef double cos_phi0

    cos_phi0 = cos(phi0)
    c[0] = fconst * cos_phi0**2
    c[1] = 2 * fconst * cos_phi0
    c[2] = fconst

@cython.boundscheck(False)
cdef double c_get_distance(double [:] diffvec, double [:] coord1, double [:] coord2) nogil:
    cdef Py_ssize_t i
    cdef double res

    diffvec[0] = coord1[0] - coord2[0]
    diffvec[1] = coord1[1] - coord2[1]
    diffvec[2] = coord1[2] - coord2[2]

    return _dnorm3d(diffvec)

@cython.boundscheck(False)
cdef double c_get_angle_from_vectors(double [:] vec1, double [:] vec2, double norm) nogil:
    cdef double dot
    dot = _ddot3d(vec1, vec2, norm)

    if dot > 1.0:
        dot = 1.0

    if dot < -1.0:
        dot = -1.0

    return acos(dot)

@cython.boundscheck(False)
cpdef double py_get_angle(double [:, :] coords):
    cdef double[:] c1, c2, c3
    cdef double[3] vec12, vec32
    cdef double[2] vec_res

    cdef double[:] v12, v32, vr

    vr = vec_res
    v12 = vec12
    v32 = vec32


    c1 = coords[0]
    c2 = coords[1]
    c3 = coords[2]

    return c_get_angle(vr, v12, v32, c1, c2, c3)

@cython.boundscheck(False)
cdef double c_get_angle(double[:] res, double [:] vec12, double [:] vec32, 
                         double [:] c1, 
                         double [:] c2, 
                         double [:] c3) nogil:
    cdef Py_ssize_t i
    cdef double dot

    res[0] = c_get_distance(vec12, c1, c2)
    res[1] = c_get_distance(vec32, c3, c2)

    return c_get_angle_from_vectors(vec12, vec32, res[0]*res[1])


@cython.boundscheck(False)
cdef double get_dihed(double [:] cross1,
                      double [:] cross2,
                      double [:] vec12,
                      double [:] vec32,
                      double [:] vec34,
                      double [:] c1,
                      double [:] c2, 
                      double [:] c3, 
                      double [:] c4) nogil:

    r12 = c_get_distance(vec12, c1, c2)
    r32 = c_get_distance(vec32, c3, c2)
    r34 = c_get_distance(vec34, c3, c4)
    _dcross_prod3d(cross1, vec12, vec32)
    _dcross_prod3d(cross2, vec32, vec34)

    phi = c_get_angle_from_vectors(cross1, cross2, 1.0)

    if _ddot3d_pure(vec12, cross2) < 0:
        phi = - phi
    return phi


@cython.boundscheck(False)
cdef double _dnorm(double [:] vector) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t length = vector.shape[0]
    cdef double res = 0.0

    for i in range(length): 
        res += vector[i]**2

    return sqrt(res)


@cython.boundscheck(False)
cdef double _dnorm3d(double [:] vec) nogil:
    return sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


@cython.boundscheck(False)
cdef double _ddot(double [:] vec1,  double [:] vec2, double scale) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t length = vec1.shape[0]
    cdef double res = 0.0

    for i in range(length): 
        res += (vec1[i] * vec2[i])

    return res/scale


@cython.boundscheck(False)
cdef double _ddot3d(double [:] vec1,  double [:] vec2, double scale) nogil:
    return (vec1[0]*vec2[0] +
            vec1[1]*vec2[1] +
            vec1[2]*vec2[2])/scale

@cython.boundscheck(False)
cdef double _ddot3d_pure(double [:] vec1,  double [:] vec2) nogil:
    return (vec1[0]*vec2[0] +
            vec1[1]*vec2[1] +
            vec1[2]*vec2[2])

@cython.boundscheck(False)
cdef void _dcross_prod3d(double [:] res, double [:] a, double [:] b) nogil:
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = a[2]*b[0] - a[0]*b[2]
    res[2] = a[0]*b[1] - a[1]*b[0]


@cython.boundscheck(False)
cdef void calc_dih_force(double [:,:] force, 
                             long [:] a,
                             double [:] vec_ij,
                             double [:] vec_kj,
                             double [:] vec_kl,
                             double [:] cross1,
                             double [:] cross2,
                             double ddphi) nogil:
    cdef double inner1, inner2, nrkj2
    cdef double nrkj, nrkj_1, nrkj_2
    cdef double aa, bb, p, q
    cdef double[3] f_i, f_j, f_l, f_k

    cdef double[3] svec

    cdef long a1, a2, a3, a4

    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    a4 = a[3]

    inner1 = _ddot3d_pure(cross1, cross1)
    inner2 = _ddot3d_pure(cross2, cross2)
    nrkj2 = _ddot3d_pure(vec_kj, vec_kj)

    nrkj_1 = 1. / sqrt(nrkj2)
    nrkj_2 = nrkj_1 * nrkj_1
    nrkj = nrkj2*nrkj_1
    #
    aa = -ddphi * nrkj / inner1

    f_i[0] = aa*cross1[0]
    f_i[1] = aa*cross1[1]
    f_i[2] = aa*cross1[2]

    bb = ddphi * nrkj / inner2

    f_l[0] = bb*cross2[0]
    f_l[1] = bb*cross2[1]
    f_l[2] = bb*cross2[2]

    p = nrkj_2 * _ddot3d_pure(vec_ij, vec_kj)
    q = nrkj_2 * _ddot3d_pure(vec_kl, vec_kj)

    svec[0] = p * f_i[0] - q * f_l[0]
    svec[1] = p * f_i[1] - q * f_l[1]
    svec[2] = p * f_i[2] - q * f_l[2]

    f_j[0] = f_i[0] - svec[0]
    f_j[1] = f_i[1] - svec[1]
    f_j[2] = f_i[2] - svec[2]

    f_k[0] = f_l[0] + svec[0]
    f_k[1] = f_l[1] + svec[1]
    f_k[2] = f_l[2] + svec[2]

    force[a1, 0] += f_i[0]
    force[a1, 1] += f_i[1]
    force[a1, 2] += f_i[2]

    force[a2, 0] -= f_j[0]
    force[a2, 1] -= f_j[1]
    force[a2, 2] -= f_j[2]

    force[a3, 0] -= f_k[0]
    force[a3, 1] -= f_k[1]
    force[a3, 2] -= f_k[2]

    force[a4, 0] += f_l[0]
    force[a4, 1] += f_l[1]
    force[a4, 2] += f_l[2]

