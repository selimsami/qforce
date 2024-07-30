import sympy
from ..symforce import N


def get_dr(xyz1, xyz2):
    dr = 0.0*N.i
    for x1, x2 in zip(xyz1, xyz2):
        dr += (x1-x2)
    return dr


def norm(vec):
    return sympy.sqrt(vec.dot(vec))


def get_r(xyz1, xyz2):
    dr = 0.0*N.i
    for x1, x2 in zip(xyz1, xyz2):
        dr += (x1-x2)
    return sympy.sqrt(dr.dot(dr))


def dotprod(xyz1, xyz2, xyz3, xyz4):
    dr1 = get_dr(xyz1, xyz2)
    dr2 = get_dr(xyz1, xyz2)
    return dr1.dot(dr2)


def get_dist(xyz1, xyz2):
    dr = 0.0*N.i
    for x1, x2 in zip(xyz1, xyz2):
        dr += (x1-x2)
    return dr, sympy.sqrt(dr.dot(dr))


def get_angle(xyz1, xyz2, xyz3, xyz4):
    dri = get_dr(xyz1, xyz2)
    drj = get_dr(xyz3, xyz4)

    nri = sympy.sqrt(dri.dot(dri))
    nrj = sympy.sqrt(drj.dot(drj))

    return sympy.acos(dri.dot(drj)/(nri*nrj))


def get_dihedral(xyz1, xyz2, xyz3, xyz4):
    vec12, r12 = get_dist(xyz1, xyz2)
    vec32, r32 = get_dist(xyz3, xyz2)
    vec34, r34 = get_dist(xyz3, xyz4)

    cross1 = vec12.cross(vec13)
    cross2 = vec32.cross(vec34)

    nc1 = norm(cross1)
    nc2 = norm(cross2)

    return sympy.acos(cross1.dot(cross2)/(nc1*nc2))
