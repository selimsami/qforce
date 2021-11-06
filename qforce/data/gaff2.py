'''This file reads the gaff2.dat file from Amber installation and converts
it to the gromacs file format.

Note that this conversion needs parmed which is not listed as a dependency.
'''

import parmed as pmd

# Note this address is the local address and is only used for the generation
# of gaff2.dat
gaff2_dat = '~/miniconda3/envs/amber/dat/leap/parm/gaff2.dat'

top = pmd.load_file(gaff2_dat)
with open('gaff2.dat', 'w') as f:
    f.write('''[ atomtypes ]
; name    mass    charge ptype  sigma      epsilon
''')
    for name in top.atom_types:
        mass = top.atom_types[name].mass
        try:
            sigma = top.atom_types[name].sigma / 10 # A to nm
        except:
            f.write('; Sigma for {} not found, set to 0.\n'.format(name))
            sigma = 0

        try:
            epsilon = top.atom_types[name].epsilon * 4.184 # kcal to kJ
        except:
            f.write('; epsilon for {} not found, set to 0.\n'.format(name))
            epsilon = 0
        f.write('{name: <10} {mass: >10}    0.000000  A     {sigma:.6f} {'
                'epsilon:.6f}\n'.format(name=name, mass=mass, sigma=sigma,
                                        epsilon=epsilon))
