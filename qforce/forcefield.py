import numpy as np
import os
#
from .elements import ATOM_SYM, ATOMMASS

"""
To do: write FF to other formats using ParmEd
"""


class ForceField():
    def __init__(self, inp, mol, neighbors, residue='MOL', exclude_all=[]):
        self.polar = inp.polar
        self.polar_title = ''
        self.mol_name = inp.job_name
        self.n_atoms = mol.n_atoms
        self.elements = mol.elements
        self.q = self.set_charge(mol.non_bonded)
        self.residue = residue
        self.comb_rule = inp.comb_rule
        self.urey = inp.urey
        self.n_excl = inp.n_excl
        self.atom_names = self.get_atom_names()
        self.masses = [round(ATOMMASS[i], 5) for i in self.elements]
        self.exclusions = self.make_exclusions(mol.non_bonded.exclusions, neighbors, exclude_all)
        if self.polar:
            self.polar_title = '_polar'

    def write_gromacs(self, inp, mol, directory, coords):
        self.write_itp(inp, mol, directory)
        self.write_top(directory)
        self.write_gro(directory, coords)

    def write_top(self, directory):
        with open(f"{directory}/gas{self.polar_title}.top", "w") as top:
            # defaults
            top.write("\n[ defaults ]\n")
            top.write(";nbfunc   comb-rule   gen-pairs   fudgeLJ   fudgeQQ\n")
            top.write(f"      1           {self.comb_rule}          no       1.0       1.0\n\n\n")

            top.write("; Include the molecule ITP\n")
            top.write(f'#include "./{self.mol_name}_qforce{self.polar_title}.itp"\n\n\n')

            size = len(self.mol_name)
            top.write("[ system ]\n")
            top.write(f"; {' '*(size-6)}name\n")
            top.write(f"{' '*(6-size)}{self.mol_name}\n\n\n")

            top.write("[ molecules ]\n")
            top.write(f"; {' '*(size-10)}compound    n_mol\n")
            top.write(f"{' '*(10-size)}{self.mol_name}        1\n")

    def write_gro(self, directory, coords, box=[20., 20., 20.]):
        n_atoms = self.n_atoms
        if self.polar:
            n_atoms *= 2
        coords *= 0.1
        with open(f"{directory}/gas{self.polar_title}.gro", "w") as gro:
            gro.write(f"{self.mol_name}\n")
            gro.write(f"{n_atoms:>6}\n")
            for i, (a_name, coord) in enumerate(zip(self.atom_names, coords), start=1):
                gro.write(f"{1:>5}{self.residue:<5}")
                gro.write(f"{a_name:>5}{i:>5}")
                gro.write(f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}\n")
            if self.polar:
                for i, coord in enumerate(coords, start=1):
                    gro.write(f"{2:>5}{self.residue:<5}")
                    gro.write(f"{'D':>5}{i+self.n_atoms:>5}")
                    gro.write(f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}\n")

            gro.write(f'{box[0]:>12.5f}{box[1]:>12.5f}{box[2]:>12.5f}\n')

    def write_itp(self, inp, mol, directory):
        with open(f"{directory}/{self.mol_name}_qforce{self.polar_title}.itp", "w") as itp:
            self.write_itp_title(itp)
            self.write_itp_parameters(itp, inp)
            self.write_itp_atoms_and_molecule(itp, mol.non_bonded)
            if self.polar:
                self.write_itp_polarization(itp)
            self.write_itp_bonds(itp, mol.terms)
            self.write_itp_angles(itp, mol.terms)
            self.write_itp_dihedrals(itp, mol.terms)
            self.write_itp_exclusions(itp)
            itp.write('\n')

    def write_itp_title(self, itp):
        itp.write(""";
;           ____         ______
;          / __ \       |  ____|
;         | |  | |______| |__ ___  _ __ ___ ___
;         | |  | |______|  __/ _ \| '__/ __/ _ \\
;         | |__| |      | | | (_) | | | (_|  __/
;          \___\_\      |_|  \___/|_|  \___\___|
;
;          Selim Sami, Maximilian F.S.J. Menger
;             University of Groningen - 2020
;          ====================================
;\n""")

    def write_itp_parameters(self, itp, inp):
        itp.write(f'; lj: {inp.lennard_jones}, charges: {inp.point_charges}\n')
        itp.write(f'; NB fitting: {inp.non_bonded}, fragment_fitting: {inp.fragment}\n')
        itp.write(f'; urey: {inp.urey}, cross_bond_angle: {inp.cross_bond_angle}\n;\n')

        # fitting parameters - temporary
        if inp.param != []:
            itp.write('; fitting parameters are (C, H, O, N):\n')
            itp.write(f'; S8: ')
            for s8 in inp.param[::2]:
                itp.write(f'{s8} ')
            itp.write(f'\n; R_ref: ')
            for r in inp.param[1::2]:
                itp.write(f'{r} ')
            itp.write('\n;\n')
        itp.write('\n')

    def write_itp_atoms_and_molecule(self, itp, non_bonded):
        # atom types
        itp.write("[ atomtypes ]\n")
        if self.comb_rule == 1:
            itp.write(";   name     mass   charge  t           c6          c12\n")
        else:
            itp.write(";   name     mass   charge  t        sigma      epsilon\n")

        for lj_type, lj_params in non_bonded.lj_type_dict.items():
            itp.write(f'{lj_type:>8} {0:>8.4f} {0:>8.4f} {"A":>2} ')
            itp.write(f'{lj_params[0]:>12.5e} {lj_params[1]:>12.5e}\n')

        if self.polar:
            itp.write(f'{"DP":>8} {0:>8.4f} {0:>8.4f} {"D":>2} {0:>12.5e} {0:>12.5e}\n')

        # molecule type
        space = " "*(len(self.mol_name)-5)
        itp.write("\n[ moleculetype ]\n")
        itp.write(f";{space}name nrexcl\n")
        itp.write(f"{self.mol_name}{self.n_excl:>7}\n")

        # atoms
        itp.write("\n[ atoms ]\n")
        itp.write(";  nr     type resnr resnm   atom cgrp     charge      mass\n")
        for i, (lj_type, a_name, q, mass) in enumerate(zip(non_bonded.lj_types, self.atom_names,
                                                           self.q, self.masses), start=1):
            itp.write(f'{i:>5}{lj_type:>9}{1:>6}{self.residue:>6}{a_name:>7}{i:>5}{q:>11.5f}')
            itp.write(f'{mass:>10.5f}\n')

        if self.polar:
            for i in range(self.n_atoms+1, 2*self.n_atoms+1):
                itp.write(f'{i:>5}{"DP":>9}{2:>6}{self.residue:>6}{"D":>7}{i-self.n_atoms:>5}')
                itp.write(f'{-8.:>11.5f}{0.:>10.5f}\n')

    def write_itp_polarization(self, itp):
        polar_dict = {1: 0.00045330, 6: 0.00130300, 7: 0.00098850, 8: 0.00083690, 16: 2.47400}
        # polar_dict = { 1: 0.000413835,  6: 0.00145,  7: 0.000971573,
        #               8: 0.000851973,  9: 0.000444747, 16: 0.002474448,
        #               17: 0.002400281, 35: 0.003492921, 53: 0.005481056}
        # polar_dict = { 1: 0.000413835,  6: 0.001288599,  7: 0.000971573,
        #               8: 0.000851973,  9: 0.000444747, 16: 0.002474448,
        #               17: 0.002400281, 35: 0.003492921, 53: 0.005481056}
        # polar_dict = { 1: 0.000205221,  6: 0.000974759,  7: 0.000442405,
        #               8: 0.000343551,  9: 0.000220884, 16: 0.001610042,
        #               17: 0.000994749, 35: 0.001828362, 53: 0.002964895}

        # polarization
        itp.write("\n[ polarization ]\n")
        itp.write(";    i     j     f         alpha\n")
        for i, elem in enumerate(self.elements):
            itp.write(f"{i+1:>6}{i+self.n_atoms+1:>6}{1:>6}{polar_dict[elem]:>14.8f}\n")

        # # thole polarization
        # if self.thole != []:
        #     itp.write("\n[ thole_polarization ]\n")
        #     itp.write(";   ai    di    aj    dj   f      a      alpha(i)      "
        #               "alpha(j)\n")
        # for tho in self.thole:
        #     itp.write("{:>6}{:>6}{:>6}{:>6}{:>4}{:>7.2f}{:>14.8f}{:>14.8f}\n".format(*tho))

    def write_itp_bonds(self, itp, terms):
        itp.write("\n[ bonds ]\n")
        itp.write(";   ai    aj     f        r0        kb\n")
        for bond in terms['bond']:
            ids = bond.atomids + 1
            equ = bond.equ * 0.1
            fconst = bond.fconst * 100
            itp.write(f'{ids[0]:>6}{ids[1]:>6}{1:>6}{equ:>10.5f}{fconst:>10.0f}\n')

    def write_itp_angles(self, itp, terms):
        itp.write("\n[ angles ]\n")
        itp.write(";   ai    aj    ak     f        th0          kth\n")
        for angle in terms['angle']:
            ids = angle.atomids + 1
            equ = np.degrees(angle.equ)
            fconst = angle.fconst

            if self.urey:
                urey = [term for term in terms['urey'] if np.array_equal(term.atomids,
                                                                         angle.atomids)]
            if not self.urey or len(urey) == 0:
                itp.write(f'{ids[0]:>6}{ids[1]:>6}{ids[2]:>6}{1:>6}{equ:>11.3f}{fconst:>13.3f}\n')
            else:
                urey_equ = urey[0].equ * 0.1
                urey_fconst = urey[0].fconst * 100
                itp.write(f'{ids[0]:>6}{ids[1]:>6}{ids[2]:>6}{5:>6}{equ:>11.3f}{fconst:>13.3f}'
                          f'{urey_equ:>10.5f}{urey_fconst:>13.3f}\n')

    def write_itp_dihedrals(self, itp, terms):
        if len(terms['dihedral']) > 0:
            itp.write("\n[ dihedrals ]\n")

        # rigid dihedrals
        if len(terms['dihedral/rigid']) > 0:
            itp.write("; rigid dihedrals \n")
            itp.write(";   ai    aj    ak    al     f        th0          kth\n")

        for dihed in terms['dihedral/rigid']:
            ids = dihed.atomids + 1
            equ = np.degrees(dihed.equ)
            fconst = dihed.fconst

            itp.write(f'{ids[0]:>6}{ids[1]:>6}{ids[2]:>6}{ids[3]:>6}{2:>6}{equ:>11.3f}')
            itp.write(f'{fconst:>13.3f}\n')

        # improper dihedrals
        if len(terms['dihedral/improper']) > 0:
            itp.write("; improper dihedrals \n")
            itp.write(";   ai    aj    ak    al     f        th0          kth\n")

        for dihed in terms['dihedral/improper']:
            ids = dihed.atomids + 1
            equ = np.degrees(dihed.equ)
            fconst = dihed.fconst

            itp.write(f'{ids[0]:>6}{ids[1]:>6}{ids[2]:>6}{ids[3]:>6}{2:>6}{equ:>11.3f}')
            itp.write(f'{fconst:>13.3f}\n')

        # flexible dihedrals
        if len(terms['dihedral/flexible']) > 0:
            itp.write("; flexible dihedrals \n")
            itp.write(';   ai    aj    ak    al     f         c0         c1         c2         c3')
            itp.write('         c4         c5\n')

        for dihed in terms['dihedral/flexible']:
            ids = dihed.atomids + 1
            c = dihed.equ

            itp.write(f'{ids[0]:>6}{ids[1]:>6}{ids[2]:>6}{ids[3]:>6}{3:>6}{c[0]:>11.3f}')
            itp.write(f'{c[1]:>11.3f}{c[2]:>11.3f}{c[3]:>11.3f}{c[4]:>11.3f}{c[5]:>11.3f}\n')

        # constrained dihedrals
        if len(terms['dihedral/constr']) > 0:
            itp.write("; constrained dihedrals \n")
            itp.write(";   ai    aj    ak    al     f        th0          kth\n")

        for dihed in terms['dihedral/constr']:
            ids = dihed.atomids + 1
            itp.write(f';{ids[0]:>6}{ids[1]:>6}{ids[2]:>6}{ids[3]:>6} - Not implemented yet\n')

    def write_itp_exclusions(self, itp):
        if any(len(exclusion) > 0 for exclusion in self.exclusions):
            itp.write("\n[ exclusions ]\n")
        for i, exclusion in enumerate(self.exclusions):
            if len(exclusion) > 0:
                exclusion = sorted(set(exclusion))
                itp.write("{} ".format(i+1))
                itp.write(("{} "*len(exclusion)).format(*exclusion))
                itp.write("\n")

    def make_exclusions(self, input_exclusions, neighbors, exclude_all):
        exclusions = [[] for _ in range(self.n_atoms)]
        polar_exclusions = [[] for _ in range(self.n_atoms)]

        for exclusion in input_exclusions:
            exclusions[exclusion[0]].append(exclusion[1]+1)
            if self.polar:
                polar_exclusions[exclusion[0]].append(exclusion[1]+1)
                polar_exclusions[exclusion[1]].append(exclusion[0]+1)
                polar_exclusions[exclusion[0]].append(self.n_atoms+exclusion[1]+1)

        for i in exclude_all:
            exclusions[i].extend(np.arange(1, self.n_atoms+1))
            if self.polar:
                exclusions[i].extend(np.arange(self.n_atoms+1, 2*self.n_atoms+1))
                polar_exclusions[i].extend(np.arange(1, 2*self.n_atoms+1))

        if self.polar:
            for i in range(self.n_atoms):
                for neigh in neighbors[self.n_excl-2][i]+neighbors[self.n_excl-1][i]:
                    polar_exclusions[i].append(self.n_atoms+neigh+1)
                for neigh in neighbors[self.n_excl-1][i]:
                    polar_exclusions[i].append(neigh+1)

        if self.polar:
            exclusions = exclusions + polar_exclusions

        return exclusions

    def get_atom_names(self):
        atom_names = []
        atom_dict = {}

        for i, elem in enumerate(self.elements):
            sym = ATOM_SYM[elem]
            if sym not in atom_dict.keys():
                atom_dict[sym] = 1
            else:
                atom_dict[sym] += 1
            atom_names.append(f'{sym}{atom_dict[sym]}')
        return atom_names

    def add_restraints(self, restraints, directory, fc=1000):
        with open(f"{directory}/{self.mol_name}_qforce{self.polar_title}.itp", "a") as itp:
            itp.write("[ dihedral_restraints ]\n")
            itp.write(";  ai   aj   ak   al type       phi   dp   kfac\n")
            for restraint in restraints:
                a1, a2, a3, a4 = restraint[0]+1
                phi = np.degrees(restraint[1])
                itp.write(f'{a1:>5}{a2:>5}{a3:>5}{a4:>5}{1:>5} {phi:>10.4f}  0.0  {fc}\n')

    def set_charge(self, non_bonded):
        q = non_bonded.q
        if self.polar:
            q = non_bonded.q+8
        return q

    # bohr2nm = 0.052917721067
    # if polar:
    #     alphas = qm.alpha*bohr2nm**3
    #     drude = {}
    #     n_drude = 1
    #     ff.atom_types.append(["DP", 0, 0, "S", 0, 0])

    #     for i, alpha in enumerate(alphas):
    #         if alpha > 0:
    #             drude[i] = mol.topo.n_atoms+n_drude
    #             ff.atoms[i][6] += 8
    #             # drude atoms
    #             ff.atoms.append([drude[i], 'DP', 2, 'MOL', f'D{atoms[i]}',
    #                              i+1, -8., 0.])
    #             ff.coords.append(ff.coords[i])
    #             # polarizability
    #             ff.polar.append([i+1, drude[i], 1, alpha])
    #             n_drude += 1
    #     ff.natom = len(ff.atoms)
    #     for i, alpha in enumerate(alphas):
    #         if alpha > 0:
    #             # exclusions for balancing the drude particles
    #             for j in mol.topo.neighbors[inp.nrexcl-2][i]+mol.topo.neighbors[inp.nrexcl-1][i]:
    #                 if alphas[j] > 0:
    #                     ff.exclu[drude[i]-1].extend([drude[j]])
    #             for j in mol.topo.neighbors[inp.nrexcl-1][i]:
    #                 ff.exclu[drude[i]-1].extend([j+1])
    #             ff.exclu[drude[i]-1].sort()
    #             # thole polarizability
    #             for neigh in [mol.topo.neighbors[n][i] for n in range(inp.nrexcl)]:
    #                 for j in neigh:
    #                     if i < j and alphas[j] > 0:
    #                         ff.thole.append([i+1, drude[i], j+1, drude[j], "2", 2.6, alpha,
    #                                          alphas[j]])
