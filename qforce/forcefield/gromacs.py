import numpy as np
#
from colt import Colt
#
from ..molecule.non_bonded import calc_sigma_epsilon
from ..forces import convert_to_inversion_rb
from ..misc import LOGO_SEMICOL
from .forcefield_base import ForcefieldSettings

class Gromacs(ForcefieldSettings):

    def __init__(self, ff):
        self.ff = ff

    def write(self, directory, coords):
        self.write_itp(directory)
        self.write_top(directory)
        self.write_gro(directory, coords)

    def write_top(self, directory):
        with open(f"{directory}/gas.top", "w") as top:
            # defaults
            top.write("\n[ defaults ]\n")
            top.write("; nbfunc    comb-rule    gen-pairs      fudgeLJ      fudgeQQ\n")
            top.write(f"{1:>8} {self.ff.comb_rule:>12} {'yes':>12} {self.ff.fudge_lj:>12} "
                      f"{self.ff.fudge_q:>12}\n\n\n")

            top.write("; Include the molecule ITP\n")
            top.write(f'#include "./{self.ff.mol_name}_qforce.itp"\n\n\n')

            size = len(self.ff.mol_name)
            top.write("[ system ]\n")
            top.write(f"; {' '*(size-6)}name\n")
            top.write(f"{' '*(6-size)}{self.ff.mol_name}\n\n\n")

            top.write("[ molecules ]\n")
            top.write(f"; {' '*(size-10)}compound    n_mol\n")
            top.write(f"{' '*(10-size)}{self.ff.mol_name}        1\n")

    def write_gro(self, directory, coords, box=[20., 20., 20.]):
        n_atoms = self.ff.n_atoms
        if self.ff.polar:
            n_atoms += len(self.ff.alpha_map.keys())
        coords_nm = coords*0.1
        with open(f"{directory}/gas.gro", "w") as gro:
            gro.write(f"{self.ff.mol_name}\n")
            gro.write(f"{n_atoms:>6}\n")
            for i, (a_name, coord) in enumerate(zip(self.ff.atom_names, coords_nm), start=1):
                gro.write(f"{1:>5}{self.ff.residue:<5}")
                gro.write(f"{a_name:>5}{i:>5}")
                gro.write(f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}\n")
            if self.ff.polar:
                for i, (atom, drude) in enumerate(self.ff.alpha_map.items(), start=1):
                    gro.write(f"{2:>5}{self.ff.residue:<5}{f'D{i}':>5}{drude+1:>5}")
                    gro.write(f"{coords_nm[atom][0]:>8.3f}{coords_nm[atom][1]:>8.3f}")
                    gro.write(f"{coords_nm[atom][2]:>8.3f}\n")
            gro.write(f'{box[0]:>12.5f}{box[1]:>12.5f}{box[2]:>12.5f}\n')

    def write_itp(self, directory):
        with open(f"{directory}/{self.ff.mol_name}_qforce.itp", "w") as itp:
            itp.write(LOGO_SEMICOL)
            self.write_itp_atoms_and_molecule(itp)
            if self.ff.polar:
                self.write_itp_polarization(itp)
            self.write_itp_bonds(itp)
            self.write_itp_angles(itp)
            self.write_itp_dihedrals(itp)
            self.write_itp_pairs(itp)
            self.write_itp_exclusions(itp)
            itp.write('\n')

    def convert_to_gromacs_nonbonded(self):
        a_types, nb_pairs, nb_1_4 = {}, {}, {}

        for pair, val in self.ff.non_bonded.lj_pairs.items():
            if self.ff.non_bonded.comb_rule != 1:
                if val[0] == 0:
                    a, b = 0, 0
                else:
                    a, b = calc_sigma_epsilon(val[0], val[1])
                    a *= 0.1
            else:
                a = val[0] * 1e-6
                b = val[1] * 1e-12

            if pair[0] == pair[1]:
                a_types[pair[0]] = [a, b]

            nb_pairs[pair] = [a, b]

        for pair, val in self.ff.non_bonded.lj_1_4.items():
            if self.ff.non_bonded.comb_rule != 1:
                a, b = calc_sigma_epsilon(val[0], val[1])
                a *= 0.1
            else:
                a = val[0] * 1e-6
                b = val[1] * 1e-12

            nb_1_4[pair] = [a, b]

        return a_types, nb_pairs, nb_1_4

    def write_itp_atoms_and_molecule(self, itp):
        gro_atomtypes, gro_nonbonded, gro_1_4 = self.convert_to_gromacs_nonbonded()

        # atom types
        itp.write("\n[ atomtypes ]\n")
        if self.ff.comb_rule == 1:
            itp.write(";   name  at_num     mass   charge  type           c6          c12\n")
        else:
            itp.write(";   name  at_num     mass   charge  type        sigma      epsilon\n")

        for lj_type, lj_params in gro_atomtypes.items():
            itp.write(
                f'{lj_type:>8} {self.ff.non_bonded.lj_atomic_number[lj_type]:>7} {0:>8.4f} {0:>8.4f} {"A":>5} ')
            itp.write(f'{lj_params[0]:>12.5e} {lj_params[1]:>12.5e}\n')

        if self.ff.polar:
            itp.write(f'{"DP":>8} {0:>8.4f} {0:>8.4f} {"S":>2} {0:>12.5e} {0:>12.5e}\n')

        # non-bonded pair types
        itp.write("\n[ nonbond_params ]\n")
        if self.ff.comb_rule == 1:
            itp.write(";  name1    name2   type             c6             c12\n")
        else:
            itp.write(";  name1    name2   type          sigma         epsilon\n")

        for lj_types, lj_params in gro_nonbonded.items():
            itp.write(f'{lj_types[0]:>8} {lj_types[1]:>8}      1')
            itp.write(f'{lj_params[0]:>15.5e} {lj_params[1]:>15.5e}\n')

        # Write 1-4 pair types
        if self.ff.n_excl == 2 and gro_1_4 != {}:
            itp.write("\n[ pairtypes ]\n")
            if self.ff.comb_rule == 1:
                itp.write(";  name1    name2   type             c6             c12\n")
            else:
                itp.write(";  name1    name2   type          sigma         epsilon\n")
            for lj_types, lj_params in gro_1_4.items():
                itp.write(f'{lj_types[0]:>8} {lj_types[1]:>8}      1')
                itp.write(f'{lj_params[0]:>15.5e} {lj_params[1]:>15.5e}\n')

        # molecule type
        space = " "*(len(self.ff.mol_name)-5)
        itp.write("\n[ moleculetype ]\n")
        itp.write(f";{space}name nrexcl\n")
        itp.write(f"{self.ff.mol_name}{3:>7}\n")

        # atoms
        itp.write("\n[ atoms ]\n")
        itp.write(";  nr      type  resnr  resnm    atom  cgrp      charge       mass\n")
        for i, (lj_type, a_name, q, mass) in enumerate(zip(self.ff.non_bonded.lj_types, self.ff.atom_names,
                                                           self.ff.q, self.ff.masses), start=1):
            itp.write(f'{i:>5} {lj_type:>9} {1:>6} {self.ff.residue:>6} {a_name:>7} {i:>5} ')
            itp.write(f'{q:>11.5f} {mass:>10.5f}\n')

        if self.ff.polar:
            for i, (atom, drude) in enumerate(self.ff.non_bonded.alpha_map.items(), start=1):
                itp.write(f'{drude+1:>5} {"DP":>9} {2:>6} {"DRU":>6} {f"D{i}":>7} {atom+1:>5}')
                itp.write(f'{-8.:>11.5f} {0.:>10.5f}\n')

    def write_itp_polarization(self, itp):
        # polarization
        itp.write("\n[ polarization ]\n")
        itp.write(";    i      j      f          alpha\n")
        for atom, drude in self.ff.non_bonded.alpha_map.items():
            alpha = self.ff.non_bonded.alpha[atom]*1e-3
            itp.write(f"{atom+1:>6} {drude+1:>6} {1:>6} {alpha:>14.8f}\n")

        # # thole polarization
        # if self.thole != []:
        #     itp.write("\n[ thole_polarization ]\n")
        #     itp.write(";   ai    di    aj    dj   f      a      alpha(i)      "
        #               "alpha(j)\n")
        # for tho in self.thole:
        #     itp.write("{:>6}{:>6}{:>6}{:>6}{:>4}{:>7.2f}{:>14.8f}{:>14.8f}\n".format(*tho))

    def write_itp_pairs(self, itp):
        if self.ff.pairs != []:
            itp.write("\n[ pairs ]\n")
            itp.write(";   ai     aj   func\n")
        for pair in self.ff.pairs:
            itp.write(f"{pair[0]+1:>6} {pair[1]+1:>6} {1:>6}\n")

    def write_itp_bonds(self, itp):
        itp.write("\n[ bonds ]\n")
        itp.write(";   ai     aj      f           r0     k_bond\n")
        for bond in self.ff.terms['bond']:
            ids = bond.atomids + 1
            equ = bond.equ * 0.1
            fconst = bond.fconst * 100
            itp.write(f'{ids[0]:>6} {ids[1]:>6} {1:>6} {equ:>12.7f} {fconst:>10.0f}\n')

        if self.ff.polar:
            itp.write(';   ai     aj      f - polar connections\n')
            for bond in self.ff.terms['bond']:
                a1, a2 = bond.atomids

                if a2 in self.ff.alpha_map.keys():
                    itp.write(f'{a1+1:>6} {self.ff.alpha_map[a2]+1:>6} {5:>6}\n')
                if a1 in self.ff.alpha_map.keys():
                    itp.write(f'{a2+1:>6} {self.ff.alpha_map[a1]+1:>6} {5:>6}\n')

    def write_itp_angles(self, itp):
        itp.write("\n[ angles ]\n")
        itp.write(";   ai     aj     ak      f       theta0      k_theta")
        if self.ff.urey:
            itp.write('           r0       k_bond')
        itp.write('\n')

        for angle in self.ff.terms['angle']:
            ids = angle.atomids + 1
            equ = np.degrees(angle.equ)
            fconst = angle.fconst

            if self.ff.urey:
                urey = [term for term in self.ff.terms['urey'] if np.array_equal(term.atomids, angle.atomids)]

            if not self.ff.urey or len(urey) == 0:
                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {1:>6} {equ:>12.5f} '
                          f'{fconst:>12.5f}\n')
            else:
                urey_equ = urey[0].equ * 0.1
                urey_fconst = urey[0].fconst * 100
                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {5:>6} {equ:>12.5f} '
                          f'{fconst:>12.5f} {urey_equ:>12.7f} {urey_fconst:>12.3f}\n')

        if 'cross_bond_bond' in self.ff.terms:
            if len(self.ff.terms['cross_bond_bond']) > 0:

                itp.write(
                    "\n;   ai     aj     ak      f         r0_1         r0_2    k_cross  - bond-bond coupling\n")
            for cross_bb in self.ff.terms['cross_bond_bond']:
                ids = cross_bb.atomids + 1
                equ1, equ2 = cross_bb.equ * 0.1
                fconst = - cross_bb.fconst * 100

                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {3:>6} {equ1:>12.7f} {equ2:>12.7f} '
                          f'{fconst:>10.1f} \n')

            print('bond-bond', len(self.ff.terms['cross_bond_bond']))
            for term in self.ff.terms['cross_bond_bond']:
                print(term.atomids+1, term.equ, -term.fconst)

        if 'cross_bond_angle' in self.ff.terms:
            print('bond-angle', len(self.ff.terms['cross_bond_angle']))
            for term in self.ff.terms['cross_bond_angle']:
                print(term.atomids+1, np.degrees(term.equ[0]), term.equ[1], -term.fconst)

        if 'cross_angle_angle' in self.ff.terms:
            print('angle-angle', len(self.ff.terms['cross_angle_angle']))
            for term in self.ff.terms['cross_angle_angle']:
                print(term.atomids+1, np.degrees(term.equ[0]),
                      np.degrees(term.equ[1]), -term.fconst)

        if '_cross_dihed_angle' in self.ff.terms:
            print('dihed-angle', len(self.ff.terms['_cross_dihed_angle']))
            for term in self.ff.terms['_cross_dihed_angle']:
                print(term.atomids+1, np.degrees(term.equ[0]),
                      np.degrees(term.equ[1]), -term.fconst)

        if '_cross_dihed_bond' in self.ff.terms:
            print('dihed-bond', len(self.ff.terms['_cross_dihed_bond']))
            for term in self.ff.terms['_cross_dihed_bond']:
                print(term.atomids+1, np.degrees(term.equ[0]),
                      term.equ[1], -term.fconst)

    def write_itp_dihedrals(self, itp):
        if len(self.ff.terms['dihedral']) > 0:
            itp.write("\n[ dihedrals ]\n")

        # rigid dihedrals
        if 'dihedral/rigid' in self.ff.terms and len(self.ff.terms['dihedral/rigid']) > 0:
            itp.write("; rigid dihedrals \n")
            itp.write(";   ai     aj     ak     al      f      theta0       k_theta\n")

            for dihed in self.ff.terms['dihedral/rigid']:
                ids = dihed.atomids + 1
                equ = np.degrees(dihed.equ)
                fconst = dihed.fconst

                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {2:>6} {equ:>11.3f} ')
                itp.write(f'{fconst:>13.3f}\n')

        # improper dihedrals
        if 'dihedral/improper' in self.ff.terms and len(self.ff.terms['dihedral/improper']) > 0:
            itp.write("; improper dihedrals \n")
            itp.write(";   ai     aj     ak     al      f      theta0       k_theta\n")

            for dihed in self.ff.terms['dihedral/improper']:
                ids = dihed.atomids + 1
                equ = np.degrees(dihed.equ)
                fconst = dihed.fconst

                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {2:>6} {equ:>11.3f} ')
                itp.write(f'{fconst:>13.3f}\n')

        # flexible dihedrals
        if len(self.ff.terms['dihedral/flexible']) > 0:
            itp.write("; flexible dihedrals \n")
            itp.write(';   ai     aj     ak     al      f          c0          c1          c2')
            itp.write('          c3          c4          c5\n')

        for dihed in self.ff.terms['dihedral/flexible']:
            ids = dihed.atomids + 1
            c = dihed.equ

            itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {3:>6} {c[0]:>11.3f} ')
            itp.write(f'{c[1]:>11.3f} {c[2]:>11.3f} {c[3]:>11.3f} {c[4]:>11.3f} {c[5]:>11.3f}\n')

        # inversion dihedrals
        if 'dihedral/inversion' in self.ff.terms and len(self.ff.terms['dihedral/inversion']) > 0:
            itp.write("; inversion dihedrals \n")
            itp.write(';   ai     aj     ak     al      f          c0          c1          c2\n')

            for dihed in self.ff.terms['dihedral/inversion']:
                ids = dihed.atomids + 1
                c0, c1, c2 = convert_to_inversion_rb(dihed.fconst, dihed.equ)
                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {3:>6}'
                          f'{c0:>11.3f} {c1:>11.3f} {c2:>11.3f} {0:>11.1f} {0:>11.1f} {0:>11.1f}\n')

    def write_itp_exclusions(self, itp):
        if any(len(exclusion) > 0 for exclusion in self.ff.exclusions):
            itp.write("\n[ exclusions ]\n")
        for i, exclusion in enumerate(self.ff.exclusions):
            if len(exclusion) > 0:
                exclusion = sorted(set(exclusion))
                itp.write("{} ".format(i+1))
                itp.write(("{} "*len(exclusion)).format(*exclusion))
                itp.write("\n")

    def add_restraints(self, restraints, directory, fc=1000):
        with open(f"{directory}/{self.ff.mol_name}_qforce.itp", "a") as itp:
            itp.write("[ dihedral_restraints ]\n")
            itp.write(";  ai    aj    ak    al  type       phi   dp   kfac\n")
            for restraint in restraints:
                a1, a2, a3, a4 = restraint[0]+1
                phi = np.degrees(restraint[1])
                itp.write(f'{a1:>5} {a2:>5} {a3:>5} {a4:>5} {1:>5} {phi:>10.4f}  0.0  {fc}\n')


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
    #             for j in (mol.topo.neighbors[self.n_excl-2][i] +
    #                       mol.topo.neighbors[self.n_excl-1][i]):
    #                 if alphas[j] > 0:
    #                     ff.exclu[drude[i]-1].extend([drude[j]])
    #             for j in mol.topo.neighbors[self.n_excl-1][i]:
    #                 ff.exclu[drude[i]-1].extend([j+1])
    #             ff.exclu[drude[i]-1].sort()
    #             # thole polarizability
    #             for neigh in [mol.topo.neighbors[n][i] for n in range(self.n_excl)]:
    #                 for j in neigh:
    #                     if i < j and alphas[j] > 0:
    #                         ff.thole.append([i+1, drude[i], j+1, drude[j], "2", 2.6, alpha,
    #                                          alphas[j]])

    # def read_itp(self, itp_file):
    #     with open(itp_file, "r") as itp:
    #         in_section = []
    #         bond_atoms, bond_r0, bond_k = [], [], []

    #         for line in itp:
    #             low_line = line.lower().strip().replace(" ", "")
    #             unsplit = line
    #             line = line.split()
    #             if low_line == "" or low_line[0] == ";":
    #                 continue
    #             elif "[" in low_line and "]" in low_line:
    #                 open_bra = low_line.index("[") + 1
    #                 close_bra = low_line.index("]")
    #                 in_section = low_line[open_bra:close_bra]
    #             elif in_section == "atomtypes":
    #                 self.atom_types.append([line[0], float(line[1]),
    #                                         float(line[2]), line[3],
    #                                         float(line[4]), float(line[5])])
    #                 self.atype.append(line[0])
    #                 self.c6.append(line[4])
    #                 self.c12.append(line[5])
    #             elif in_section == "moleculetype":
    #                 self.mol_type = line[0]
    #             elif in_section == "atoms":
    #                 self.atoms.append([int(line[0]), line[1], int(line[2]),
    #                                    line[3], line[4], line[5], float(line[6]),
    #                                    float(line[7])])
    #                 self.natom += 1
    #             elif in_section == "bonds":
    #                 bond_atoms = (line[0:2])
    #                 bond_r0 = (float(line[3]) * 10)
    #                 bond_k = (float(line[4]) / 100)
    #                 self.bond.append([bond_atoms, bond_r0, bond_k])
    #                 self.bonds.append(unsplit)
    #             elif in_section == "angles":
    #                 angle_atoms = (line[0:3])
    #                 angle_theta0 = float(line[4])
    #                 angle_k = float(line[5])
    #                 self.angle.append([angle_atoms, angle_theta0, angle_k])
    #                 self.angles.append(unsplit)
    #             elif in_section == "dihedrals":
    #                 self.dihedrals.append(unsplit)
    #                 if line[4] == "3":
    #                     self.rbdihed.append([line[0:4], line[4:]])
    #                 elif line[4] == "2":
    #                     self.idihed.append([line[0:4], line[4:]])

    #             elif in_section == "pairs":
    #                 self.pairs.append(sorted([int(line[0]), int(line[1])]))
