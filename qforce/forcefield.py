import numpy as np
#
from .elements import ATOM_SYM, ATOMMASS
from .molecule.non_bonded import calc_sigma_epsilon
from .forces import convert_to_inversion_rb
from .misc import LOGO_SEMICOL


class ForceField():
    def __init__(self, job_name, config, mol, neighbors, exclude_all=[]):
        self.polar = config.ff._polar
        self.mol_name = job_name
        self.n_atoms = mol.n_atoms
        self.elements = mol.elements
        self.q = self.set_charge(mol.non_bonded)
        self.residue = config.ff.res_name[:5]
        self.comb_rule = mol.non_bonded.comb_rule
        self.fudge_lj = mol.non_bonded.fudge_lj
        self.fudge_q = mol.non_bonded.fudge_q
        self.urey = config.terms.urey
        self.n_excl = config.ff.n_excl
        self.atom_names = self.get_atom_names()
        self.masses = [round(ATOMMASS[i], 5) for i in self.elements]
        self.exclusions = self.make_exclusions(mol.non_bonded, neighbors, exclude_all)
        self.pairs = self.make_pairs(neighbors, mol.non_bonded)

        if self.polar:
            self.polar_title = '_polar'
        else:
            self.polar_title = ''

    def write_gromacs(self, directory, mol, coords):
        self.write_itp(mol, directory)
        self.write_top(directory)
        self.write_gro(directory, coords, mol.non_bonded.alpha_map)

    def write_top(self, directory):
        with open(f"{directory}/gas{self.polar_title}.top", "w") as top:
            # defaults
            top.write("\n[ defaults ]\n")
            top.write("; nbfunc    comb-rule    gen-pairs      fudgeLJ      fudgeQQ\n")
            top.write(f"{1:>8} {self.comb_rule:>12} {'yes':>12} {self.fudge_lj:>12} "
                      f"{self.fudge_q:>12}\n\n\n")

            top.write("; Include the molecule ITP\n")
            top.write(f'#include "./{self.mol_name}_qforce{self.polar_title}.itp"\n\n\n')

            size = len(self.mol_name)
            top.write("[ system ]\n")
            top.write(f"; {' '*(size-6)}name\n")
            top.write(f"{' '*(6-size)}{self.mol_name}\n\n\n")

            top.write("[ molecules ]\n")
            top.write(f"; {' '*(size-10)}compound    n_mol\n")
            top.write(f"{' '*(10-size)}{self.mol_name}        1\n")

    def write_gro(self, directory, coords, alpha_map, box=[20., 20., 20.]):
        n_atoms = self.n_atoms
        if self.polar:
            n_atoms += len(alpha_map.keys())
        coords_nm = coords*0.1
        with open(f"{directory}/gas{self.polar_title}.gro", "w") as gro:
            gro.write(f"{self.mol_name}\n")
            gro.write(f"{n_atoms:>6}\n")
            for i, (a_name, coord) in enumerate(zip(self.atom_names, coords_nm), start=1):
                gro.write(f"{1:>5}{self.residue:<5}")
                gro.write(f"{a_name:>5}{i:>5}")
                gro.write(f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}\n")
            if self.polar:
                for i, (atom, drude) in enumerate(alpha_map.items(), start=1):
                    gro.write(f"{2:>5}{self.residue:<5}{f'D{i}':>5}{drude+1:>5}")
                    gro.write(f"{coords_nm[atom][0]:>8.3f}{coords_nm[atom][1]:>8.3f}")
                    gro.write(f"{coords_nm[atom][2]:>8.3f}\n")
            gro.write(f'{box[0]:>12.5f}{box[1]:>12.5f}{box[2]:>12.5f}\n')

    def write_itp(self, mol, directory):
        with open(f"{directory}/{self.mol_name}_qforce{self.polar_title}.itp", "w") as itp:
            itp.write(LOGO_SEMICOL)
            self.write_itp_atoms_and_molecule(itp, mol.non_bonded)
            if self.polar:
                self.write_itp_polarization(itp, mol.non_bonded)
            self.write_itp_bonds(itp, mol.terms, mol.non_bonded.alpha_map)
            self.write_itp_angles(itp, mol.terms)
            self.write_itp_dihedrals(itp, mol.terms)
            self.write_itp_pairs(itp)
            self.write_itp_exclusions(itp)
            itp.write('\n')

    def convert_to_gromacs_nonbonded(self, non_bonded):
        a_types, nb_pairs, nb_1_4 = {}, {}, {}

        for pair, val in non_bonded.lj_pairs.items():
            if non_bonded.comb_rule != 1:
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

        for pair, val in non_bonded.lj_1_4.items():
            if non_bonded.comb_rule != 1:
                a, b = calc_sigma_epsilon(val[0], val[1])
                a *= 0.1
            else:
                a = val[0] * 1e-6
                b = val[1] * 1e-12

            nb_1_4[pair] = [a, b]

        return a_types, nb_pairs, nb_1_4

    def write_itp_atoms_and_molecule(self, itp, non_bonded):
        gro_atomtypes, gro_nonbonded, gro_1_4 = self.convert_to_gromacs_nonbonded(non_bonded)

        # atom types
        itp.write("\n[ atomtypes ]\n")
        if self.comb_rule == 1:
            itp.write(";   name  at_num     mass   charge  type           c6          c12\n")
        else:
            itp.write(";   name  at_num     mass   charge  type        sigma      epsilon\n")

        for lj_type, lj_params in gro_atomtypes.items():
            itp.write(f'{lj_type:>8} {non_bonded.lj_atomic_number[lj_type]:>7} {0:>8.4f} {0:>8.4f} {"A":>5} ')
            itp.write(f'{lj_params[0]:>12.5e} {lj_params[1]:>12.5e}\n')

        if self.polar:
            itp.write(f'{"DP":>8} {0:>8.4f} {0:>8.4f} {"S":>2} {0:>12.5e} {0:>12.5e}\n')

        # non-bonded pair types
        itp.write("\n[ nonbond_params ]\n")
        if self.comb_rule == 1:
            itp.write(";  name1    name2   type             c6             c12\n")
        else:
            itp.write(";  name1    name2   type          sigma         epsilon\n")

        for lj_types, lj_params in gro_nonbonded.items():
            itp.write(f'{lj_types[0]:>8} {lj_types[1]:>8}      1')
            itp.write(f'{lj_params[0]:>15.5e} {lj_params[1]:>15.5e}\n')

        # Write 1-4 pair types
        if self.n_excl == 2 and gro_1_4 != {}:
            itp.write("\n[ pairtypes ]\n")
            if self.comb_rule == 1:
                itp.write(";  name1    name2   type             c6             c12\n")
            else:
                itp.write(";  name1    name2   type          sigma         epsilon\n")
            for lj_types, lj_params in gro_1_4.items():
                itp.write(f'{lj_types[0]:>8} {lj_types[1]:>8}      1')
                itp.write(f'{lj_params[0]:>15.5e} {lj_params[1]:>15.5e}\n')

        # molecule type
        space = " "*(len(self.mol_name)-5)
        itp.write("\n[ moleculetype ]\n")
        itp.write(f";{space}name nrexcl\n")
        itp.write(f"{self.mol_name}{3:>7}\n")

        # atoms
        itp.write("\n[ atoms ]\n")
        itp.write(";  nr      type  resnr  resnm    atom  cgrp      charge       mass\n")
        for i, (lj_type, a_name, q, mass) in enumerate(zip(non_bonded.lj_types, self.atom_names,
                                                           self.q, self.masses), start=1):
            itp.write(f'{i:>5} {lj_type:>9} {1:>6} {self.residue:>6} {a_name:>7} {i:>5} ')
            itp.write(f'{q:>11.5f} {mass:>10.5f}\n')

        if self.polar:
            for i, (atom, drude) in enumerate(non_bonded.alpha_map.items(), start=1):
                itp.write(f'{drude+1:>5} {"DP":>9} {2:>6} {"DRU":>6} {f"D{i}":>7} {atom+1:>5}')
                itp.write(f'{-8.:>11.5f} {0.:>10.5f}\n')

    def write_itp_polarization(self, itp, non_bonded):
        # polarization
        itp.write("\n[ polarization ]\n")
        itp.write(";    i      j      f          alpha\n")
        for atom, drude in non_bonded.alpha_map.items():
            alpha = non_bonded.alpha[atom]*1e-3
            itp.write(f"{atom+1:>6} {drude+1:>6} {1:>6} {alpha:>14.8f}\n")

        # # thole polarization
        # if self.thole != []:
        #     itp.write("\n[ thole_polarization ]\n")
        #     itp.write(";   ai    di    aj    dj   f      a      alpha(i)      "
        #               "alpha(j)\n")
        # for tho in self.thole:
        #     itp.write("{:>6}{:>6}{:>6}{:>6}{:>4}{:>7.2f}{:>14.8f}{:>14.8f}\n".format(*tho))

    def write_itp_pairs(self, itp):
        if self.pairs != []:
            itp.write("\n[ pairs ]\n")
            itp.write(";   ai     aj   func\n")
        for pair in self.pairs:
            itp.write(f"{pair[0]+1:>6} {pair[1]+1:>6} {1:>6}\n")

    def write_itp_bonds(self, itp, terms, alpha_map):
        itp.write("\n[ bonds ]\n")
        itp.write(";   ai     aj      f         r0     k_bond\n")
        for bond in terms['bond']:
            ids = bond.atomids + 1
            equ = bond.equ * 0.1
            fconst = bond.fconst * 100
            itp.write(f'{ids[0]:>6} {ids[1]:>6} {1:>6} {equ:>10.5f} {fconst:>10.0f}\n')

        if self.polar:
            itp.write(';   ai     aj      f - polar connections\n')
            for bond in terms['bond']:
                a1, a2 = bond.atomids

                if a2 in alpha_map.keys():
                    itp.write(f'{a1+1:>6} {alpha_map[a2]+1:>6} {5:>6}\n')
                if a1 in alpha_map.keys():
                    itp.write(f'{a2+1:>6} {alpha_map[a1]+1:>6} {5:>6}\n')

    def write_itp_angles(self, itp, terms):
        itp.write("\n[ angles ]\n")
        itp.write(";   ai     aj     ak      f     theta0    k_theta")
        if self.urey:
            itp.write('         r0     k_bond')
        itp.write('\n')

        for angle in terms['angle']:
            ids = angle.atomids + 1
            equ = np.degrees(angle.equ)
            fconst = angle.fconst

            if self.urey:
                urey = [term for term in terms['urey'] if np.array_equal(term.atomids,
                                                                         angle.atomids)]
            if not self.urey or len(urey) == 0:
                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {1:>6} {equ:>10.3f} '
                          f'{fconst:>10.3f}\n')
            else:
                urey_equ = urey[0].equ * 0.1
                urey_fconst = urey[0].fconst * 100
                itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {5:>6} {equ:>10.3f} '
                          f'{fconst:>10.3f} {urey_equ:>10.5f} {urey_fconst:>10.1f}\n')

    def write_itp_dihedrals(self, itp, terms):
        if len(terms['dihedral']) > 0:
            itp.write("\n[ dihedrals ]\n")

        # rigid dihedrals
        if len(terms['dihedral/rigid']) > 0:
            itp.write("; rigid dihedrals \n")
            itp.write(";   ai     aj     ak     al      f      theta0       k_theta\n")

        for dihed in terms['dihedral/rigid']:
            ids = dihed.atomids + 1
            equ = np.degrees(dihed.equ)
            fconst = dihed.fconst

            itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {2:>6} {equ:>11.3f} ')
            itp.write(f'{fconst:>13.3f}\n')

        # improper dihedrals
        if len(terms['dihedral/improper']) > 0:
            itp.write("; improper dihedrals \n")
            itp.write(";   ai     aj     ak     al      f      theta0       k_theta\n")

        for dihed in terms['dihedral/improper']:
            ids = dihed.atomids + 1
            equ = np.degrees(dihed.equ)
            fconst = dihed.fconst

            itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {2:>6} {equ:>11.3f} ')
            itp.write(f'{fconst:>13.3f}\n')

        # flexible dihedrals
        if len(terms['dihedral/flexible']) > 0:
            itp.write("; flexible dihedrals \n")
            itp.write(';   ai     aj     ak     al      f          c0          c1          c2')
            itp.write('          c3          c4          c5\n')

        for dihed in terms['dihedral/flexible']:
            ids = dihed.atomids + 1
            c = dihed.equ

            itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {3:>6} {c[0]:>11.3f} ')
            itp.write(f'{c[1]:>11.3f} {c[2]:>11.3f} {c[3]:>11.3f} {c[4]:>11.3f} {c[5]:>11.3f}\n')

        # inversion dihedrals
        if len(terms['dihedral/inversion']) > 0:
            itp.write("; inversion dihedrals \n")
            itp.write(';   ai     aj     ak     al      f          c0          c1          c2\n')

        for dihed in terms['dihedral/inversion']:
            ids = dihed.atomids + 1
            c0, c1, c2 = convert_to_inversion_rb(dihed.fconst, dihed.equ)
            itp.write(f'{ids[0]:>6} {ids[1]:>6} {ids[2]:>6} {ids[3]:>6} {3:>6}'
                      f'{c0:>11.3f} {c1:>11.3f} {c2:>11.3f} {0:>11.1f} {0:>11.1f} {0:>11.1f}\n')

    def write_itp_exclusions(self, itp):
        if any(len(exclusion) > 0 for exclusion in self.exclusions):
            itp.write("\n[ exclusions ]\n")
        for i, exclusion in enumerate(self.exclusions):
            if len(exclusion) > 0:
                exclusion = sorted(set(exclusion))
                itp.write("{} ".format(i+1))
                itp.write(("{} "*len(exclusion)).format(*exclusion))
                itp.write("\n")

    def make_pairs(self, neighbors, non_bonded):
        polar_pairs = []

        if self.n_excl == 2:
            if self.polar:
                for a1, a2 in non_bonded.pairs:
                    if a2 in non_bonded.alpha_map.keys():
                        polar_pairs.append([a1, non_bonded.alpha_map[a2]])
                    if a1 in non_bonded.alpha_map.keys():
                        polar_pairs.append([a2, non_bonded.alpha_map[a1]])
                    if a1 in non_bonded.alpha_map.keys() and a2 in non_bonded.alpha_map.keys():
                        polar_pairs.append([non_bonded.alpha_map[a1], non_bonded.alpha_map[a2]])

        return non_bonded.pairs+polar_pairs

    def make_exclusions(self, non_bonded, neighbors, exclude_all):
        exclusions = [[] for _ in range(self.n_atoms)]

        # input exclusions  for exclusions if outside of n_excl
        for a1, a2 in non_bonded.exclusions+non_bonded.pairs:
            if all([a2 not in neighbors[i][a1] for i in range(self.n_excl+1)]):
                exclusions[a1].append(a2+1)

        # fragment capping atom exclusions
        for i in exclude_all:
            exclusions[i].extend(np.arange(1, self.n_atoms+1))

        if self.polar:
            exclusions = self.polarize_exclusions(non_bonded.alpha_map, non_bonded.exclusions,
                                                  neighbors, exclude_all, exclusions)

        return exclusions

    def polarize_exclusions(self, alpha_map, input_exclusions, neighbors, exclude_all, exclusions):
        n_polar_atoms = len(alpha_map.keys())
        exclusions.extend([[] for _ in range(n_polar_atoms)])

        # input exclusions
        for exclu in input_exclusions:
            if exclu[0] in alpha_map.keys():
                exclusions[alpha_map[exclu[0]]].append(exclu[1]+1)
            if exclu[1] in alpha_map.keys():
                exclusions[alpha_map[exclu[1]]].append(exclu[0]+1)
            if exclu[0] in alpha_map.keys() and exclu[1] in alpha_map.keys():
                exclusions[alpha_map[exclu[0]]].append(alpha_map[exclu[1]]+1)

        # fragment capping atom exclusions
        for i in exclude_all:
            exclusions[i].extend(np.arange(self.n_atoms+1, self.n_atoms+n_polar_atoms+1))
            if i in alpha_map.keys():
                exclusions[alpha_map[i]].extend(np.arange(1, self.n_atoms+n_polar_atoms+1))

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
            itp.write(";  ai    aj    ak    al  type       phi   dp   kfac\n")
            for restraint in restraints:
                a1, a2, a3, a4 = restraint[0]+1
                phi = np.degrees(restraint[1])
                itp.write(f'{a1:>5} {a2:>5} {a3:>5} {a4:>5} {1:>5} {phi:>10.4f}  0.0  {fc}\n')

    def set_charge(self, non_bonded):
        q = np.copy(non_bonded.q)
        if self.polar:
            q[list(non_bonded.alpha_map.keys())] += 8
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
