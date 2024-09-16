import numpy as np
#
from colt import Colt
#
from .gromacs import Gromacs
from .openmm import OpenMM
from .mchem import MChem
from ..elements import ATOM_SYM, ATOMMASS


class ForceField(Colt):

    implemented_md_software = {'gromacs': Gromacs,
                               'openmm': OpenMM,
                               'mchem': MChem,
                               }

    _user_input = """
# MD software to print the final results in
output_software = gromacs :: str :: gromacs, openmm, mchem

# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent
n_equiv = 4 :: int

# Number of first n neighbors to exclude in the forcefield
n_excl = 2 :: int :: [2, 3]

# Lennard jones method for the forcefield
lennard_jones = opls_auto :: str :: [gromos_auto, gromos, opls_auto, opls, gaff, gaff2, charmm36, ext]

# Use GDMA to compute multipoles
do_multipole = false :: bool

# GDMA executable
gdma_exec = gdma :: str

# Use externally provided point charges in the file "ext_q" in the job directyory
ext_charges = no :: bool

# Scale QM charges to account for condensed phase polarization (should be set to 1 for gas phase)
charge_scaling = 1.2 :: float

# If user chooses ext_charges=True, by default fragments will still use the chosen QM method for
# determining fragment charges. This is to avoid spuriously high charges on capping hydrogens.
# However, using QM charges for fragments and ext_charges for the molecule can also result in
# inaccuracies if these two charges are very different.
use_ext_charges_for_frags = no :: bool

# Additional exclusions (GROMACS format)
exclusions = :: literal

# Switch standard non-bonded interactions between two atoms to pair interactions
# (provide atom pairs in each row)
pairs = :: literal

# Path for the external FF library for Lennard-Jones parameters (GROMACS format).
ext_lj_lib = :: folder, optional

# Lennard-Jones fudge parameter for 1-4 interactions for when "lennard_jones" is set to "ext".
ext_lj_fudge = :: float, optional

# Coulomb fudge parameter for 1-4 interactions for when "lennard_jones" is set to "ext".
ext_q_fudge =  :: float, optional

# Lennard-Jones combinations rules for when "lennard_jones" is set to "ext" (GROMACS numbering).
ext_comb_rule =  :: int, optional :: [1, 2, 3]

# Name of the atom type for capping hydrogens for when "lennard_jones" is set to "ext"
ext_h_cap = :: str, optional

# Set all dihedrals as rigid (no dihedral scans)
all_rigid = no :: bool

# Only put bond-angle coupling when both bond atoms overlap with angle atoms
ba_couple_1_shared = no :: bool

# TEMP: Chosen period for dihedrals
cosine_dihed_period = 0 :: int :: [0, 2, 3]

# Residue name printed on the force field file (Max 5 characters)
res_name = MOL :: str

"""

    def __init__(self, software, job, config, mol, neighbors, exclude_all=[]):
        self.mol_name = job.name
        self.n_atoms = mol.n_atoms
        self.atomids = mol.atomids
        self.q = np.copy(mol.non_bonded.q)
        self.residue = config.ff.res_name[:5]
        self.comb_rule = mol.non_bonded.comb_rule
        self.fudge_lj = mol.non_bonded.fudge_lj
        self.fudge_q = mol.non_bonded.fudge_q
        self.n_excl = config.ff.n_excl
        self.atom_names, self.symbols = self.get_atom_names()
        self.masses = [round(ATOMMASS[i], 5) for i in self.atomids]
        self.exclusions = self.make_exclusions(mol.non_bonded, neighbors, exclude_all)
        self.pairs = mol.non_bonded.pairs
        self.cosine_dihed_period = config.ff.cosine_dihed_period
        self.do_multipole = config.ff.do_multipole
        self.terms = mol.terms
        self.topo = mol.topo
        self.non_bonded = mol.non_bonded
        self.bond_dissociation_energies = self._get_bond_dissociation_energies(job.md_data)

        self.software = self._set_md_software(software)

    def _set_md_software(self, selection):
        try:
            software = self.implemented_md_software[selection](self)
        except KeyError:
            raise KeyError(f'"{selection}" software is not implemented.')
        return software

    def _get_bond_dissociation_energies(self, md_data):
        dissociation_energies = {}

        bde_dict = self._read_bond_dissociation_energy_csv(md_data)

        for a1, a2, props in self.topo.graph.edges(data=True):
            print(a1, a2, props)
            if props['type'] not in bde_dict:
                raise ValueError('Morse potential chosen, but dissociation energy not known for this atom number pair '
                                 f'with the bond order in parenthesis: {props["type"]}. '
                                 'You can add this too csv file in the data directory')

            neighs1 = [edge[2]['type'] for edge in self.topo.graph.edges(a1, data=True)]
            neighs2 = [edge[2]['type'] for edge in self.topo.graph.edges(a2, data=True)]
            e_dis, n_matches = self._choose_bond_type(bde_dict[props['type']], neighs1, neighs2)

            if self.atomids[a1] == self.atomids[a2]:
                e_dis2, n_matches2 = self._choose_bond_type(bde_dict[props['type']], neighs1, neighs2, 1, 0)
                if n_matches2 > n_matches:
                    e_dis = e_dis2

            dissociation_energies[(a1, a2)] = float(e_dis)
        return dissociation_energies

    def _read_bond_dissociation_energy_csv(self, md_data):
        bde_dict = {}

        with open(f'{md_data}/bond_dissociation_energy.csv', 'r') as file:
            file.readline()
            for line in file:
                a1, a2, b_order, a1_neighs, a2_neighs, de = line.split(',')[:6]
                a1_neighs = self._read_bond_dissociation_neighbors(a1, a1_neighs)
                a2_neighs = self._read_bond_dissociation_neighbors(a2, a2_neighs)
                name = f'{a1}({float(b_order):.1f}){a2}'

                if name in bde_dict:
                    bde_dict[name].append((a1_neighs, a2_neighs, de))
                else:
                    bde_dict[name] = [(a1_neighs, a2_neighs, de)]

        return bde_dict

    @staticmethod
    def _choose_bond_type(bde_list, neighs1, neighs2, ndx1=0, ndx2=1):
        highest_match = 0
        match_type = None

        for bde_type in bde_list:
            current_sum1 = sum(typ in neighs1 for typ in bde_type[ndx1])
            current_sum2 = sum(typ in neighs2 for typ in bde_type[ndx2])
            current_sum = current_sum1 + current_sum2

            if current_sum > highest_match:
                highest_match = current_sum
                match_type = bde_type
            elif highest_match == 0 and bde_type[0] == [] and bde_type[1] == []:
                match_type = bde_type

        return match_type[2], highest_match

    @staticmethod
    def _read_bond_dissociation_neighbors(a, neighs):
        neighs_formatted = []
        if neighs != '*' and neighs != '':
            neighs = neighs.split()
            for neigh in neighs:
                a_neigh, bo_neigh = neigh.split('_')
                neighs_formatted.append(f'{min(a, a_neigh)}({float(bo_neigh):.1f}){max(a, a_neigh)}')
        return neighs_formatted

    def make_exclusions(self, non_bonded, neighbors, exclude_all):
        exclusions = [[] for _ in range(self.n_atoms)]

        # input exclusions  for exclusions if outside of n_excl
        for a1, a2 in non_bonded.exclusions+non_bonded.pairs:
            if all([a2 not in neighbors[i][a1] for i in range(self.n_excl+1)]):
                exclusions[a1].append(a2+1)

        # fragment capping atom exclusions
        for i in exclude_all:
            exclusions[i].extend(np.arange(1, self.n_atoms+1))

        return exclusions

    def get_atom_names(self):
        atom_names = []
        symbols = []
        atom_dict = {}

        for i, elem in enumerate(self.atomids):
            sym = ATOM_SYM[elem]
            symbols.append(sym)
            if sym not in atom_dict.keys():
                atom_dict[sym] = 1
            else:
                atom_dict[sym] += 1
            atom_names.append(f'{sym}{atom_dict[sym]}')
        return atom_names, np.array(symbols)

