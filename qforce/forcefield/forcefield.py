import numpy as np
#
from colt import Colt
#
from .gromacs import Gromacs
from .openmm import OpenMM
from qforce.elements import ATOM_SYM, ATOMMASS

implemented_md_software = {'gromacs': Gromacs,
                           'openmm': OpenMM,
                           }


class ForceField(Colt):
    _user_input = """
# MD software to print the final results in
output_software = gromacs :: str :: gromacs, openmm

# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent
n_equiv = 4 :: int

# Number of first n neighbors to exclude in the forcefield
n_excl = 2 :: int :: [2, 3]

# Lennard jones method for the forcefield
lennard_jones = opls_auto :: str :: [gromos_auto, gromos, opls_auto, opls, gaff, gaff2, charmm36, ext]

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

# write the force field with Morse potential
morse = no :: bool

# write the force field with cosine angle potential
cos_angle = no :: bool

# Use D4 method
_d4 = no :: bool

# Residue name printed on the force field file (Max 5 characters)
res_name = MOL :: str

# Polarize a coordinate file and quit (requires itp_file)
_polarize = no :: bool

# Name of itp file (only needed for polarize option)
_itp_file = itp_file_missing :: str

# Make a polarizable FF
_polar = no :: bool

# Scale the C6 dispersion interactions in the polarizable version of the FF
_polar_c6_scale = 0.8 :: float

# Specifically not scale some of the atoms
_polar_not_scale_c6 = :: literal

# Manual polarizabilities in the file ext_alpha
_ext_alpha = no :: bool

"""

    def __init__(self, software, job_name, config, mol, neighbors, exclude_all=[]):
        self.polar = config.ff._polar
        self.mol_name = job_name
        self.n_atoms = mol.n_atoms
        self.elements = mol.elements
        self.q = self.set_charge(mol.non_bonded)
        self.residue = config.ff.res_name[:5]
        self.comb_rule = mol.non_bonded.comb_rule
        self.fudge_lj = mol.non_bonded.fudge_lj
        self.fudge_q = mol.non_bonded.fudge_q
        self.alpha_map = mol.non_bonded.alpha_map
        self.urey = config.terms.urey
        self.n_excl = config.ff.n_excl
        self.atom_names, self.symbols = self.get_atom_names()
        self.masses = [round(ATOMMASS[i], 5) for i in self.elements]
        self.exclusions = self.make_exclusions(mol.non_bonded, neighbors, exclude_all)
        self.pairs = self.make_pairs(neighbors, mol.non_bonded)
        self.morse = config.ff.morse
        self.cos_angle = config.ff.cos_angle
        self.terms = mol.terms
        self.topo = mol.topo
        self.non_bonded = mol.non_bonded

        self.software = self._set_md_software(software)

    def _set_md_software(self, selection):
        try:
            software = implemented_md_software[selection](self)
        except KeyError:
            raise KeyError(f'"{selection}" software is not implemented.')
        return software

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
        symbols = []
        atom_dict = {}

        for i, elem in enumerate(self.elements):
            sym = ATOM_SYM[elem]
            symbols.append(sym)
            if sym not in atom_dict.keys():
                atom_dict[sym] = 1
            else:
                atom_dict[sym] += 1
            atom_names.append(f'{sym}{atom_dict[sym]}')
        return atom_names, np.array(symbols)

    def set_charge(self, non_bonded):
        q = np.copy(non_bonded.q)
        if self.polar:
            q[list(non_bonded.alpha_map.keys())] += 8
        return q
