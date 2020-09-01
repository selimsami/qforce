from qctools.colt import AskQuestions, Colt

questions = """
# Directory where the fragments are saved
frag_lib = ~/qforce_fragments :: str 

# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent 
n_equiv = 4 :: int

# Number of first n neighbors to exclude in the forcefield
n_excl = 3 :: int :: [2, 3]

# Point charges used in the forcefield 
point_charges = d4 :: str :: [d4, cm5, esp, ext]

# Lennard jones method for the forcefield
lennard_jones = d4 :: str :: [d4, ext]

# Scaling of the vibrational frequencies (not implemented)
vibr_coef = 1.0 :: float 

# Use Urey-Bradley angle terms
urey = yes :: bool 

# Use Bond-Angle cross term
cross_bond_angle = no :: bool

job_script = :: literal
exclusions = :: literal

[qm]

# Number of dihedral scan steps to perform 
scan_no = 23 :: int

# Step size for the dihedral scan
scan_step = 15.0 :: float 

# Total charge of the system 
charge = 0 :: int 

# Multiplicity of the system
multi = 1 :: int

# QM method to be used
method = PBEPBE :: str

# QM basis set to be used
basis = 6-31+G(D) :: str

# Dispersion correction to include
disp = GD3BJ :: str

# Number of processors for the QM calculation
nproc = 1 :: int

# Memory setting for the QM calculation
mem = 4GB :: str

"""


class QForce(Colt):
    _questions = """
# Directory where the fragments are saved
frag_lib = ~/qforce_fragments :: str 

# Number of n equivalent neighbors needed to consider two atoms equivalent
# Negative values turns off equivalence, 0 makes same elements equivalent 
n_equiv = 4 :: int

# Number of first n neighbors to exclude in the forcefield
n_excl = 3 :: int :: [2, 3]

# Point charges used in the forcefield 
point_charges = d4 :: str :: [d4, cm5, esp, ext]

# Lennard jones method for the forcefield
lennard_jones = d4 :: str :: [d4, ext]

# Scaling of the vibrational frequencies (not implemented)
vibr_coef = 1.0 :: float 

# Use Urey-Bradley angle terms
urey = yes :: bool 

# Use Bond-Angle cross term
cross_bond_angle = no :: bool

job_script = :: literal
exclusions = :: literal

[qm]

# Number of dihedral scan steps to perform 
scan_no = 23 :: int

# Step size for the dihedral scan
scan_step = 15.0 :: float 

# Total charge of the system 
charge = 0 :: int 

# Multiplicity of the system
multi = 1 :: int

# QM method to be used
method = PBEPBE :: str

# QM basis set to be used
basis = 6-31+G(D) :: str

# Dispersion correction to include
disp = GD3BJ :: str

# Number of processors for the QM calculation
nproc = 1 :: int

# Memory setting for the QM calculation
mem = 4GB :: str

"""

    def __init__(self, answers):
        for key, value in answers.items():
            setattr(self, key, value)

    @classmethod
    def from_config(cls, answers):
        return cls(answers)


class CommandLine(Colt):

    _questions = """
    # Directory where the fragments are saved
    frag_lib = ~/qforce_fragments :: str 

    # Number of n equivalent neighbors needed to consider two atoms equivalent
    # Negative values turns off equivalence, 0 makes same elements equivalent 
    n_equiv = 4 :: int

    file = not_given :: str
    """

    def __init__(self, answers):
        print(answers)
        if answers['file'] != 'not_given':
            quest = AskQuestions("qforce", questions, config=answers['file'])
            answers.update(quest.check_only('answer.txt'))
        print(answers)

    @classmethod
    def from_config(cls, answers):
        return answers


# print(CommandLine.from_commandline())

qforce = QForce.from_questions("qforce", config='qforce.ini', check_only=True)
print(qforce.frag_lib)
answers = questions.check_only('text.ini')
#print(answers)
