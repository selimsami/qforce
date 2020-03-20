from .elements import ATOM_SYM


class Forcefield():
    """
    Scope:
    -----
    Read GROMACS force field files (ITP and GRO) and create an object
    """

    def __init__(self, itp_file=None, gro_file=None):
        self.atom_types = []
        self.atoms = []
        self.atype = []
        self.c6 = []
        self.c12 = []
        self.bond = []
        self.angle = []
        self.idihed = []
        self.rbdihed = []
        self.pairs = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.flexible = []
        self.constrained = []
        self.coords = []
        self.atomids = []
        self.symbols = []
        self.polar = []
        self.thole = []
        self.exclu = []
        self.natom = 0

        if itp_file is not None:
            self.read_itp(itp_file)
            self.maxresnr = self.atoms[-1][2]
        if gro_file is not None:
            self.read_gro(gro_file)

    def read_itp(self, itp_file):
        with open(itp_file, "r") as itp:
            in_section = []
            bond_atoms, bond_r0, bond_k = [], [], []

            for line in itp:
                low_line = line.lower().strip().replace(" ", "")
                unsplit = line
                line = line.split()
                if low_line == "" or low_line[0] == ";":
                    continue
                elif "[" in low_line and "]" in low_line:
                    open_bra = low_line.index("[") + 1
                    close_bra = low_line.index("]")
                    in_section = low_line[open_bra:close_bra]
                elif in_section == "atomtypes":
                    self.atom_types.append([line[0], float(line[1]),
                                            float(line[2]), line[3],
                                            float(line[4]), float(line[5])])
                    self.atype.append(line[0])
                    self.c6.append(line[4])
                    self.c12.append(line[5])
                elif in_section == "moleculetype":
                    self.mol_type = line[0]
                elif in_section == "atoms":
                    self.atoms.append([int(line[0]), line[1], int(line[2]),
                                       line[3], line[4], line[5], float(line[6]),
                                       float(line[7])])
                    self.natom += 1
                elif in_section == "bonds":
                    bond_atoms = (line[0:2])
                    bond_r0 = (float(line[3]) * 10)
                    bond_k = (float(line[4]) / 100)
                    self.bond.append([bond_atoms, bond_r0, bond_k])
                    self.bonds.append(unsplit)
                elif in_section == "angles":
                    angle_atoms = (line[0:3])
                    angle_theta0 = float(line[4])
                    angle_k = float(line[5])
                    self.angle.append([angle_atoms, angle_theta0, angle_k])
                    self.angles.append(unsplit)
                elif in_section == "dihedrals":
                    self.dihedrals.append(unsplit)
                    if line[4] == "3":
                        self.rbdihed.append([line[0:4], line[4:]])
                    elif line[4] == "2":
                        self.idihed.append([line[0:4], line[4:]])

                elif in_section == "pairs":
                    self.pairs.append(sorted([int(line[0]), int(line[1])]))

    def read_gro(self, gro_file):
        with open(gro_file, "r") as gro:
            self.title = gro.readline()
            self.gro_natom = int(gro.readline())
            for i in range(self.gro_natom):
                line = gro.readline()
                sym = ''.join(i for i in line[9:15].strip() if not i.isdigit())
                x = float(line[21:29].strip())
                y = float(line[29:37].strip())
                z = float(line[37:45].strip())
                self.symbols.append(sym)
                self.atomids.append(ATOM_SYM.index(sym))
                self.coords.append([x, y, z])
            self.box = gro.readline()
