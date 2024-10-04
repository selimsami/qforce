import os
#
from colt import Colt
import numpy as np
#
from .creator import CostumStructureCreator
from .additionalstructures import AdditionalStructureCreator



class Computations(Colt):

    _user_input = """
    energy_element_weights = 15000 :: float
    gradient_element_weights = 100 :: float
    hessian_element_weights = 0.1 :: float

    hessian_weight = 1 :: float
    dihedral_weight = 1 :: float
    """

    def __init__(self, creators, energy_ele_weight, gradient_ele_weight, hessian_ele_weight,
                 hessian_weight, dihedral_weight):
        self.creators = creators
        self.energy_weight = energy_ele_weight
        self.gradient_weight = gradient_ele_weight
        self.hessian_weight = hessian_ele_weight
        #
        self._hessian_weight = hessian_weight
        self._dihedral_weight = dihedral_weight

    def add_hessians(self, hessout):
        """add hessian"""
        self.creators['hessian'] = HessianOutput(self._hessian_weight, hessout)

    def add_dihedrals(self, scans):
        """add computed dihedrals"""
        self.creators['dihedrals'] = DihedralOutput(self._dihedral_weight, scans)

    def register(self, name, creator):
        """add a new creator"""
        if not isinstance(creator, CostumStructureCreator):
            raise ValueError(f"Can not register '{type(creator)}' "
                             "only register CostumStructureCreator instances")
        if name == 'dihedrals':
            creator.weight = self._dihedral_weight
        elif name == 'hessian':
            creator.weight = self._hessian_weight
        self.creators[name] = creator

    @classmethod
    def from_config(cls, config):
        creators = {}
        for name, clss in AdditionalStructureCreator.classes().items():
            _cls = clss.from_config(getattr(config, name))
            if _cls is not None:
                creators[name] = _cls
        return cls(creators, config.energy_element_weights,
                   config.gradient_element_weights, config.hessian_element_weights,
                   config.hessian_weight, config.dihedral_weight)

    @classmethod
    def _extend_user_input(cls, questions):
        for name, clss in AdditionalStructureCreator.classes().items():
            questions.generate_block(name, clss.colt_user_input)

    def run(self, qm):
        parent = qm.pathways.getdir("addstruct", create=True)

        for i, (name, creator) in enumerate(self.creators.items()):
            folder = parent / f'{creator.name}'
            os.makedirs(folder, exist_ok=True)
            creator.folder = folder

        # prestep
        for name, creator in self.creators.items():
            creator.setup_pre(qm)

        for name, creator in self.creators.items():
            cal = creator.check_pre()
            if cal is not None:
                qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                               'Creating the necessary input file and exiting...\nPlease run the '
                               'calculation and put the output files in the same directory.\n'
                               'Necessary output files and the corresponding extensions '
                               f"are:\n{cal.missing_as_string()}\n\n\n")

        for name, creator in self.creators.items():
            creator.parse_pre(qm)

        # mainstep
        for name, creator in self.creators.items():
            creator.setup_main(qm)

        for name, creator in self.creators.items():
            cal = creator.check_main()
            if cal is not None:
                qm.logger.exit(f"Required output file(s) not found in '{cal.folder}' .\n"
                               'Creating the necessary input file and exiting...\nPlease run the '
                               'calculation and put the output files in the same directory.\n'
                               'Necessary output files and the corresponding extensions '
                               f"are:\n{cal.missing_as_string()}\n\n\n")

        for name, creator in self.creators.items():
            creator.parse_main(qm)

    def _get_lowest_energy_and_coords(self):
        minimum = np.inf
        coords = None

        for creator in self.creators.values():
            for calctype in [creator.enouts(), creator.gradouts(), creator.hessouts()]:
                if calctype:
                    argmin = np.argmin((out.energy for out in calctype))
                    lowest = calctype[argmin].energy
                    if lowest < minimum:
                        minimum = lowest
                        coords = calctype[argmin].coords
        return minimum, coords

    def normalize(self):
        """set all energies to the minimum one"""
        emin, coords = self._get_lowest_energy_and_coords()
        self.subtract_energy(emin)
        return emin, coords

    def subtract_energy(self, energy):
        """subtract energy from all qm energies, forces and hessian are not affected"""
        for creator in self.creators.values():
            for out in creator.enouts():
                out.energy -= energy
            for out in creator.gradouts():
                out.energy -= energy
            for out in creator.hessouts():
                out.energy -= energy

    def enitr(self):
        for creator in self.creators.values():
            weight = creator.weight * self.energy_weight
            for out in creator.enouts():
                yield weight, out

    def graditr(self):
        for creator in self.creators.values():
            weight = creator.weight * self.gradient_weight
            for out in creator.gradouts():
                yield weight, out

    def hessitr(self):
        for creator in self.creators.values():
            weight = creator.weight * self.hessian_weight
            for out in creator.hessouts():
                yield weight, out


class HessianOutput(CostumStructureCreator):

    def __init__(self, weight, hessout):
        super().__init__(weight)
        if not isinstance(hessout, (tuple, list)):
            hessout = [hessout]
        self._hessout = hessout

    def enouts(self):
        return []

    def gradouts(self):
        return []

    def hessouts(self):
        return self._hessout

    def setup_main(self, qm):
        pass

    def check_main(self):
        pass

    def parse_main(self, qm):
        pass


class DihedralOutput(CostumStructureCreator):

    def __init__(self, weight, gradouts):
        super().__init__(weight)
        self._gradouts = gradouts

    def enouts(self):
        return []

    def gradouts(self):
        return self._gradouts

    def hessouts(self):
        return []

    def setup_main(self, qm):
        pass

    def check_main(self):
        pass

    def parse_main(self, qm):
        pass
