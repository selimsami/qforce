import os
#
import numpy as np
#
from .creator import CustomStructureCreator
from .xtbmd import XTBMolecularDynamics
from .additionalstructures import StructuresFromFile


class Computations_(CustomStructureCreator):

    _user_input = """
    energy_element_weights = 15000 :: float
    gradient_element_weights = 100 :: float
    hessian_element_weights = 0.1 :: float

    hessian_weight = 1 :: float
    dihedral_weight = 1 :: float
    """

    classes = {
            'xtbmd': XTBMolecularDynamics,
            'fromfile': StructuresFromFile,
    }

    def __init__(self, folder, energy_ele_weight, gradient_ele_weight, hessian_ele_weight,
                 hessian_weight, dihedral_weight, activatable=None):
        self.folder = folder
        self.creators = {}
        self.energy_weight = energy_ele_weight
        self.gradient_weight = gradient_ele_weight
        self.hessian_weight = hessian_ele_weight
        #
        self._hessian_weight = hessian_weight
        self._dihedral_weight = dihedral_weight
        # classes that can be activated later on
        if activatable is None:
            activatable = {}
        self._activatable = activatable

    @classmethod
    def from_config(cls, config, folder):
        activatable = {}
        for name, cls_ in cls.classes.items():
            settings = getattr(config, name)
            activatable[name] = (cls_, settings)
        return cls(folder, config.energy_element_weights,
                   config.gradient_element_weights, config.hessian_element_weights,
                   config.hessian_weight, config.dihedral_weight, activatable=activatable)

    @classmethod
    def _extend_user_input(cls, questions):
        for name, cls_ in cls.classes.items():
            questions.generate_block(name, cls_.colt_user_input)

    def activate(self, name, *args, **kwargs):
        cls, settings = self._activatable.get(name, (None, None))
        if cls is None:
            raise ValueError(f"Class '{name}' is not activatable")
        folder = self.folder / f'{name}'
        creator = cls.from_config(settings, folder, *args, **kwargs)
        if creator is not None:
            self.register(name, creator)

    def add_hessians(self, hessout):
        """add hessian"""
        self.creators['hessian'] = HessianOutput(self._hessian_weight, hessout)

    def add_dihedrals(self, scans):
        """add computed dihedrals"""
        self.creators['dihedrals'] = DihedralOutput(self._dihedral_weight, scans)

    def register(self, name, creator):
        """add a new creator"""
        if not isinstance(creator, CustomStructureCreator):
            raise ValueError(f"Can not register '{type(creator)}' "
                             "only register CustomStructureCreator instances")
        if name == 'dihedrals':
            creator.weight = self._dihedral_weight
        elif name == 'hessian':
            creator.weight = self._hessian_weight
        self.creators[name] = creator

    def setup_pre(self, qm):
        #
        for i, (name, creator) in enumerate(self.creators.items()):
            folder = self.folder / f'{name}'
            os.makedirs(folder, exist_ok=True)
            creator.folder = folder

        for _, creator in self.creators.items():
            creator.setup_pre(qm)

    def check_pre(self):
        for _, creator in self.creators.items():
            cal = creator.check_pre()
            if cal is not None:
                return cal
        return None

    def parse_pre(self, qm):
        for _, creator in self.creators.items():
            creator.parse_pre(qm)

    def setup_main(self, qm):
        for _, creator in self.creators.items():
            creator.setup_main(qm)

    def check_main(self):
        for _, creator in self.creators.items():
            cal = creator.check_main()
            if cal is not None:
                return cal
        return None

    def parse_main(self, qm):
        for _, creator in self.creators.items():
            creator.parse_main(qm)

    def setup_post(self, qm):
        for _, creator in self.creators.items():
            creator.setup_post(qm)

    def check_post(self):
        for _, creator in self.creators.items():
            cal = creator.check_post()
            if cal is not None:
                return cal
        return None

    def parse_post(self, qm):
        for _, creator in self.creators.items():
            creator.parse_post(qm)

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


class HessianOutput(CustomStructureCreator):

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


class DihedralOutput(CustomStructureCreator):

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


def Computations(config, folder):
    return Computations_.from_config(config, folder)
