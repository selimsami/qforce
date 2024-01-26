import numpy as np
#
from ase.calculators.calculator import Calculator
from ase.units import Hartree, mol, kJ, Bohr
#
from .forces import calc_imp_diheds


nm2Bohr = 10.0/Bohr


class QForce(Calculator):

    implemented_properties = ('energy', 'forces')

    def __init__(self, terms, dihedral_restraints=None, atomicunits=False, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.terms = terms
        if dihedral_restraints is None:
            dihedral_restraints = []
        self.dihedral_restraints = dihedral_restraints
        self.atomicunits = atomicunits

    def calculate(self, atoms, properties, system_changes, *args, **kwargs):
        #
        coords = atoms.get_positions()
        if system_changes:
            for name in self.implemented_properties:
                self.results.pop(name, None)

        if 'forces' not in self.results or 'energy' not in self.results:
            self.results = {'energy': 0.0, 'forces': np.zeros((len(atoms), 3))}

            for term in self.terms:
                self.results['energy'] += term.do_force(coords, self.results['forces'])

            for atoms, phi0 in self.dihedral_restraints:
                calc_imp_diheds(coords, atoms, phi0, 10000, self.results['forces'])

            if self.atomicunits is True:
                self.results['forces'] *= ((kJ/(mol*Hartree))/nm2Bohr)
                self.results['energy'] *= kJ/(mol*Hartree)
