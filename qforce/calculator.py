import numpy as np
from ase.calculators.calculator import Calculator
from .forces import calc_imp_diheds


class QForce(Calculator):

    implemented_properties = ('energy', 'forces')

    def __init__(self, terms, dihedral_restraints=[], **kwargs):
        Calculator.__init__(self, **kwargs)
        self.terms = terms
        self.dihedral_restraints = dihedral_restraints

    def calculate(self, atoms, properties, system_changes, *args, **kwargs):
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
