import numpy as np
from ase.calculators.calculator import Calculator


class QForce(Calculator):

    implemented_properties = ('energy', 'forces')

    def __init__(self, terms, ignores, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.terms = terms
        self.ignores = ignores

    def calculate(self, atoms, properties, system_changes, *args, **kwargs):
        coords = atoms.get_positions()
        if system_changes:
            for name in self.implemented_properties:
                self.results.pop(name, None)

        if 'forces' not in self.results or 'energy' not in self.results:
            self.results = {'energy': 0.0, 'forces': np.zeros((len(atoms), 3))}

            with self.terms.add_ignore(self.ignores):
                for term in self.terms:
                    self.results['energy'] += term.do_force(coords, self.results['forces'])
