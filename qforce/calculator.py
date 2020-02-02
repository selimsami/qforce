from ase.calculators.calculator import Calculator


class QForce(Calculator):

    implemented_properties = ('energy', 'forces')
    
    def __init__(self, terms, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.terms = terms
        
    def calculate(self, atoms, properties, system_changes, *args, **kwargs):
        coords = atoms.get_positions()
        if system_changes:
            for name in self.implemented_properties:
                self.results.pop(name, None)
        if 'energy' not in self.results:
            self.results = {'energy': 0.0}
        if 'forces' not in self.results:
            self.results['forces'] = np.zeros(len(atoms)* 3)
            for term in terms:
                term.do_force(coords, self.results['forces']
