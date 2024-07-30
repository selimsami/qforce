from sympy import Symbol, Matrix, lambdify
from sympy.vector import CoordSys3D


N = CoordSys3D('N')


class SymbolicCalculator:


    def __init__(self, natoms, terms):
        self._coord_symbols = [[Symbol(f'r_{idx}{s}') for s in 'xyz']
                         for idx in range(natoms)]
        self._coords = [[sym*i for sym, i in zip(symbol, N)]
                        for symbol in self._coord_symbols]
                       

        self._energy, self._consts = self._get_energy_expression(terms)

        self.get_energy = lambdify([self._coord_symbols, list(self._consts.values())], self._energy)

    def _get_energy_expression(self, terms):
        consts = {}
        with terms.add_ignore(['dihedral/flexible', 'charge_flux']):
            for term in terms:
                consts[term.idx] = None
            consts = {i: Symbol(f'k_{i}') for i in consts}
            consts = {i: consts[i] for i in sorted(consts)}
            print(consts)

            energy = 0.0

            for i, term in enumerate(terms):
                energy += term.get_sympy_term(self._coords, consts[term.idx])


        return energy, consts
