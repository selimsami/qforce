import numpy as np
#
from .baseterms import TermBase
#
from ..forces import calc_pairs


class NonBondedTerms(TermBase):

    name = 'NonBondedTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_pairs(crd, self.atomids, self.equ, force)

    @classmethod
    def get_terms(cls, topo):
        """get terms"""

        non_bonded_terms = cls.get_terms_container()

        eps0 = 1389.35458

        for i, a1 in enumerate(topo.atoms):
            for j, a2 in enumerate(topo.atoms):
                if i < j and all([j not in topo.neighbors[c][i] for c in range(topo.n_excl)]):
                    sigma = 0.5 * (topo.sigma[a1] + topo.sigma[a2])
                    epsilon = (topo.epsilon[a1] * topo.epsilon[a2])**0.5
                    sigma6 = sigma**6
                    c6 = 4 * epsilon * sigma6
                    c12 = c6 * sigma6
                    qq = topo.q[a1] * topo.q[a2] * eps0
                    term_type = '-'.join([topo.types[i], topo.types[j]])
                    non_bonded_terms.append(cls([i, j], np.array([c6, c12, qq]), term_type))
        return non_bonded_terms
