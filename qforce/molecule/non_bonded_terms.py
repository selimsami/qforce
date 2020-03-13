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
        for i in range(topo.n_atoms):
            for j in range(i+1, topo.n_atoms):
                close_neighbor = any([j in topo.neighbors[c][i] for c in range(topo.n_excl)])
                if not close_neighbor and (i, j) not in topo.exclusions:
                    pair_name = tuple(sorted([topo.lj_types[i], topo.lj_types[j]]))
                    params = topo.lj_pairs[pair_name][:]
                    term_type = '-'.join(pair_name)
                    params.append(topo.q[i] * topo.q[j] * eps0)
                    non_bonded_terms.append(cls([i, j], np.array(params), term_type))
        return non_bonded_terms
