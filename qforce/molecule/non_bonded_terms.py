import numpy as np
#
from .baseterms import TermBase
from ..forces import calc_pairs


class NonBondedTerms(TermBase):

    name = 'NonBondedTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_pairs(crd, self.atomids, self.equ, force)

    @classmethod
    def get_terms(cls, topo, non_bonded):
        """get terms"""

        non_bonded_terms = cls.get_terms_container()

        eps0 = 1389.35458
        for i in range(topo.n_atoms):
            for j in range(i+1, topo.n_atoms):
                close_neighbor = any([j in topo.neighbors[c][i] for c in range(non_bonded.n_excl)])

                if not close_neighbor and (i, j) not in non_bonded.exclusions:
                    pair_name = tuple(sorted([non_bonded.lj_types[i], non_bonded.lj_types[j]]))
                    term_type = '-'.join(pair_name)

                    if ((non_bonded.n_excl == 2 and j in topo.neighbors[2][i]) or  # 1-4 inter.
                            (i, j) in non_bonded.pairs):  # extra pair interactions
                        if pair_name in non_bonded.lj_1_4.keys():
                            param = non_bonded.lj_1_4[pair_name][:]
                        else:
                            param = [p*non_bonded.fudge_lj for p in non_bonded.lj_pairs[pair_name]]
                        param.append(non_bonded.q[i]*non_bonded.q[j]*eps0*non_bonded.fudge_q)
                    else:
                        param = non_bonded.lj_pairs[pair_name][:]
                        param.append(non_bonded.q[i]*non_bonded.q[j]*eps0)
                    non_bonded_terms.append(cls([i, j], np.array(param), term_type))
        return non_bonded_terms
