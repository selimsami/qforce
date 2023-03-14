import numpy as np
from ase.units import _eps0, kJ, mol, J, m

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

        inv_eps0 = 1/(4*np.pi*_eps0) * m / J / kJ * mol  # from F m-1 to kJ mol−1 Angstrom e−2

        for i in range(topo.n_atoms):
            for j in range(i+1, topo.n_atoms):
                close_neighbor = any([j in topo.neighbors[c][i] for c in range(non_bonded.n_excl)])

                if not close_neighbor and (i, j) not in non_bonded.exclusions:
                    pair_name = tuple(sorted([non_bonded.lj_types[i], non_bonded.lj_types[j]]))
                    term_type = '-'.join(pair_name)

                    if (i, j) in non_bonded.pairs:
                        if pair_name in non_bonded.lj_1_4.keys():
                            param = non_bonded.lj_1_4[pair_name][:]
                        else:
                            param = [p*non_bonded.fudge_lj for p in non_bonded.lj_pairs[pair_name]]
                        param.append(non_bonded.q[i]*non_bonded.q[j]*inv_eps0*non_bonded.fudge_q)

                    else:
                        param = non_bonded.lj_pairs[pair_name][:]
                        param.append(non_bonded.q[i]*non_bonded.q[j]*inv_eps0)

                    non_bonded_terms.append(cls([i, j], np.array(param), term_type))
        return non_bonded_terms
