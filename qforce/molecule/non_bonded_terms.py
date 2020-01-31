import numpy as np
import pulp
#
from .baseterms import TermBase
#
from ..forces import get_dist, calc_pairs


class NonBondedTerms(TermBase):

    name = 'NonBondedTerm'

    def _calc_forces(self, crd, force, fconst):
        calc_pairs(crd, self.atomids, self.equ, fconst, force)

    @classmethod
    def get_terms(cls, topo):
        """get terms"""

        non_bonded_terms = cls.get_terms_container()

        average_equivalent_terms([qm.q, qm.c6, qm.c12], topo)
        sum_charges_to_qtotal(topo, qm)





        return non_bonded_terms


def average_equivalent_terms(terms, eq):
    avg_terms = []
    for term in terms:
        avg_term = []
        term = np.array(term)
        for l in eq:
            total = 0
            for a in l:
                total += term[a]
            avg_term.append(round(total/len(l), 5))
        avg_terms.append(avg_term)
    return np.array(avg_terms)


def sum_charges_to_qtotal(topo, qm):
    q_total = qm.q.sum().round()
    total = sum([round(qm.q[i], 5)*len(l) for i, l in enumerate(topo.list)])
    extra = int(100000 * round(total - q_total, 5))
    if extra != 0:
        if extra > 0:
            sign = 1
        else:
            sign = -1
            extra = - extra

        n_eq = [len(l) for l in topo.list]
        no = [f"{i:05d}" for i, _ in enumerate(n_eq)]

        var = pulp.LpVariable.dicts("x", no, lowBound=0, cat='Integer')
        prob = pulp.LpProblem('prob', pulp.LpMinimize)
        prob += pulp.lpSum([var[n] for n in no])
        prob += pulp.lpSum([eq * var[no[i]] for i, eq
                            in enumerate(n_eq)]) == extra
        prob.solve()

        if prob.status == 1:
            for i, v in enumerate(prob.variables()):
                qm.q[i] -= sign * v.varValue / 100000
        else:
            print('Failed to equate total of charges to the total charge of '
                  'the system. Do so manually')


def calc_pair_list(mol, qm, nrexcl):
    eps0 = 1389.35458
    qm.sigma = (qm.c12/qm.c6)**(1/6)
    qm.epsilon = qm.c6 / (4*qm.sigma**6)

    for i, a1 in enumerate(mol.atoms):
        for j, a2 in enumerate(mol.atoms):
            if i < j and all([j not in mol.neighbors[c][i]
                              for c in range(nrexcl)]):
                sigma = 0.5 * (qm.sigma[a1] + qm.sigma[a2])
                epsilon = (qm.epsilon[a1] * qm.epsilon[a2])**0.5
                sigma6 = sigma**6
                c6 = 4 * epsilon * sigma6
                c12 = c6 * sigma6
                qq = qm.q[a1] * qm.q[a2] * eps0
                mol.pair_list.append([i, j, c6, c12, qq])