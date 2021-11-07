import numpy as np

from AkMatch import AkMatch
from utils import Timer


class OMatch:
    # Convert the input list to the five-tuple as prescribed in the paper
    def __init__(self, E, S, n, ni):
        akMatch = AkMatch(E, S)
        values = self.populate_m(akMatch, n, ni)
        M_dt = [('i', np.int_), ('k', np.int_), ('alpha', np.int_), ('beta', np.int_)]
        self.M = np.array(values, dtype=M_dt)
        self.M.sort(order='beta')
        # Enumerate the predecessor of a window/match
        self.p = [self.calc_p(i) for i in range(0, len(self.M))]
        Timer.lap('OMatch initialization done.')

    # Populates the matches for all i, k values
    @staticmethod
    def populate_m(akMatch, n, ni):
        # Pseudo-entry to maintain 1-indexed array
        all_values = [(-1, -1, -1, -1)]
        for i in range(1, n + 1):
            for k in range(1, ni[i] + 1):
                L = akMatch.find_matches(i, k)
                # (i, k, alpha, beta)
                # the last entry w is OMITTED to save some space and make sure the data syncs
                values = [(i, k, l[1], l[-1]) for l in L]
                # Sort the tuples according to non-decreasing order of beta
                all_values.extend(values)
        return all_values

    # Find a max-weight sequence of compatible matches
    def max_weight_sequence(self, w):
        # OPT stores the optimal window selection at each window index
        M_len = len(self.M)
        OPT = np.full((M_len,), -1, np.int_)
        OPT[0] = 0

        # Find the max-weight matches
        for j in range(1, M_len):
            i = self.M[j]['i']
            k = self.M[j]['k']
            if w[i][k] + OPT[self.p[j]] > OPT[j - 1]:
                OPT[j] = w[i][k] + OPT[self.p[j]]
            else:
                OPT[j] = OPT[j - 1]

        # Backtrack to find the selected interval
        Mw = []
        j = M_len - 1
        while j > 0:
            i = self.M[j]['i']
            k = self.M[j]['k']
            if w[i][k] + OPT[self.p[j]] > OPT[j - 1]:
                Mw.append(self.M[j])
                j = self.p[j]
            else:
                j -= 1

        Mw.reverse()
        # Pseudo-element to maintain 1-based array
        Mw.insert(0, ())

        return Mw, self.p

    # Calculate the predecessor of a window/match
    def calc_p(self, i):
        if i == 0:
            return 0
        i_alpha = self.M[i]['alpha']
        i -= 1
        while i > 0:
            # if M[t][beta] < M[i][alpha]
            if self.M[i]['beta'] < i_alpha:
                return i
            i -= 1
        return 0
