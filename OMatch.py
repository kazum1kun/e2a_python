import numpy as np


class OMatch:
    # Convert the input list to the five-tuple as prescribed in the paper
    def __init__(self, M, i, k):
        # (i, k, alpha, beta)
        # the last entry w is OMITTED to save some space and make sure the data syncs
        self.M = [(i, k, m[1], m[-1]) for m in M]
        # Sort the tuples according to non-decreasing order of beta
        self.M.sort(key=lambda x: x[3])
        # Pseudo-entry to maintain 1-indexed array
        self.M.insert(0, (-1, -1, -1, -1))
        # Enumerate the predecessor of a window/match
        self.p = [self.calc_p(i) for i in range(0, len(M) + 1)]

    # Find a max-weight sequence of compatible matches
    def max_weight_sequence(self, w):
        # OPT stores the optimal window selection at each window index
        M_len = len(self.M) - 1
        OPT = np.full((M_len + 1,), -1, np.int_)
        OPT[0] = 0

        # Find the max-weight matches
        for j in range(1, M_len + 1):
            i = self.M[j][0]
            k = self.M[j][1]
            if w[i, k] + OPT[self.p[j]] > OPT[j - 1]:
                OPT[j] = w[i][k] + OPT[self.p[j]]
            else:
                OPT[j] = OPT[j - 1]

        # Backtrack to find the selected interval
        # Pseudo-element to maintain 1-based array
        Mw = [()]
        j = M_len
        while j > 0:
            i = self.M[j][0]
            k = self.M[j][1]
            if w[i][k] + OPT[self.p[j]] > OPT[j - 1]:
                Mw.append(self.M[j])
                j = self.p[j]
            else:
                j -= 1

        return Mw, self.p

    # Calculate the predecessor of a window/match
    def calc_p(self, i):
        if i == 0:
            return 0
        i_alpha = self.M[i][2]
        i -= 1
        while i > 0:
            # if M[t][beta] < M[i][alpha]
            if self.M[i][3] < i_alpha:
                return i
            i -= 1
        return 0
