import numpy as np


class OMatch:
    # Convert the input list to the five-tuple as prescribed in the paper
    def __init__(self, M, i, k, W):
        # (i, k, alpha, beta, w)
        self.M = [(i, k, m[0], m[-1], w) for m, w in zip(M, W)]
        # Sort the tuples according to non-decreasing order of beta
        self.M.sort(key=lambda x: x[3])
        # Enumerate the predecessor of a window/match
        self.p = [self.calc_p(i) for i in range(0, len(M) + 1)]

    # Find a max-weight sequence of compatible matches
    def max_weight_sequence(self):
        # OPT stores the optimal window selection at each window index
        M_len = len(self.M)
        OPT = np.empty((M_len + 1,), dtype=np.dtype(int))
        OPT[0] = 0

        # Find the max-weight matches
        for j in range(1, M_len + 1):
            if self.M[j - 1][4] + OPT[self.p[j]] > OPT[j - 1]:
                OPT[j] = self.M[j - 1][4] + OPT[self.p[j]]
            else:
                OPT[j] = OPT[j - 1]

        # Backtrack to find the selected interval
        Mw = []
        j = M_len
        while j > 0:
            if self.M[j - 1][4] + OPT[self.p[j]] > OPT[j - 1]:
                Mw.append(self.M[j - 1])
                j = self.p[j]
            else:
                j -= 1

        return Mw

    # Calculate the predecessor of a window/match
    def calc_p(self, i):
        if i == 0:
            return 0
        i_alpha = self.M[i - 1][2]
        i -= 1
        while i > 0:
            # if M[t][beta] < M[i][alpha]
            if self.M[i - 1][3] < i_alpha:
                return i
            i -= 1
        return 0
