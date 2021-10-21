import numpy as np


class SLN:
    def __init__(self, M, p, w, S, oMatch):
        self.M = M
        self.p = p
        self.w = w
        self.S = S
        self.oMatch = oMatch

    def sln_1d(self, i, k):
        x = 0
        x_opt = 0
        self.w[i][k] = x
        f_opt = self.f()
        p = self.p
        M = self.M
        done = False

        while not done:
            m = len(self.M)
            OPT = np.zeros((m,), np.int_)
            a = np.zeros((m,), np.int_)
            sigma = np.infty
            delta = np.zeros((m,), np.int_)

            for j in range(1, m + 1):
                # M[j].w is variable (by default theta entries are all zeros)
                if M[j][0] == i and M[j][1] == k:
                    delta[j] = 1
                # Pre-calc M[j].w for convenience
                j_weight = self.w[M[j][0], M[j][1]]

                if j_weight + OPT[p[j]] > OPT[j - 1]:
                    OPT[j] = j_weight + OPT[p[j]]
                    a[j] = a[p[j]] + delta[j]
                    if a[p[j] + delta[j] < a[j - 1]]:
                        sigma = np.min[sigma, (j_weight + OPT[p[j]] - OPT[j - 1]) /
                                       (a[j - 1] - a[self.p[j] - delta[j]])]
                elif j_weight + OPT[p[j]] < OPT[j - 1]:
                    OPT[j] = OPT[j - 1]
                    a[j] = a[j - 1]
                    if a[p[j] + delta[j] > a[j - 1]]:
                        sigma = np.min[sigma, (j_weight + OPT[p[j]] - OPT[j - 1]) /
                                       (a[j - 1] - a[p[j] - delta[j]])]
                else:
                    if a[p[j]] + delta[j] <= a[j - 1]:
                        OPT[j] = OPT[j - 1]
                        a[j] = a[j - 1]
                    else:
                        OPT[j] = j_weight + OPT[p[j]]
                        a[j] = a[p[j]] + delta[j]

            if sigma == np.infty:
                sigma = 2
                done = True

            mu = x + sigma
            nu = x + sigma / 2
            self.w[i, k] = nu
            f_new = self.f()
            if f_new < f_opt or (f_new == f_opt and nu == sigma / 2):
                x_opt = nu
                f_opt = f_new
            self.w[i, k] = mu
            f_new = self.f()
            if f_new < f_opt:
                x_opt = mu
                f_opt = f_new
            if not done:
                x = mu

        return x_opt, f_opt

    # Calculates the edit distance between two input sequences. This variant is also called
    # Levenshtein distance as insertions, deletions, and modifications on individual chars are
    # allowed (each with cost of 1)
    def calc_edit_distance(self, seq1, seq2):
        m = len(seq1)
        n = len(seq2)

        # dist[i,j] holds the Levenshtein distance between the first i chars of seq1 and first j chars of seq2
        dist = np.zeros((m, n), np.int_)

        # Prefixes of seq1 can be transformed to empty prefix of seq2 by removing all chars
        for i in range(1, m):
            dist[i][0] = i

        # Empty prefix of seq1 can be transformed to prefixes of seq2 by inserting every chars
        for j in range(1, n):
            dist[0][j] = j

        # Calculate the rest of the entries using DP
        for j in range(1, n):
            for i in range(1, m):
                if seq1[i] == seq2[j]:
                    sub_cost = 0
                else:
                    sub_cost = 1

                # Deletion, insertion, and substitution costs
                dist[i][j] = np.min([dist[i - 1][j] + 1,
                                     dist[i][j - 1] + 1,
                                     dist[i - 1][j - 1] + sub_cost])

        return dist[m - 1][n - 1]

    # Using given Mw, obtain a corresponding device event sequence and calculates the distance between the sequence and
    # given events
    def f(self):
        # Recalculate M using the updated weights
        self.M = self.oMatch.max_weight_sequence(self.w)
        m = len(self.M)
        Aw = np.full((m + 1,), -1, np.int_)
        # Assign Aw[j] to the corresponding i value
        for j in range(1, m + 1):
            Aw[j] = self.M[j][0]

        E1 = []
        for A in Aw:
            E1.extend(self.S[A][1])

        return self.calc_edit_distance(E1, self.S)
