import numpy as np
from OMatch import OMatch
from utils import Timer
import nltk


class SLN:
    def __init__(self, mappings, events):
        self.p = None
        self.n = len(mappings)
        # Calculate ni values and the N value
        ni = [len(mappings[activity]) - 1 for activity in range(1, self.n + 1)]
        self.N = np.sum(ni)
        k = np.max(ni)
        # Make array 1-based
        ni.insert(0, 0)
        self.ni = ni

        self.w = np.zeros((self.n + 1, k + 1), np.float_)
        self.S = mappings
        self.E = events[:, 1]
        self.oMatch = OMatch(self.E, self.S, self.n, self.ni)
        self.M = self.oMatch.M

        self.f_called = -1

    def sln_1d(self, i, k):
        x = 0
        x_opt = 0
        self.w[i][k] = x
        f_opt = self.f()
        A_old = self.get_aw()
        M = self.M
        done = False
        m = len(self.M)
        p = self.p
        itr_num = 1

        while not done:
            OPT = np.zeros((m,), np.int_)
            a = np.zeros((m,), np.int_)
            sigma = np.infty
            delta = np.zeros((m,), np.int_)
            Aw_diff = False

            for j in range(1, m):
                # M[j].w is variable (by default theta entries are all zeros)
                if M[j]['i'] == i and M[j]['k'] == k:
                    delta[j] = 1
                # Pre-store M[j].w for convenience
                Mj_w = self.w[M[j]['i']][M[j]['k']]

                if Mj_w + OPT[p[j]] > OPT[j - 1]:
                    OPT[j] = Mj_w + OPT[p[j]]
                    a[j] = a[p[j]] + delta[j]
                    if a[p[j]] + delta[j] < a[j - 1]:
                        sigma = np.min([sigma, (Mj_w + OPT[p[j]] - OPT[j - 1]) /
                                       (a[j - 1] - a[p[j]] - delta[j])])
                elif Mj_w + OPT[p[j]] < OPT[j - 1]:
                    OPT[j] = OPT[j - 1]
                    a[j] = a[j - 1]
                    if a[p[j]] + delta[j] > a[j - 1]:
                        sigma = np.min([sigma, (Mj_w + OPT[p[j]] - OPT[j - 1]) /
                                       (a[j - 1] - a[p[j]] - delta[j])])
                else:
                    if a[p[j]] + delta[j] <= a[j - 1]:
                        OPT[j] = OPT[j - 1]
                        a[j] = a[j - 1]
                    else:
                        OPT[j] = Mj_w + OPT[p[j]]
                        a[j] = a[p[j]] + delta[j]

            if sigma == np.infty:
                sigma = 2
                done = True

            mu = x + sigma
            nu = x + sigma / 2
            self.w[i][k] = nu
            A_new = self.get_aw()

            if not np.array_equal(A_new, A_old) or nu == sigma / 2:
                Aw_diff = True
                f_new = self.f()
                A_old = A_new
                if f_new < f_opt or (f_new == f_opt and nu == sigma / 2):
                    x_opt = nu
                    f_opt = f_new

            self.w[i][k] = mu
            A_new = self.get_aw()
            if not np.array_equal(A_new, A_old):
                Aw_diff = True
                f_new = self.f()
                A_old = A_new
                if f_new < f_opt:
                    x_opt = mu
                    f_opt = f_new

            if not done:
                x = mu

            print(f'Inner Iteration {itr_num}, {i=}, {k=}, {Aw_diff=}, {f_opt=}, {f_new=}')
            itr_num += 1

        return x_opt, f_opt

    def sln_nd(self, C):
        Timer.lap('ND SLN started')
        n = self.n
        ni = self.ni
        # Init the weights for S1s to C and the rest to 1
        for i in range(1, n + 1):
            self.w[i][1] = C
            for k in range(2, ni[i] + 1):
                self.w[i][k] = 1
        done = False
        f_opt = self.f()
        itr_num = 1

        # Coordinate descent
        while not done:
            # Weight normalization
            W = 0
            for i in range(1, n + 1):
                for k in range(1, ni[i] + 1):
                    W += self.w[i][k]

            for i in range(1, n + 1):
                for k in range(1, ni[i] + 1):
                    self.w[i][k] = self.N * self.w[i][k] / W

            done = True

            # 1D minimization
            for i in range(1, n + 1):
                for k in range(1, ni[i] + 1):
                    print(f'\n===========Iteration {itr_num}, optimizing SLN for {i=}, {k=}===========')
                    x_old = self.w[i][k]
                    x_new, f_new = self.sln_1d(i, k)
                    print(f'After optimizing, {f_new=}, {f_opt=}')
                    if f_new < f_opt:
                        done = False
                        self.w[i][k] = x_new
                        f_opt = f_new
                    else:
                        self.w[i][k] = x_old
                    Timer.lap(f'Done, f is called {self.f_called} times for {i=}, {k=}')
                    self.f_called = 0

                    print(self.w)
                    for j in range(1, len(self.w)):
                        if self.w[j][1] == 0:
                            print(f'w[{j}][1] value is 0, exiting')
                            return
            itr_num += 1
        return self.w, self.get_aw()

    # Calculates the edit distance between two input sequences. This variant is also called
    # Levenshtein distance as insertions, deletions, and modifications on individual chars are
    # allowed (each with cost of 1)
    @staticmethod
    def calc_edit_distance(seq1, seq2):
        m = len(seq1)
        n = len(seq2)
        if m == 0:
            return n

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

    # Calculate the distance between the sequence and given events
    def f(self):
        self.f_called += 1
        Aw = self.get_aw()
        E1 = []
        for A in Aw[1:]:
            E1.extend(self.S[A][1][1:])

        return nltk.edit_distance(E1, self.E[1:])

    # Using given Mw, obtain a corresponding device event sequence
    def get_aw(self):
        # Recalculate M using the updated weights
        Mw, self.p = self.oMatch.max_weight_sequence(self.w)
        m = len(Mw)
        Aw = np.full((m,), -1, np.int_)

        # Assign Aw[j] to the corresponding i value
        for j in range(1, m):
            Aw[j] = Mw[j]['i']

        return Aw
