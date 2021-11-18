import logging as log

import editdistance
import numpy as np
from OMatch import OMatch


class SLN:
    def __init__(self, mappings, events, ni):
        self.p = None
        self.n = len(mappings)
        # Calculate ni values and the N value
        self.N = np.sum(ni)
        k = np.max(ni)
        # Make array 1-based
        self.ni = ni

        self.w = np.zeros((self.n + 1, k + 1), np.float_)
        self.S = mappings
        self.E = events
        self.oMatch = OMatch(self.E, self.S, self.n, self.ni)
        self.M = self.oMatch.M

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
        itr_num = 0

        while not done:
            itr_num += 1
            OPT = np.zeros((m,), np.float_)
            a = np.zeros((m,), np.int_)
            sigma = np.infty
            delta = np.zeros((m,), np.int_)

            for j in range(1, m):
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

            if not np.array_equal(A_new[1:], A_old[1:]) or nu == sigma / 2:
                f_new = self.f()
                A_old = A_new
                if f_new < f_opt or (f_new == f_opt and nu == sigma / 2):
                    x_opt = nu
                    f_opt = f_new

            self.w[i][k] = mu
            A_new = self.get_aw()
            if not np.array_equal(A_new[1:], A_old[1:]):
                f_new = self.f()
                A_old = A_new
                if f_new < f_opt:
                    x_opt = mu
                    f_opt = f_new

            if not done:
                x = mu

        return x_opt, f_opt

    def sln_nd(self, C):
        # Timer.lap('ND SLN started')
        n = self.n
        ni = self.ni
        # Init the weights for S1s to C and the rest to 1
        for i in range(1, n + 1):
            self.w[i][1] = C
            for k in range(2, ni[i] + 1):
                self.w[i][k] = 1
        done = False
        f_opt = self.f()
        itr_num = 0

        # Coordinate descent
        while not done:
            # Weight normalization
            W = np.sum(self.w)
            self.w = self.N * self.w / W

            done = True
            itr_num += 1

            # 1D minimization
            for i in range(1, n + 1):
                for k in range(1, ni[i] + 1):
                    x_old = self.w[i][k]
                    x_new, f_new = self.sln_1d(i, k)
                    if f_new < f_opt:
                        self.w[i][k] = x_new
                        f_opt = f_new
                        if f_opt > 0:
                            done = False
                    else:
                        self.w[i][k] = x_old

                    for j in range(1, len(self.w)):
                        if self.w[j][1] == 0:
                            log.warning(f'w[{j}][1] value is 0')

                    if f_opt == 0:
                        return self.w, self.get_aw(), f_opt
        return self.w, self.get_aw(), f_opt

    # Calculate the distance between the sequence and given events
    def f(self):
        Aw = self.get_aw()
        E_calc = []
        for A in Aw[1:]:
            E_calc.extend(self.S[A[0]][A[1]][1:])

        return editdistance.eval(E_calc, self.E[1:])

    # Using given Mw, obtain a corresponding device event sequence
    def get_aw(self):
        # Recalculate M using the updated weights
        Mw, self.p = self.oMatch.max_weight_sequence(self.w)
        self.Mw = Mw

        return Mw
