import logging as log

import nltk
import numpy as np

from OMatch import OMatch
from utils import Timer


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

        log.debug(f'1D SLN starting for {k=}, {i=}')
        log.debug(f'Line 1:\tcurrent {f_opt=}\n'
                  f'\t\tAw={A_old}\n')
        log.debug('Current M value is: \n')
        log.debug(f'{M}\n')
        log.debug('Calculated Mw is:\n')
        log.debug(f'{self.Mw}')
        log.debug('Current w values are:\n'
                  f'{self.w}')
        log.debug(f'Current p values are: {p}')

        while not done:
            log.debug(f'--------Inner iteration {itr_num} starting, {i=}, {k=}--------')
            OPT = np.zeros((m,), np.float_)
            a = np.zeros((m,), np.int_)
            sigma = np.infty
            delta = np.zeros((m,), np.int_)
            Aw_diff = False

            for j in range(1, m):
                log.debug(f'Line 3: {j=}, {M[j]=}')
                # M[j].w is variable (by default theta entries are all zeros)
                if M[j]['i'] == i and M[j]['k'] == k:
                    log.debug(f'Line 4: True, delta[{j}]=1')
                    delta[j] = 1
                else:
                    log.debug(f'Line 4: False, delta[{j}]=0')
                # Pre-store M[j].w for convenience
                Mj_w = self.w[M[j]['i']][M[j]['k']]

                if Mj_w + OPT[p[j]] > OPT[j - 1]:
                    OPT[j] = Mj_w + OPT[p[j]]
                    a[j] = a[p[j]] + delta[j]
                    log.debug(f'Line 8: True, new OPT[{j}] is {OPT[j]}, new a[{j}] is {a[j]}')
                    if a[p[j]] + delta[j] < a[j - 1]:
                        log.debug(
                            f'{sigma=}, M[j].w={Mj_w}, {OPT[p[j]]=}, {OPT[j-1]=}, {a[j-1]=}, {a[p[j]]=}, {delta[j]=}, {p[j]=}')
                        sigma = np.min([sigma, (Mj_w + OPT[p[j]] - OPT[j - 1]) /
                                        (a[j - 1] - a[p[j]] - delta[j])])
                        log.debug(f'Line 11: sigma is updated, new value is {sigma :.3f}\n')
                    else:
                        log.debug(f'Line 11: sigma not updated\n')
                elif Mj_w + OPT[p[j]] < OPT[j - 1]:
                    OPT[j] = OPT[j - 1]
                    a[j] = a[j - 1]
                    log.debug(f'Line 13: True, new OPT[{j}] is {OPT[j]}, new a[{j}] is {a[j]}')
                    if a[p[j]] + delta[j] > a[j - 1]:
                        log.debug(
                            f'{sigma=}, M[j].w={Mj_w}, {OPT[p[j]]=}, {OPT[j-1]=}, {a[j-1]=}, {a[p[j]]=}, {delta[j]=}, {p[j]=}')
                        sigma = np.min([sigma, (Mj_w + OPT[p[j]] - OPT[j - 1]) /
                                        (a[j - 1] - a[p[j]] - delta[j])])
                        log.debug(f'Line 16: sigma is updated, new value is {sigma :.3f}\n')
                    else:
                        log.debug(f'Line 16: sigma not updated\n')
                else:
                    if a[p[j]] + delta[j] <= a[j - 1]:
                        OPT[j] = OPT[j - 1]
                        a[j] = a[j - 1]
                        log.debug(f'Line 19: True, new OPT[{j}] is {OPT[j]}, new a[{j}] is {a[j]}\n')
                    else:
                        OPT[j] = Mj_w + OPT[p[j]]
                        a[j] = a[p[j]] + delta[j]
                        log.debug(f'Line 22: True, new OPT[{j}] is {OPT[j]}, new a[{j}] is {a[j]}\n')

            if sigma == np.infty:
                sigma = 2
                done = True
            mu = x + sigma
            nu = x + sigma / 2
            log.debug(f'Line 25: {sigma=:.3f}')
            log.debug(f'Line 26: {mu=:.3f}, {nu=:.3f}')

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

            log.info(f'Inner Iteration {itr_num: <3} finished, {i=: <2}, {k=}, '
                     f'Aw_diff={Aw_diff.__repr__(): <5}, {f_opt=: <3}, {f_new=: <3}')
            log.debug(f'\nAw_new={A_new}\n')

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

        log.info(f'\nND SLN Init finished, current optimal value is {f_opt}\n'
                 'w value is')
        log.info(self.w)

        # Coordinate descent
        while not done:
            # Weight normalization
            W = np.sum(self.w)
            self.w = self.N * self.w / W

            done = True

            # 1D minimization
            for i in range(1, n + 1):
                for k in range(1, ni[i] + 1):
                    log.info(f'\n============Iteration {itr_num}, optimizing SLN for {i=}, {k=}============')
                    x_old = self.w[i][k]
                    x_new, f_new = self.sln_1d(i, k)
                    log.info(f'Iteration {itr_num} done. After optimizing, {f_new=}, {f_opt=}\n')
                    if f_new < f_opt:
                        done = False
                        self.w[i][k] = x_new
                        f_opt = f_new
                    else:
                        self.w[i][k] = x_old
                    Timer.lap(f'f is called {self.f_called} times for {i=}, {k=}')
                    self.f_called = 0

                    log.info('\nThe w value after this iteration is')
                    log.info(self.w)
                    for j in range(1, len(self.w)):
                        if self.w[j][1] == 0:
                            log.warning(f'w[{j}][1] value is 0')
            itr_num += 1
        return self.w, self.get_aw(), f_opt

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
        self.Mw = Mw
        Aw = np.full((m,), -1, np.int_)

        # Assign Aw[j] to the corresponding i value
        for j in range(1, m):
            Aw[j] = Mw[j]['i']

        return Aw
