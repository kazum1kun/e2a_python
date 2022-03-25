import numpy as np


class WindowMatch:
    def __init__(self, E, S):
        self.E = E
        self.S = S

    # Find the LCS between two input event sequences
    def find_matches(self, A, theta_d, theta_c):
        curr_seq = self.S[A][1]
        n = len(curr_seq) - 1
        m = len(self.E) - 1
        M = set()

        # Init the solution matrix
        c = np.full((m + 1, n + 1), 0, np.int_)

        i = 1
        l = 0

        # Define the window
        w_s = 1
        w_e = theta_d

        while w_e <= m:
            c[:] = 0
            # Iterate through events
            while i <= w_e:
                for j in range(n + 1):
                    # Found an event match
                    if self.E[i] == curr_seq[j]:
                        c[i, j] = c[i - 1, j - 1] + 1
                    # Otherwise, use the highest neighbor
                    else:
                        c[i, j] = np.max((c[i - 1, j], c[i, j - 1]))

                i += 1

            # Check if the match meets the confidence threshold
            if c[w_e, n] / n >= theta_c:
                l += 1
                row = w_e
                col = n
                m_l = []

                # Backtrack to reconstruct the match
                while col > 0:
                    if self.E[row] == curr_seq[col]:
                        m_l.insert(0, row)
                        row -= 1
                        col -= 1
                    elif c[row - 1, col] > c[row, col - 1]:
                        row -= 1
                    else:
                        col -= 1

                    if row < w_s:
                        col -= 1
                M.add(tuple(m_l))

            w_s += 1
            i = w_s
            w_e += 1

        # Covert the set back to a list and sort it according to its first event
        M_list = list(M)
        M_list = sorted(M_list)
        return M_list
