import numpy as np


class WindowMatch:
    def __init__(self, E, S):
        self.event_time = E[:, 0]
        self.event_seq = E[:, 1]
        self.S = S

    # Find the LCS between two input event sequences
    def find_matches(self, A, theta_d, theta_c):
        activity_seq = self.S[A][1]
        n = len(activity_seq) - 1
        m = len(self.event_seq) - 1
        M = set()

        # Init the solution matrix
        c = np.full((m + 1, n + 1), 0, np.int_)

        i = 1
        l = 0

        # Define the window
        w_s = 1
        w_e = 1

        while w_e <= m:
            # Find the biggest window that satisfy the time constraint
            while w_e + 1 <= m and self.event_time[w_e + 1] - self.event_time[w_s] <= theta_d:
                w_e += 1
            c[:] = 0
            # Iterate through events
            while i <= w_e:
                for j in range(n + 1):
                    # Found an event match
                    if self.event_seq[i] == activity_seq[j]:
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
                    if self.event_seq[row] == activity_seq[col]:
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

            # Slide the window forward
            w_s += 1
            i = w_s
            w_e = w_s

        # Covert the set back to a list and sort it according to its first event
        M_list = list(M)
        M_list = sorted(M_list)
        return M_list
