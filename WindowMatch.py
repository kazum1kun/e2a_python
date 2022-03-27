import numpy as np


class WindowMatch:
    def __init__(self, E, S):
        self.event_time = E[:, 0]
        self.event_seq = E[:, 1]
        self.S = S

    # Find the LCS between two input event sequences
    def find_matches(self, A, k, theta_d):
        activity_seq = self.S[A][k]
        n = len(activity_seq) - 1
        m = len(self.event_seq) - 1
        M = set()

        # Init the solution matrix
        c = np.full((m + 1, n + 1), 0, np.int_)

        # Starting point of the next match
        w_s = 1

        while w_s <= m:
            # Search for the next suitable start point
            while w_s + 1 <= m and self.event_seq[w_s] != activity_seq[1]:
                w_s += 1
            i = w_s
            end_time = self.event_time[w_s] + theta_d
            last_match = 0
            c[:] = 0
            # Iterate through events till we exceed time allowed
            while i <= m and self.event_time[i] <= end_time:
                for j in range(1, n + 1):
                    # Found an event match
                    if self.event_seq[i] == activity_seq[j]:
                        c[i, j] = c[i - 1, j - 1] + 1
                        # Keep track of the last occurrence of the last event in the activity sequence
                        if j == n:
                            last_match = i
                    # Otherwise, use the highest neighbor
                    else:
                        c[i, j] = np.max((c[i - 1, j], c[i, j - 1]))

                i += 1

            # Check if the match meets the confidence threshold
            if c[i - 1, n] == n:
                row = i - 1
                col = n
                m_l = []

                # Backtrack to reconstruct the match
                while col > 0:
                    # Special treatment for the last item in the match
                    if col == n:
                        m_l.insert(0, last_match)
                        col -= 1
                        row -= 1
                        continue
                    # Keep going up the rows till number starts to decrease
                    while c[row - 1, col] == c[row, col]:
                        row -= 1
                    m_l.insert(0, row)
                    row -= 1
                    col -= 1
                    # if self.event_seq[row] == activity_seq[col]:
                    #     m_l.insert(0, row)
                    #     row -= 1
                    #     col -= 1
                    # elif c[row - 1, col] >= c[row, col - 1]:
                    #     row -= 1
                    # else:
                    #     col -= 1
                M.add(tuple(m_l))

            # Push the window forward
            w_s += 1

        # Covert the set back to a list and sort it according to its first event
        M_list = list(M)
        M_list = sorted(M_list)
        return M_list
