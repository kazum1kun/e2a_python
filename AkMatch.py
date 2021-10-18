import numpy as np


class AkMatch:
    def __init__(self):
        self.seq = {1: [[], ['', 'a', 'b']],
                    2: [[], ['', 'a', 'b', 'c'], ['', 'a', 'b']]}
        self.E = ['', 'a', 'b', 'a', 'b', 'd', 'c', 'a', 'b', 'c']

    # Find the LCS between two input event sequences
    def find_matches(self, E, A, k):
        # Initialization
        eta = len(self.seq[A][k]) - 1
        L = []
        l = 0
        start = 1
        i = 1
        m = len(E) - 1
        c = np.full((m + 1, eta + 1), -1, dtype=np.dtype(int))
        flag = True

        while flag:
            for j in range(0, eta + 1):
                c[start - 1][j] = 0
            curr_seq = self.seq[A][k]

            # Find the LCS of E and the given sequence of events
            while i <= m:
                if i + 1 > m:
                    flag = False

                c[i][0] = 0
                # Compare events one by one to build the LCS
                for j in range(1, eta + 1):
                    # Found an event match
                    if E[i] == curr_seq[j]:
                        c[i][j] = c[i-1][j-1] + 1
                    # Otherwise, go with the highest neighbor
                    elif c[i-1][j] >= c[i][j-1]:
                        c[i][j] = c[i-1][j]
                    else:
                        c[i][j] = c[i][j-1]

                if c[i][eta] == eta:
                    # Found a full match, backtrack to find the solution
                    l += 1
                    row = i
                    col = eta
                    psi_i = np.full((col + 1,), -1, dtype=np.dtype(int))

                    while col > 0:
                        if E[row] == curr_seq[col]:
                            psi_i[col] = row
                            row -= 1
                            col -= 1
                        elif c[row - 1][col] >= c[row][col-1]:
                            row -= 1
                        else:
                            col -= 1
                    L.append(psi_i)
                    start = psi_i[1] + 1
                    i = start
                    flag = True
                    break
                i += 1
        return L
