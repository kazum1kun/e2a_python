import numpy as np


class SLN:
    def __init__(self, M, p, w):
        self.M = M
        self.p = p
        self.w = w

    def sln_1d(self, i, k):
        x = 0
        x_opt = 0
        self.w[i][k] = x
        f_opt = 0


    # Calculates the edit distance between two input sequences. This variant is also called
    # Levenshtein distance as insertions, deletions, and modifications on individual chars are
    # allowed (each with cost of 1)
    def calc_edit_distance(self, seq1, seq2):
        m = len(seq1)
        n = len(seq2)

        # dist[i,j] holds the Levenshtein distance between the first i chars of seq1 and first j chars of seq2
        dist = np.zeros((m, n), np.dtype(int))

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

    # Using given Mw, obtain a corresponding device event sequence
    def obtain_sequence(self):
        m = len(self.M)
        Aw = np.full((m + 1,), -1, dtype=np.dtype(int))
        # Assign Aw[j] to the corresponding i value
        for j in range(1, m + 1):
            Aw[j] = self.M[j][0]

        