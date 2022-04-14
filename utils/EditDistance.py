from multiprocessing import Pool

import numpy as np
import os
import psutil


class EditDistance:
    def __init__(self, prob_match, actual):
        self.prob_match = prob_match
        self.actual = actual
        self.prob_match.insert(0, -1)
        # self.actual.insert(0, -1)
        self.n = len(self.prob_match) - 1
        self.m = len(self.actual) - 1

        self.c = np.zeros((self.n + 1, self.m + 1))

        # Perform Initialization
        for i in range(0, self.n + 1):
            self.c[i, 0] = i
        for j in range(0, self.m + 1):
            self.c[0, j] = j

    # Start = start coord of the diagonal (inclusive)
    # End = end coord of the diagonal (inclusive)
    # start > end
    # This function assumes start and end are valid (i.e. no oob numbers)
    def calc_segment(self, start, end, strict_mode):
        # Limit CPU usage by setting its priority to "below normal"
        # pid = psutil.Process(os.getpid())
        # pid.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

        i = start[0]
        j = start[1]

        while True:
            if (isinstance(self.prob_match[i], int) and self.prob_match[i] == self.actual[j]) or \
                    (isinstance(self.prob_match[i], list) and not strict_mode and
                     self.actual[j] in self.prob_match[i]):
                self.c[i, j] = self.c[i - 1, j - 1]
            else:
                self.c[i, j] = np.min((self.c[i - 1, j - 1], self.c[i, j - 1], self.c[i - 1, j])) + 1

            if i == end[0] and j == end[1]:
                break

            i -= 1
            j += 1

    # To prevent slowdown due to excessive overhead, we start multithreading once the diagonal grows larger than
    # a threshold.
    # The total number of threads is given by min(ceil(d_len/threshold), num_cores)
    def calc_ed(self, strict_mode, threshold=2000):
        num_cores = psutil.cpu_count()

        start = [1, 1]
        end = [1, 1]
        # pool = Pool()
        # Diags can be constructed by two moving points:
        # Starting from (1, 1), one travel down and one travel right. They follow corners. Eventually they will
        # meet at (n, m)
        while True:
            # d_len = start[0] - end[0] + 1

            # Diag length is short, no need for multiple threads
            # if d_len <= threshold:
            self.calc_segment(start, end, strict_mode)
            # else:
            #     num_threads = int(np.min((np.ceil(d_len/threshold), num_cores)))
            #
            #     avg_length = int(np.floor(d_len/num_threads))
            #     residual = d_len % num_threads
            #
            #     # Perform partition on the diag
            #     starts = []
            #     ends = []
            #     starts.append(start)
            #     ends.append([start[0] - (avg_length + residual) + 1,
            #                  start[1] + (avg_length + residual) - 1])
            #
            #     for i in range(1, num_threads):
            #         starts.append([ends[i-1][0] - 1, ends[i-1][1] + 1])
            #         ends.append([ends[i-1][0] - avg_length, ends[i-1][1] + avg_length])
            #
            #     # Start a pool of workers
            #     pool.imap_unordered(self.calc_segment, zip(starts, ends))
            # Check if we are done with the matrix
            if start[0] == end[0] and start[1] == end[1] and start[0] == self.n:
                break

            # Move on to next diagonal
            if start[0] < self.n:
                start[0] += 1
            else:
                start[1] += 1
            if end[1] < self.m:
                end[1] += 1
            else:
                end[0] += 1

        # # Wait for every process to finish
        # pool.close()
        # pool.join()

        return self.c[self.n, self.m]
