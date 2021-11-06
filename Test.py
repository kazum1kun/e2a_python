from SLN import SLN
from utils.FileReader import *
from utils import Timer


def main():
    file_ext = '-reduced'
    activities = read_activities(f'data/activities/activities{file_ext}.txt')
    events = read_events(f'data/events/events{file_ext}.txt')
    mappings = read_mappings(f'data/mappings/mappings{file_ext}.txt')
    Timer.lap('Text read finished.')

    sln = SLN(mappings, events)
    w, Aw = sln.sln_nd(10)
    print(w)
    print(Aw)


def test_speed():
    import nltk
    import random
    import string

    for N in [100, 1000, 5000, 10000, 15000]:
        s1 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        s2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        Timer.lap(f'Start! N={N}')
        print(nltk.edit_distance(s1, s2))
        Timer.lap('nltk finished!')
        print(calc_edit_distance(s1, s2))
        Timer.lap('calcedit finished!')


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


def verify_correctness(actual, truth):
    pass


if __name__ == '__main__':
    main()
