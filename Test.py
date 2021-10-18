from AkMatch import AkMatch
from OMatch import OMatch


def main():
    E = ['', 'a', 'b', 'a', 'b', 'd', 'c', 'a', 'b', 'c']
    akMatch = AkMatch()
    L = akMatch.find_matches(E, 2, 1)
    print(L)

    L = [(1, 5), (2, 7), (6, 8), (3, 11), (9, 12), (10, 13)]

    oMatch = OMatch(L, 2, 1, [2, 4, 4, 7, 2, 1])
    Mw = oMatch.max_weight_sequence()
    print(Mw)


if __name__ == '__main__':
    main()
