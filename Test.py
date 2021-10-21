from AkMatch import AkMatch
from OMatch import OMatch
from SLN import SLN
from FileReader import *


def main():
    E = ['', 'a', 'b', 'a', 'b', 'd', 'c', 'a', 'b', 'c']
    akMatch = AkMatch()
    # L = akMatch.find_matches(E, 2, 1)
    # print(L)

    # L = [(-1, 1, 5), (-1, 2, 7), (-1, 6, 8), (-1, 3, 11), (-1, 9, 12), (-1, 10, 13)]
    #
    # oMatch = OMatch(L, 2, 1, [2, 4, 4, 7, 2, 1])
    # Mw = oMatch.max_weight_sequence()
    # print(Mw)
    # sln = SLN(1,1,1,1,1)
    # ed = sln.calc_edit_distance(['', 'j', 'a', 'b', 'c', 'd'], ['', 'e'])
    # print(ed)

    # read_activities('data/activities.txt')
    # read_events('data/events.txt')
    read_mappings('data/mappings.txt')


if __name__ == '__main__':
    main()
