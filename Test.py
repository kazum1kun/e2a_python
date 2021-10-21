from AkMatch import AkMatch
from OMatch import OMatch
from SLN import SLN
from FileReader import *


def main():
    activities = read_activities('data/activities.txt')
    events = read_events('data/events.txt')
    mappings = read_mappings('data/mappings.txt')

    akMatch = AkMatch(events[:, 1], mappings)
    L = akMatch.find_matches(2, 1)
    print(L)

    L = [(-1, 1, 5), (-1, 2, 7), (-1, 6, 8), (-1, 3, 11), (-1, 9, 12), (-1, 10, 13)]

    oMatch = OMatch(L, 2, 1, [2, 4, 4, 7, 2, 1])
    Mw = oMatch.max_weight_sequence()
    print(Mw)
    sln = SLN(1,1,1,1,1)
    ed = sln.calc_edit_distance(['', 'j', 'a', 'b', 'c', 'd'], ['', 'e'])
    print(ed)




if __name__ == '__main__':
    main()
