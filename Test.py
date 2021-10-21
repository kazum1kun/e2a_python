from AkMatch import AkMatch
from OMatch import OMatch
from SLN import SLN
from FileReader import *

import numpy as np


def main():
    activities = read_activities('data/activities.txt')
    events = read_events('data/events.txt')
    mappings = read_mappings('data/mappings.txt')

    akMatch = AkMatch(events[:, 1], mappings)
    L = akMatch.find_matches(2, 1)

    w = np.ones((22, 3), np.int_)
    oMatch = OMatch(L, 2, 1, w)
    Mw = oMatch.max_weight_sequence()
    print(Mw)





if __name__ == '__main__':
    main()
