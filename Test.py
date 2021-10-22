from AkMatch import AkMatch
from OMatch import OMatch
from SLN import SLN
from FileReader import *

import numpy as np


def main():
    activities = read_activities('data/activities.txt')
    events = read_events('data/events.txt')
    mappings = read_mappings('data/mappings.txt')

    n = len(mappings)
    # Calculate ni values and the N value
    ni = [len(activity) - 1 for activity in mappings]
    N = np.sum(ni)
    k = np.max(ni)
    # Make array 1-based
    ni.insert(0, 0)

    akMatch = AkMatch(events[:, 1], mappings)
    L = akMatch.find_matches(2, 1)

    w = np.zeros((n + 1, k + 1), np.int_)
    oMatch = OMatch(L, 2, 1)
    Mw = oMatch.max_weight_sequence(w)
    print(Mw)


if __name__ == '__main__':
    main()
