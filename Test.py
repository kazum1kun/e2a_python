from AkMatch import AkMatch
from OMatch import OMatch
from SLN import SLN
from FileReader import *

import numpy as np


def main():
    activities = read_activities('data/activities.txt')
    events = read_events('data/events.txt')
    mappings = read_mappings('data/mappings.txt')

    sln = SLN(mappings, events)
    sln.sln_nd()


if __name__ == '__main__':
    main()
