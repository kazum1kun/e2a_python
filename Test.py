from SLN import SLN
from FileReader import *


def main():
    activities = read_activities('data/activities.txt')
    events = read_events('data/events.txt')
    mappings = read_mappings('data/mappings.txt')

    sln = SLN(mappings, events)
    w, Aw = sln.sln_nd(1)
    print(Aw)


def verify_correctness(actual, truth):
    pass


if __name__ == '__main__':
    main()
