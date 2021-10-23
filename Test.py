from SLN import SLN
from FileReader import *
import Timer


def main():
    # activities = read_activities('data/activities/activities.txt')
    # events = read_events('data/events/events.txt')
    # mappings = read_mappings('data/mappings/mappings.txt')
    activities = read_activities('data/activities/activities-ex1.txt')
    events = read_events('data/events/events-ex1.txt')
    mappings = read_mappings('data/mappings/mappings-ex1.txt')
    Timer.lap('Text read finished.')

    sln = SLN(mappings, events)
    w, Aw = sln.sln_nd(1)
    print(Aw)


def verify_correctness(actual, truth):
    pass


if __name__ == '__main__':
    main()
