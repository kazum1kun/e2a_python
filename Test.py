from SLN import SLN
from FileReader import *
import Timer


def main():
    file_ext = '-ex2'
    activities = read_activities(f'data/activities/activities{file_ext}.txt')
    events = read_events(f'data/events/events{file_ext}.txt')
    mappings = read_mappings(f'data/mappings/mappings{file_ext}.txt')
    Timer.lap('Text read finished.')

    sln = SLN(mappings, events)
    w, Aw = sln.sln_nd(1)
    print(Aw)


def verify_correctness(actual, truth):
    pass


if __name__ == '__main__':
    main()
