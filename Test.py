import logging as log

import nltk

from SLN import SLN
from utils import Timer
from utils.FileReader import *


def main():
    log.basicConfig(format='%(message)s', level=log.DEBUG)

    file_ext = '-test'
    mapping_ext = '-test'
    activities = read_activities(f'data/activities/activities{file_ext}.txt')
    events = read_events(f'data/events/events{file_ext}.txt')
    mappings = read_mappings(f'data/mappings/mappings{mapping_ext}.txt')
    Timer.lap('Text read finished.')

    sln = SLN(mappings, events)
    w, Aw, f_opt = sln.sln_nd(2)
    print('\nThe final weight matrix is')
    print(w)
    print(f'\nThe final {f_opt=}')
    print('\nThe final activities calculated is')
    print(Aw[1:])
    diff = nltk.edit_distance(Aw[1:], activities[1:, 1])
    print(f'\nThe calculated activities are {diff} edits away from the actual activities')


if __name__ == '__main__':
    main()
