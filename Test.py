import logging as log

import nltk

from SLN import SLN
from utils import Timer
from utils.FileReader import *


def main():
    log.basicConfig(format='%(message)s', level=log.INFO)

    file_ext = '-real300'
    mapping_ext = ''
    activities = read_activities(f'data/activities/activities{file_ext}.txt')
    events = read_events(f'data/events/events{file_ext}.txt')
    mappings = read_mappings(f'data/mappings/mappings{mapping_ext}.txt')
    Timer.lap('Text read finished.')

    sln = SLN(mappings, events)
    _, Aw, f_opt = sln.sln_nd(2)
    Timer.lap('Finished!')
    print(f'\nThe final f_opt={f_opt}')
    print('\nThe final activities calculated is')
    activities_calc = [x[0] for x in Aw[1:]]
    print(activities_calc)
    diff = nltk.edit_distance(activities_calc, activities[1:, 1])
    print(f'\nThe calculated activities are {diff} edits away from the actual activities')


if __name__ == '__main__':
    main()
