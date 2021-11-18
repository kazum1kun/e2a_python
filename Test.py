import logging as log
import time

import nltk
from tqdm import tqdm

from SLN import SLN
from utils import Timer
from utils.FileReader import *
from OMatch import OMatch

import difflib


def main():
    log.basicConfig(format='%(message)s', level=log.INFO)

    file_ext = '-revised'
    mapping_ext = '-q'
    activities = read_activities(f'data/activities/activities{file_ext}.txt')
    events = read_events(f'data/events/events{file_ext}.txt')
    mappings = read_mappings(f'data/mappings/mappings{mapping_ext}.txt')
    Timer.lap('Text read finished')

    n = len(mappings)
    # Calculate ni values and the N value
    ni = [len(mappings[activity]) - 1 for activity in range(1, n + 1)]
    # Make array 1-based
    ni.insert(0, 0)
    ni = ni

    oMatch = OMatch(events[:, 1], mappings, n, ni)
    Timer.lap('OMatch initialization done')
    segments, intervals = split_events(oMatch.M, events[:, 1])
    Timer.lap('Segmentation finished')
    f_opt_total = 0
    counter = 0
    Aw_all = []
    full = []
    for i in activities[1:, 1]:
        full.append(i)
    print(full)

    segments_bar = tqdm(segments)
    for segment, interval in zip(segments_bar, intervals):
        counter += 1
        segments_bar.set_description(f'Processing segments, current f_opt={f_opt_total}')
        # start = time.perf_counter()
        sln = SLN(mappings, segment, ni)
        _, Aw, f_opt = sln.sln_nd(1)
        f_opt_total += f_opt
        Aw_all.extend(Aw[1:])

        # print(f'Segment {counter: >3}, [{intervals[counter-1][0]:>5},{intervals[counter-1][1]: <5}] '
        #       f'(len={intervals[counter-1][1] - intervals[counter-1][0] + 1: >3}), '
        #       f'{f_opt=:>2}, cumulative f_opt={f_opt_total:>3}, spent {time.perf_counter() - start:.2f} seconds')

        # calc_diff(segment, Aw, mappings)

        # print('\n----------------------------------------------')

    Timer.lap('Finished!')
    print(f'\nThe final f_opt={f_opt_total}')
    print('\nThe final activities calculated is')
    activities_calc = [x[0] for x in Aw_all]
    print(activities_calc)
    diff = nltk.edit_distance(activities_calc, activities[1:, 1])
    print(f'\nThe calculated activities are {diff} edits away from the actual activities')


# Splits the input events into smaller blocks based on the parameter and make sure none of the events are interrupted
def split_events(M, E):
    # Calculate deg, which is then number of crosses a split point makes to matches
    deg = np.array([np.count_nonzero(np.logical_and(M['alpha'] <= i, M['beta'] > i))
                    for i in range(1, len(E))])
    # If any split point has zero deg, it is a suitable candidate to be split
    candidates = np.argwhere(deg == 0).flatten() + 1

    # Create intervals based on the candidates
    intervals = [(candidates[i] + 1, candidates[i+1]) for i in range(0, len(candidates) - 1)]
    intervals.insert(0, (1, candidates[0]))

    # Create segments of the event using the interval
    segments = [np.insert(E[i[0]:i[1] + 1].copy(), 0, 0) for i in intervals]

    return segments, intervals


def calc_diff(E_truth, Aw_new, S):
    E_truth_chr = [chr(event) for event in E_truth[1:]]
    E_truth_str = ''.join(E_truth_chr)

    events_new = ''
    E_new = []
    for A in Aw_new[1:]:
        events = S[A[0]][A[1]][1:]
        events_chr = [chr(event) for event in events]
        events_str = ''.join(events_chr)
        E_new.append(events_str)
        events_new = ''.join(E_new)

    m = difflib.SequenceMatcher(a=E_truth_str, b=events_new)

    for tag, i1, i2, j1, j2 in m.get_opcodes():
        print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
            tag, i1, i2, j1, j2, E_truth_str[i1:i2], events_new[j1:j2]))


if __name__ == '__main__':
    main()
