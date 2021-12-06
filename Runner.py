import functools
import logging as log
from collections import Counter
from multiprocessing import Pool

import editdistance
from tqdm import tqdm

from OMatch import OMatch
from SLN import SLN
from utils.FileReader import *
from utils.Timer import Timer


def run_e2a(data_ext, map_ext, C=1, method='seg_multi'):
    timer = Timer()
    log.basicConfig(format='%(message)s', level=log.INFO)

    file_ext = data_ext
    mapping_ext = map_ext
    C = C
    activities = read_activities(f'data/activities/activities{file_ext}.txt')
    events = read_events(f'data/events/events{file_ext}.txt')
    mappings = read_mappings(f'data/mappings/mappings{mapping_ext}.txt')
    timer.lap('Text read finished')

    log.info(f'Activity length: {len(activities) - 1}, event length {len(events) - 1}')
    n = len(mappings)
    # Calculate ni values and the N value
    ni = [len(mappings[activity]) - 1 for activity in range(1, n + 1)]
    # Make array 1-based
    ni.insert(0, 0)
    ni = ni

    f_opt_total = 0
    Aw_all = []

    if method == 'seg_multi' or method == 'seg':
        oMatch = OMatch(events[:, 1], mappings, n, ni)
        timer.lap('OMatch initialization done')
        segments, intervals = split_events(oMatch.M, events[:, 1])
        timer.lap('Segmentation finished')

        if method == 'seg_multi':
            # Multithreading version - by far the fastest and same accuracy as separating version
            process_segment_partial = functools.partial(process_segment, C, mappings, ni)
            # Generator to compact the arguments for the imap function since it takes only one argument
            indexed_segments = ([i, segment] for i, segment in zip(range(len(segments)), segments))

            # Create a pool of worker threads and distribute the segments to these workers
            pool = Pool()
            # results = pool.starmap(process_segment_partial, zip(range(len(segments)), segments))
            results = list(tqdm(pool.imap_unordered(process_segment_partial, indexed_segments), total=len(segments),
                                desc='Processing segments...'))
            pool.close()
            pool.join()

            # Sort the results according to the index
            results.sort(key=lambda x: x[0])
            for res in results:
                f_opt_total += res[1]
                if res[2]:
                    Aw_all.extend(res[2])

        # Much faster version that separates input into different segments. Accuracy may suffer for a little bit.
        elif method == 'seg':
            segments_bar = tqdm(segments)
            for segment, interval in zip(segments_bar, intervals):
                segments_bar.set_description(f'Processing segments, current f_opt={f_opt_total}')
                sln = SLN(mappings, segment, ni)
                _, Aw, f_opt = sln.sln_nd(C)
                f_opt_total += f_opt
                Aw_all.extend(Aw[1:])

    # Non-separating version
    elif method == 'mono':
        sln = SLN(mappings, events[:, 1], ni)
        _, Aw, f_opt = sln.sln_nd(C)
        f_opt_total += f_opt
        Aw_all.extend(Aw[1:])

    timer.lap('Finished!')
    print(f'\nThe final f_opt={f_opt_total}')
    # print('\nThe final activities calculated is')
    activities_calc = [x[0] for x in Aw_all]
    # print(activities_calc)
    diff = editdistance.eval(activities_calc, activities[1:, 1])
    print(f'\nThe calculated activities are {diff} edits away from the actual activities')

    calc_counter = Counter(activities_calc)
    actual_counter = Counter(activities[1:, 1])
    calc_counter.subtract(actual_counter)
    # Remove zeroes
    diff_counter = {k: v for k, v in calc_counter.items() if v != 0}
    print(f'\nCalc minus actual is\n{diff_counter}')
    missed = 0
    extra = 0
    for _, v in diff_counter.items():
        if v < 0:
            missed -= v
        if v > 0:
            extra += v
    print(f'\nActivity missed: {missed}, activity extra: {extra}')

    error_pct = diff / len(activities[1:, 1])
    print(f'Accuracy is {1 - error_pct :.5f}')


# Splits the input events into smaller blocks based on the parameter and make sure none of the events are interrupted
def split_events(M, E):
    # Calculate deg, which is then number of crosses a split point makes to matches
    deg = np.array([np.count_nonzero(np.logical_and(M['alpha'] <= i, M['beta'] > i))
                    for i in range(1, len(E))])
    # If any split point has zero deg, it is a suitable candidate to be split
    candidates = np.argwhere(deg == 0).flatten() + 1

    # Create intervals based on the candidates
    intervals = [(candidates[i] + 1, candidates[i + 1]) for i in range(0, len(candidates) - 1)]
    intervals.insert(0, (1, candidates[0]))

    # Create segments of the event using the interval
    segments = [np.insert(E[i[0]:i[1] + 1].copy(), 0, 0) for i in intervals]

    return segments, intervals


def process_segment(C, mappings, ni, indexed_segment):
    sln = SLN(mappings, indexed_segment[1], ni)
    _, Aw, f_opt = sln.sln_nd(C)

    return indexed_segment[0], f_opt, Aw[1:]


if __name__ == '__main__':
    for num in [100, 333, 1000, 2959, 10000, 30000, 100000]:
        run_e2a(f'-synth{num}', '-q')
        print('===================================================')
    for num in [100, 333, 1000, 2959, 10000, 30000, 100000]:
        run_e2a(f'-synth{num}_aqtcfail', '-synth_aqtcfail')
