import functools
import logging as log
from multiprocessing import Pool

import editdistance
from tqdm import tqdm

from OMatch import OMatch
from SLN import SLN
from utils import Timer
from utils.FileReader import *


def main():
    log.basicConfig(format='%(message)s', level=log.INFO)

    file_ext = ''
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

    C = 1
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

    # Update the f_opt_total and unpack the results
    f_opt_total = 0
    Aw_all = []

    # Sort the results according to the index
    results.sort(key=lambda x: x[0])
    for res in results:
        f_opt_total += res[1]
        if res[2]:
            Aw_all.extend(res[2])

    # Much faster version that separates input into different segments. Accuracy may suffer for a little bit.
    # segments_bar = tqdm(segments)
    # for segment, interval in zip(segments_bar, intervals):
    # counter += 1
    # segments_bar.set_description(f'Processing segments, current f_opt={f_opt_total}')
    # sln = SLN(mappings, segment, ni)
    # _, Aw, f_opt = sln.sln_nd(1)
    # f_opt_total += f_opt
    # Aw_all.extend(Aw[1:])

    # Non-separating version
    # sln = SLN(mappings, events[:, 1], ni)
    # _, Aw, f_opt = sln.sln_nd(1)
    # f_opt_total += f_opt
    # Aw_all.extend(Aw[1:])

    Timer.lap('Finished!')
    print(f'\nThe final f_opt={f_opt_total}')
    print('\nThe final activities calculated is')
    activities_calc = [x[0] for x in Aw_all]
    print(activities_calc)
    diff = editdistance.eval(activities_calc, activities[1:, 1])
    print(f'\nThe calculated activities are {diff} edits away from the actual activities')


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
    main()
