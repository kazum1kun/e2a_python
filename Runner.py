import functools
import logging as log
import os
import time
from multiprocessing import Pool
import platform

import editdistance
import psutil
from tqdm import tqdm

from OMatch import OMatch
from SLN import SLN
from k_missing.KGenerator import compare
from utils.FileReader import *
from utils.FileWriter import *
from utils.Timer import Timer


def run_e2a(act_file, event_file, map_file, aoi=None, C=1, method='seg_multi'):
    timer = Timer()
    C = C

    activities = read_activities(act_file)
    events = read_events(event_file)
    mappings = read_mappings(map_file)
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
        oMatch = OMatch(events, mappings, n, ni)
        timer.lap('OMatch initialization done')
        segments, intervals = split_events(oMatch.M, events)
        timer.lap('Segmentation finished')

        if method == 'seg_multi':
            # Multithreading version - by far the fastest and same accuracy as separating version
            process_segment_partial = functools.partial(process_segment, C, mappings, ni)
            # Generator to compact the arguments for the imap function since it takes only one argument
            indexed_segments = ([i, segment] for i, segment in zip(range(len(segments)), segments))

            # Create a pool of worker threads and distribute the segments to these workers
            pool = Pool()
            num_cpu = psutil.cpu_count() * 2

            # results = pool.starmap(process_segment_partial, zip(range(len(segments)), segments))
            results = list(tqdm(pool.imap_unordered(process_segment_partial, indexed_segments,
                                                    chunksize=np.max((int(len(segments) / num_cpu), 1))),
                                total=len(segments), desc='Processing segments...', disable=True))
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
        sln = SLN(mappings, events, ni)
        _, Aw, f_opt = sln.sln_nd(C)
        f_opt_total += f_opt
        Aw_all.extend(Aw[1:])

    timer.lap('Finished SLN')
    activities_calc = [x[0] for x in Aw_all]
    ed_original = editdistance.eval(activities_calc, activities[1:, 1])

    sequences = read_mappings_list(map_file)
    sequence_mapping = compare(sequences)
    sequences_dict = {}
    for entry in sequences:
        sequences_dict[entry[0]] = entry[1]

    prob_matches = []
    for aw in Aw_all:
        possible_matches = sequence_mapping[sequences_dict[tuple(aw)[:2]]]
        if len(possible_matches) > 1:
            prob_matches.append(([idx[0] for idx in possible_matches], aw[4]))
        else:
            prob_matches.append((possible_matches[0][0], aw[4]))

    # Tally the lengths of the solution space
    lengths = {}
    total = 0
    for match in prob_matches:
        if isinstance(match[0], list):
            length = len(match[0])
        else:
            length = 1
        total += length

        if length in lengths:
            lengths[length] += 1
        else:
            lengths[length] = 1

    lengths_sorted = sorted(lengths.items())
    avg = total / len(prob_matches)

    # ed = EditDistance([match[0] for match in prob_matches], activities[:, 1])

    # Strict mode: matches have to be exact to count as correct
    # In order to do so, replace all the activities we are uncertain to act -1,
    # so it's guaranteed to be counted as wrong
    # matches_strict = [act[0] if isinstance(act[0], int) else -1 for act in prob_matches]
    # diff_strict = editdistance.eval(matches_strict, activities[1:, 1])
    # diff_strict = ed.calc_ed(True)

    # Lax mode: for any matches it is only necessary to include it in the guesses to be count as correct
    # diff_lax = ed.calc_ed(False)

    if aoi:
        all_occ_timestamps = [int(match[1]) for match in prob_matches if equal_lax_array(match[0], aoi)]
    else:
        all_occ_timestamps = None

    # Convert the data types to be JSON-compatible
    prob_matches = [[[m[0]], int(m[1])] if isinstance(m[0], int) else [m[0], int(m[1])] for m in prob_matches]

    # Return results as a dictionary
    res = {
        "avg_length": avg,
        "length_dist": lengths_sorted,
        "diff_strict": "skipped",
        "diff_lax": "skipped",
        "diff_original": ed_original,
        "matches": prob_matches,
        "all_timestamps": all_occ_timestamps,
    }

    timer.lap('All done!')
    timer.reset()

    return res


def equal_lax_array(calculated, expected_array):
    return any([equal_lax(calculated, expected) for expected in expected_array])


def equal_lax(calculated, expected):
    if isinstance(calculated, int):
        return calculated == expected
    else:
        return expected in calculated


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
    segments = [np.insert(E[i[0]:i[1] + 1].copy(), 0, 0, 0) for i in intervals]

    return segments, intervals


def process_segment(C, mappings, ni, indexed_segment):
    # Limit CPU usage by setting its priority to "below normal"
    pid = psutil.Process(os.getpid())
    os_ = platform.system()
    if os_ == 'Windows':
        pid.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        pid.nice(-10)
    sln = SLN(mappings, indexed_segment[1], ni)
    _, Aw, f_opt = sln.sln_nd(C)

    return indexed_segment[0], f_opt, Aw[1:]


def main():
    log.basicConfig(filename='debug.log', format='%(message)s', level=log.INFO)
    #
    # progress_bar = tqdm(range(1200), desc='Processing test cases...')
    # for itr in range(100):
    #     for scenario in ['none', 'RS', 'AL_RS_SC']:
    #         mapping_file = f'data/mappings/k_missing/{scenario}_fail.txt'
    #         for act_len in [387, 1494, 2959, 10000]:
    #             out_folder = f'data/output/synth/{act_len}_rand'
    #             if not os.path.exists(out_folder):
    #                 os.mkdir(out_folder)
    #
    #             out_file = f'{out_folder}/{itr}_{scenario}.json'
    #
    #             # Skip the entries that are already computed
    #             if os.path.exists(out_file):
    #                 print(f'Skipping {out_file}')
    #                 progress_bar.update(1)
    #                 continue
    #
    #             activity_file = f'data/activities/synth/{act_len}_rand/{itr}_{scenario}.txt'
    #             event_file = f'data/events/synth/{act_len}_rand/{itr}_{scenario}.txt'
    #
    #             res = run_e2a(activity_file, event_file, mapping_file, aoi=[1, 2, 3, 4, 5, 6])
    #             write_dictionary_json(res, out_file)
    #
    #             progress_bar.update(1)

    # Test multithreading speedup
    progress_bar = tqdm(range(40), desc='Processing test cases...')
    scenario = 'none'

    mapping_file = f'data/mappings/k_missing/{scenario}_fail.txt'

    for act_len in [387, 1494, 2959]:
        time_multi = []
        time_single = []

        for itr in range(10):
            activity_file = f'data/activities/synth/{act_len}_rand/{itr}_{scenario}.txt'
            event_file = f'data/events/synth/{act_len}_rand/{itr}_{scenario}.txt'

            # Multi-threaded version
            start = time.time()
            run_e2a(activity_file, event_file, mapping_file, aoi=None)
            end = time.time()
            time_multi.append(end - start)

            # Single-threaded version
            start = time.time()
            run_e2a(activity_file, event_file, mapping_file, aoi=None, method='mono')
            end = time.time()
            time_single.append(end - start)

            progress_bar.update(1)

        print(f'act_len = {act_len}, multi avg={sum(time_multi) / len(time_multi)}, '
              f'single avg={sum(time_single) / len(time_single)}')


if __name__ == '__main__':
    main()
