import functools
import logging as log
import os
from collections import Counter
from multiprocessing import Pool
import psutil

import editdistance
from tqdm import tqdm
import numpy as np

from OMatch import OMatch
from SLN import SLN
from utils.FileReader import *
from utils.FileWriter import *
from utils.Timer import Timer
from k_missing.KGenerator import compare


def run_e2a(act_file, event_file, map_file, C=1, method='seg_multi'):
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
                                desc='Processing segments...', disable=False))
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

    fin_time = timer.get_elapsed()
    timer.lap('Finished!')
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
            prob_matches.append([idx[0] for idx in possible_matches])
        else:
            prob_matches.append(possible_matches[0][0])

    # Tally the lengths of the solution space
    lengths = {}
    total = 0
    for match in prob_matches:
        if isinstance(match, list):
            length = len(match)
        else:
            length = 1
        total += length

        if length in lengths:
            lengths[length] += 1
        else:
            lengths[length] = 1

    lengths_sorted = sorted(lengths.items())
    avg = total / len(prob_matches)

    # Strict mode: matches have to be exact to count as correct
    diff_strict, missed_strict, extra_strict, error_pct_strict = \
        calc_ed(prob_matches, activities[1:, 1], strict_mode=True)

    # Lax mode: for any matches it is only necessary to include it in the guesses to be count as correct
    diff_lax, missed_lax, extra_lax, error_pct_lax = calc_ed(prob_matches, activities[1:, 1], strict_mode=False)

    # Return results as a dictionary
    res = {
        "avg_length": avg,
        "length_dist": lengths_sorted,
        "diff_strict": diff_strict,
        "diff_lax": diff_lax,
        "error_strict": error_pct_strict,
        "error_lax": error_pct_lax,
        "diff_original":ed_original
    }

    return res


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
    # Limit CPU usage by setting its priority to "below normal"
    pid = psutil.Process(os.getpid())
    pid.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    sln = SLN(mappings, indexed_segment[1], ni)
    _, Aw, f_opt = sln.sln_nd(C)

    return indexed_segment[0], f_opt, Aw[1:]


def get_diff(calculated, actual):
    diff = editdistance.eval(calculated, actual)

    calc_counter = Counter(calculated)
    actual_counter = Counter(actual)
    calc_counter.subtract(actual_counter)
    # Remove zeroes
    diff_counter = {k: v for k, v in calc_counter.items() if v != 0}

    missed = 0
    extra = 0
    for _, v in diff_counter.items():
        if v < 0:
            missed -= v
        if v > 0:
            extra += v

    error_pct = diff / len(actual)
    return diff, missed, extra, error_pct


def calc_ed(prob_match, actual, strict_mode):
    m = len(prob_match)
    n = len(actual)
    c = np.zeros((m + 1, n + 1))

    for j in range(0, n + 1):
        c[0, j] = j
    for i in range(0, m + 1):
        c[i, 0] = i

    missed = 0
    extra = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if isinstance(prob_match[i - 1], int) and prob_match[i - 1] == actual[j - 1] or \
               isinstance(prob_match[i - 1], list) and not strict_mode and actual[j - 1] in prob_match[i - 1]:
                c[i, j] = c[i - 1, j - 1]
            else:
                c[i, j] = np.min((c[i-1, j-1], c[i, j-1], c[i-1, j])) + 1

    error_pct = c[m, n] / n

    return c[m, n], missed, extra, error_pct

# def approximate_lax_dist(prob_match, actual, tolerance):
#     idx_actual = 0
#     idx_match = 0
#     match = 0
#     miss = 0
#     extra = 0
#
#     for act in actual:
#         for i in range(tolerance):
#             temp_idx = idx_match + i
#             if temp_idx < len(prob_match):
#                 if (isinstance(prob_match[temp_idx], int) and act == prob_match[temp_idx]) or \
#                         (isinstance(prob_match[temp_idx], list) and act in prob_match[temp_idx]):
#                     match += 1
#                     idx_match = temp_idx + 1
#                     break
#                 else:
#                     miss += 1
#                     break
#         if idx_actual - idx_match == tolerance:
#             idx_match += tolerance
#             extra += tolerance
#
#         idx_actual += 1
#     extra += len(prob_match) - idx_match
#
#     diff = len(actual) - match
#     error_pct = diff / len(actual)
#
#     return diff, miss, extra, error_pct


def main():
    log.basicConfig(filename='debug.log', format='%(message)s', level=log.INFO)

    progress_bar = tqdm(range(60), desc='Processing synth test cases...')

    for itr in range(100):
        for scenario in ['none', 'RS', 'AL_RS_SC']:
            mapping_file = f'data/mappings/k_missing/{scenario}_fail.txt'
            for act_len in [387, 1494, 2959, 10000]:
            # for itr in range(2):
                out_file = f'data/output/synth/{act_len}/{itr}_{scenario}.json'

                # Skip the entries that are already computed
                if os.path.exists(out_file):
                    print(f'Skipping {out_file}')
                    progress_bar.update(1)
                    continue

                activity_file = f'data/activities/synth/{act_len}/{itr}_{scenario}.txt'
                event_file = f'data/events/synth/{act_len}/{itr}_{scenario}.txt'

                res = run_e2a(activity_file, event_file, mapping_file)
                write_dictionary_json(res, out_file)

                progress_bar.update(1)

    # mapping = f'data/mappings/with_q.txt'
    # for length in [387, 1494, 2959]:
    #     act_file = f'data/activities/synth/dev_fails/none_{length}.txt'
    #     event_file = f'data/events/synth/dev_fails/none_{length}.txt'
    #
    #     print(f'Running original length {length}')
    #     run_e2a(act_file, event_file, mapping, max_cpu=0.8)
    #
    # for activity in ['AQ', 'RS', 'AL_RS_SC']:
    #     for length in [387, 1494, 2959]:
    #         mapping = f'data/mappings/k_missing/{activity}_fail.txt'
    #         act_file = f'data/activities/synth/dev_fails/{activity}_{length}.txt'
    #         event_file = f'data/events/synth/dev_fails/{activity}_{length}.txt'
    #
    #         print(f'Running {activity} length {length}')
    #         run_e2a(act_file, event_file, mapping, max_cpu=0.8)



    # progress_bar = tqdm(range(1000), desc='Processing synth test cases...')
    # # input_types = ['real', 'normal', 'fail']
    # input_types = ['normal']
    # for input_type in input_types:
    #     if input_type == 'real' or input_type == 'normal':
    #         mapping = 'data/mappings/with_q.txt'
    #     else:
    #         mapping = 'data/mappings/synth_combined.txt'
    #
    #     if input_type == 'real':
    #         for length in [387, 1494, 2959]:
    #             activity_file = f'data/activities/real/{length}.txt'
    #             event_file = f'data/events/real/{length}.txt'
    #             res = run_e2a(activity_file, event_file, mapping)
    #
    #             print(f'\n\nType: {input_type} length: {length}\n'
    #                   f'Time: {res[0]:.5f}, Missed: {res[3]:.5f}, Extra: {res[4]:.5f}, ED (Act): {res[2]:.5f}, '
    #                   f'ED (Event): {res[1]:.5f}, Acc: {1 - res[5]:.5f}')
    #             print('====================================================')
    #     else:
    #         # for length in [387, 1494, 2959, 10000, 30000]:
    #         for length in [387]:
    #             time = []
    #             missed = []
    #             extra = []
    #             ed_act = []
    #             ed_event = []
    #             acc = []
    #
    #             for itr in range(100):
    #                 if not os.path.exists(f'data/output/synth/{length}'):
    #                     os.mkdir(f'data/output/synth/{length}')
    #
    #                 if input_type == 'normal':
    #                     activity_file = f'data/activities/synth/{length}/{itr}.txt'
    #                     event_file = f'data/events/synth/{length}/{itr}.txt'
    #
    #                 else:
    #                     activity_file = f'data/activities/synth/{length}/{itr}_aqtcfail.txt'
    #                     event_file = f'data/events/synth/{length}/{itr}_aqtcfail.txt'
    #
    #                 res = run_e2a(activity_file, event_file, mapping)
    #                 time.append(res[0])
    #                 ed_event.append(res[1])
    #                 ed_act.append(res[2])
    #                 missed.append(res[3])
    #                 extra.append(res[4])
    #                 acc.append(1 - res[5])
    #
    #                 progress_bar.update(1)
    #
    #             print(f'\n\nType: {input_type} length: {length}\n'
    #                   f'Time: avg={np.mean(time):.5f}, median={np.median(time):.5f}, min={np.min(time):.5f}, max={np.max(time):.5f}, std={np.std(time):.5f}\n'
    #                   f'Missed: avg={np.mean(missed):.5f}, median={np.median(missed):.5f}, min={np.min(missed):.5f}, max={np.max(missed):.5f}, std={np.std(missed):.5f}\n'
    #                   f'Extra: avg={np.mean(extra):.5f}, median={np.median(extra):.5f}, min={np.min(extra):.5f}, max={np.max(extra):.5f}, std={np.std(extra):.5f}\n'
    #                   f'ED (Act): avg={np.mean(ed_act):.5f}, median={np.median(ed_act):.5f}, min={np.min(ed_act):.5f}, max={np.max(ed_act):.5f}, std={np.std(ed_act):.5f}\n'
    #                   f'ED (Event): avg={np.mean(ed_event):.5f}, median={np.median(ed_event):.5f}, min={np.min(ed_event):.5f}, max={np.max(ed_event):.5f}, std={np.std(ed_event):.5f}\n'
    #                   f'Acc: avg={np.mean(acc):.5f}, median={np.median(acc):.5f}, min={np.min(acc):.5f}, max={np.max(acc):.5f}, std={np.std(acc):.5f}')
    #             print('====================================================')


if __name__ == '__main__':
    main()
