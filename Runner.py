import functools
import logging as log
import os
from collections import Counter
from multiprocessing import Pool

import editdistance
from tqdm import tqdm

from OMatch import OMatch
from SLN import SLN
from utils.FileReader import *
from utils.Timer import Timer


def run_e2a(act_file, event_file, map_file, C=1, method='seg_multi'):
    timer = Timer()
    log.basicConfig(format='%(message)s', level=log.WARNING)
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
                                desc='Processing segments...', disable=True))
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
    # print('\nThe final activities calculated is')
    activities_calc = [x[0] for x in Aw_all]
    # print(activities_calc)
    diff = editdistance.eval(activities_calc, activities[1:, 1])

    calc_counter = Counter(activities_calc)
    actual_counter = Counter(activities[1:, 1])
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
    # print(f'\nActivity missed: {missed}, activity extra: {extra}')

    error_pct = diff / len(activities[1:, 1])
    return fin_time, f_opt_total, diff, missed, extra, error_pct, diff_counter


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


def main():
    progress_bar = tqdm(range(400), desc='Processing synth test cases...')
    input_types = ['normal', 'fail']
    for input_type in input_types:
        if input_type == 'real' or input_type == 'normal':
            mapping = 'data/mappings/with_q.txt'
        else:
            mapping = 'data/mappings/synth_combined.txt'

        if input_type == 'real':
            for length in [387, 1494, 2959]:
                activity_file = f'data/activities/real/{length}.txt'
                event_file = f'data/events/real/{length}.txt'
                res = run_e2a(activity_file, event_file, mapping)

                print(f'\n\nType: {input_type} length: {length}\n'
                      f'Time: {res[0]:.5f}, Missed: {res[3]:.5f}, Extra: {res[4]:.5f}, ED (Act): {res[2]:.5f}, '
                      f'ED (Event): {res[1]:.5f}, Acc: {1 - res[5]:.5f}')
                print('====================================================')
        else:
            for length in [10000, 30000]:
                time = []
                missed = []
                extra = []
                ed_act = []
                ed_event = []
                acc = []

                for itr in range(40, 100):
                    if not os.path.exists(f'data/output/synth/{length}'):
                        os.mkdir(f'data/output/synth/{length}')

                    if input_type == 'normal':
                        activity_file = f'data/activities/synth/{length}/{itr}.txt'
                        event_file = f'data/events/synth/{length}/{itr}.txt'
                        diff_output = f'data/output/synth/{length}/{itr}.txt'
                    else:
                        activity_file = f'data/activities/synth/{length}/{itr}_aqtcfail.txt'
                        event_file = f'data/events/synth/{length}/{itr}_aqtcfail.txt'
                        diff_output = f'data/output/synth/{length}/{itr}_aqtcfail.txt'
                    res = run_e2a(activity_file, event_file, mapping)
                    time.append(res[0])
                    ed_event.append(res[1])
                    ed_act.append(res[2])
                    missed.append(res[3])
                    extra.append(res[4])
                    acc.append(1 - res[5])

                    # with open(diff_output, 'w') as out_file:
                    #     out_file.write(res[6].__repr__())
                    progress_bar.update(1)

                print(f'\n\nType: {input_type} length: {length}\n'
                      f'Time: avg={np.mean(time):.5f}, median={np.median(time):.5f}, min={np.min(time):.5f}, max={np.max(time):.5f}, std={np.std(time):.5f}\n'
                      f'Missed: avg={np.mean(missed):.5f}, median={np.median(missed):.5f}, min={np.min(missed):.5f}, max={np.max(missed):.5f}, std={np.std(missed):.5f}\n'
                      f'Extra: avg={np.mean(extra):.5f}, median={np.median(extra):.5f}, min={np.min(extra):.5f}, max={np.max(extra):.5f}, std={np.std(extra):.5f}\n'
                      f'ED (Act): avg={np.mean(ed_act):.5f}, median={np.median(ed_act):.5f}, min={np.min(ed_act):.5f}, max={np.max(ed_act):.5f}, std={np.std(ed_act):.5f}\n'
                      f'ED (Event): avg={np.mean(ed_event):.5f}, median={np.median(ed_event):.5f}, min={np.min(ed_event):.5f}, max={np.max(ed_event):.5f}, std={np.std(ed_event):.5f}\n'
                      f'Acc: avg={np.mean(acc):.5f}, median={np.median(acc):.5f}, min={np.min(acc):.5f}, max={np.max(acc):.5f}, std={np.std(acc):.5f}')
                print('====================================================')


if __name__ == '__main__':
    main()
