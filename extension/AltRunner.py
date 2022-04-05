import glob
import os
from functools import partial
from multiprocessing import Pool
from statistics import NormalDist

import psutil
from tqdm import tqdm

from WindowMatch import WindowMatch
from utils.FileReader import *


# Get check if the given activity occurred near the given timestamp
# NOTE: delta will be applied both ways (forward and backward), effectively resulting in 2*delta search zone
def check_activity(activities, time, delta, aoi):
    activity_idx = np.where(np.logical_and(activities[:, 1] == aoi,
                                           np.logical_and(activities[:, 0] >= time - delta,
                                                          activities[:, 0] <= time + delta)))[0]

    # If we found multiple, return the one closest to the time given
    if len(activity_idx) > 1:
        closest_idx = np.argmin(np.abs(activities[activity_idx, 0] - time))
        ret = activity_idx[closest_idx]
    elif len(activity_idx) == 1:
        ret = activity_idx[0]
    else:
        ret = None

    return ret


# Check a sequence of timestamps and compare it against the ground truth
def check_input(activities, times, aoi, delta):
    correct = 0
    incorrect = 0
    duplicate = 0
    used_idx = set()
    incorrect_idx = set()

    for time in times:
        # check_partial = partial(check_activity, activities, time, delta)
        # res = list(map(check_partial, aoi))
        res = check_activity(activities, time, delta, aoi)

        if not res:
            incorrect += 1
            incorrect_idx.add(time)
        else:
            act_idx = res
            if act_idx in used_idx:
                duplicate += 1
            else:
                correct += 1
                used_idx.add(act_idx)

    # Merge incorrect to a bigger window to smooth over data
    incorrect_sorted = sorted(incorrect_idx)
    if len(incorrect_sorted) == 0:
        return correct, incorrect, 0
    start = incorrect_sorted[0]
    last = incorrect_sorted[0]
    incorrect_adj = 0
    for i in range(len(incorrect_sorted)):
        if (incorrect_sorted[i] - last) <= delta:
            last = incorrect_sorted[i]
            continue
        incorrect_adj += (last - start) / (delta * 2)
        start = incorrect_sorted[i]
        last = incorrect_sorted[i]

    return correct, incorrect, incorrect_adj


# Check a sequence of timestamps and compare it against the ground truth
def check_input_list(activities, times, aoi, delta):
    correct = 0
    incorrect = 0
    duplicate = 0
    used_idx = set()

    for time in times:
        check_partial = partial(check_activity, activities, time, delta)
        res = list(map(check_partial, aoi))
        # # check_partial = partial(check_activity, activities, time, delta)
        # # res = list(map(check_partial, aoi))
        # res = check_activity(activities, time, delta, aoi)

        if not any(res):
            incorrect += 1
        else:
            act_idx = next(item for item in res if item is not None)
            if act_idx in used_idx:
                duplicate += 1
            else:
                correct += 1
                used_idx.add(act_idx)

    return correct, incorrect


# Verify whether specific activities did occur on the timestamps specified in the output file
def verify_matches(output_file, activity_file, aoi):
    times = read_json(output_file)["all_timestamps"]
    activities = read_activities(activity_file)

    ak_correct_total = 0
    ak_miss_total = 0
    ak_incorrect_total = 0
    expected_total = 0

    # for act in aoi:
    correct, incorrect = check_input_list(activities, times, aoi, 5)

    expected = np.sum([np.count_nonzero(activities[:, 1] == i) for i in aoi])
    # expected = np.count_nonzero(activities[:, 1] == act)
    miss = expected - correct

    ak_correct_total += correct
    ak_miss_total += miss
    ak_incorrect_total += incorrect
    expected_total += expected

    return ak_correct_total, ak_miss_total, ak_incorrect_total, expected_total


def start_verifier():
    for scenario in ['none', 'RS', 'AL_RS_SC']:
        for act_len in [387]:
        # for act_len in [387, 1494, 2959, 10000]:
            all_output = sorted(glob.glob(f'data/output/synth/{act_len}/*_{scenario}_new.json'))
            all_acts = sorted(glob.glob(f'data/activities/synth/{act_len}/*_{scenario}.txt'))

            correct_total = 0
            incorrect_total = 0
            expected_total = 0
            missed_total = 0

            count = 0
            for output, act in zip(all_output, all_acts):
                if count > 10:
                    break
                aoi = [1, 2, 3, 4, 5, 6]
                correct, miss, incorrect, expected = verify_matches(output, act, aoi)
                correct_total += correct
                incorrect_total += incorrect
                expected_total += expected
                missed_total += miss
                count += 1

            print(f'{scenario=}, {act_len=}, expected = {expected_total}, correct={correct_total}, '
                  f'missed={missed_total}, incorrect={incorrect_total}, total={correct_total + incorrect_total}')


# Run the matching algorithms
def run_matching(map_file, aoi, delta, act_event):
    pid = psutil.Process(os.getpid())
    pid.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    act_file = act_event[0]
    event_file = act_event[1]

    # Alternative runner which tests AkMatch vs WindowMatch
    activities = read_activities(act_file)
    events = read_events(event_file)
    mappings = read_mappings(map_file)

    wm = WindowMatch(events, mappings)

    # Calculate ni values
    ni = [len(mappings[activity]) - 1 for activity in range(1, len(mappings) + 1)]
    # Make array 1-based
    ni.insert(0, 0)

    wm_correct_total = 0
    wm_miss_total = 0
    wm_incorrect_total = 0
    expected_total = 0
    incorrect_adj_total = 0

    for act in aoi:
        # Using set to prevent dups
        start_wm = set()

        duration = 20
        for k in range(1, ni[act] + 1):
            # Run the matching on each algorithm and add the timestamps of each match found
            result_wm = wm.find_matches(act, k, duration)
            start_wm.update([x[1] for x in result_wm])

        wm_correct, wm_incorrect, wm_incorrect_adj = check_input(activities, start_wm, act, delta)
        expected = np.sum(np.count_nonzero(activities[:, 1] == act))

        wm_miss = expected - wm_correct
        wm_correct_total += wm_correct
        wm_miss_total += wm_miss
        wm_incorrect_total += wm_incorrect
        expected_total += expected
        incorrect_adj_total += wm_incorrect_adj

    return wm_correct_total, wm_miss_total, wm_incorrect_total, expected_total, incorrect_adj_total


# Get the expected activity duration given a percentile (result centered around mean)
def get_expected_duration(activity, pct):
    # Hard-coded data from experiment data
    mean = [0, 15.927, 43.086, 10.553, 38.152, 10.553, 35.542]
    std = [0, 1.352, 9.363, 1.016, 8.668, 0.973, 12.047]

    # Calculate the inverse cdf given the percentile
    max_duration = NormalDist(mu=mean[activity], sigma=std[activity]).inv_cdf(pct)

    return max_duration


def start_matching():
    for scenario in ['none', 'RS', 'AL_RS_SC']:
        for act_len in [387, 1494, 2959, 10000]:
            all_acts = sorted(glob.glob(f'data/activities/synth/{act_len}/*_{scenario}.txt'))
            all_events = sorted(glob.glob(f'data/events/synth/{act_len}/*_{scenario}.txt'))

            mapping = f'data/mappings/k_missing/{scenario}_fail.txt'
            aoi = [1, 2, 3, 4, 5, 6]

            wm_correct_total = 0
            wm_missed_total = 0
            wm_incorrect_total = 0
            expected_total = 0
            wm_incorrect_adj_total = 0
            #
            # pool = Pool()
            # partial_run = partial(run_matching, mapping, aoi, 5)
            #
            #
            #
            # results = list(tqdm(pool.imap_unordered(partial_run, zip(all_acts, all_events)),
            #                     total=len(all_acts), desc="Processing testcases..."))

            # pool.close()
            # pool.join()
            pbar = tqdm(range(100))
            itr = 0
            for acts_events in zip(all_acts, all_events):
                wm_correct, wm_miss, wm_incorrect, expected, wm_incorrect_adj = \
                    run_matching(mapping, aoi, 5, acts_events)
                wm_correct_total += wm_correct
                wm_missed_total += wm_miss
                wm_incorrect_total += wm_incorrect
                expected_total += expected
                wm_incorrect_adj_total += wm_incorrect_adj
                pbar.update(1)
                itr += 1

            # wm_correct_total = np.sum([result[0] for result in results])
            # wm_missed_total = np.sum([result[1] for result in results])
            # wm_incorrect_total = np.sum([result[2] for result in results])
            # wm_duplicate_total = np.sum([result[3] for result in results])
            # expected_total = np.sum([result[4] for result in results])

            print(f'{scenario=}, {act_len=}\n'
                  f'{wm_correct_total=}, {wm_missed_total=}, {wm_incorrect_total=} (adj = {wm_incorrect_adj_total})\n'
                  f'{expected_total=}\n')
