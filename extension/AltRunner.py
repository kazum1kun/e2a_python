import glob
import os
from functools import partial
from multiprocessing import Pool
from statistics import NormalDist

import psutil
from tqdm import tqdm

from extension.WindowMatch import WindowMatch
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
def check_input(activities, times, aoi, delta, incorrect_times):
    correct = 0
    duplicate = 0
    used_idx = set()

    for time in times:
        # check_partial = partial(check_activity, activities, time, delta)
        # res = list(map(check_partial, aoi))
        res = check_activity(activities, time, delta, aoi)

        if not res:
            if time not in incorrect_times:
                incorrect_times.add(time)
        else:
            act_idx = res
            if act_idx in used_idx:
                duplicate += 1
            else:
                correct += 1
                used_idx.add(act_idx)

    return correct


# Check a sequence of timestamps and compare it against the ground truth
def check_input_list(activities, times, aoi, delta):
    correct = 0
    duplicate = 0
    used_idx = set()
    incorrect_times = set()

    for time in times:
        check_partial = partial(check_activity, activities, time, delta)
        res = list(map(check_partial, aoi))

        if not any(res):
            incorrect_times.add(time)
        else:
            act_idx = next(item for item in res if item is not None)
            if act_idx in used_idx:
                duplicate += 1
            else:
                correct += 1
                used_idx.add(act_idx)

    incorrect = len(incorrect_times)

    return correct, incorrect


# Verify whether specific activities did occur on the timestamps specified in the output file
def verify_matches(output_file, activity_file, aoi, delta):
    times = read_json(output_file)["all_timestamps"]
    activities = read_activities(activity_file)

    ak_correct, ak_incorrect = check_input_list(activities, times, aoi, delta)

    ak_expected = np.sum([np.count_nonzero(activities[:, 1] == i) for i in aoi])
    ak_miss = ak_expected - ak_correct

    return ak_correct, ak_miss, ak_incorrect


def start_verifier(delta):
    for scenario in ['none', 'RS', 'AL_RS_SC']:
        for act_len in [387, 1494, 2959, 10000]:
            all_output = sorted(glob.glob(f'data/output/synth/{act_len}_rand/*_{scenario}.json'))
            all_acts = sorted(glob.glob(f'data/activities/synth/{act_len}_rand/*_{scenario}.txt'))

            correct_total = 0
            incorrect_total = 0
            missed_total = 0

            for output, act in zip(all_output, all_acts):
                aoi = [1, 2, 3, 4, 5, 6]
                correct, miss, incorrect= verify_matches(output, act, aoi, delta)
                correct_total += correct
                incorrect_total += incorrect
                missed_total += miss

            print(f'{scenario=}, {act_len=}, correct={correct_total}, '
                  f'missed={missed_total}, incorrect={incorrect_total}')


def start_verifier_real(delta):
    for length in [387, 1494, 2959]:
        output = f'data/output/real/{length}.json'
        act = f'data/activities/real/{length}.txt'

        aoi = [1, 2, 3, 4, 5, 6]
        correct, missed, incorrect = verify_matches(output, act, aoi, delta)

        print(f'scenario=real_med, {length=}, {correct=}, {missed=}, {incorrect=}')


# Run the matching algorithms
def run_matching(map_file, aoi, delta, theta, act_event):
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
    expected_total = 0

    incorrect_times = set()

    for act in aoi:
        # Using set to prevent dups
        start_wm = set()

        for k in range(1, ni[act] + 1):
            # Run the matching on each algorithm and add the timestamps of each match found
            result_wm = wm.find_matches(act, k, theta[act - 1])
            start_wm.update([x[1] for x in result_wm])

        wm_correct = check_input(activities, start_wm, act, delta, incorrect_times)
        expected = np.sum(np.count_nonzero(activities[:, 1] == act))

        wm_miss = expected - wm_correct
        wm_correct_total += wm_correct
        wm_miss_total += wm_miss
        expected_total += expected

    incorrect = len(incorrect_times)
    incorrect_adj = combine_incorrect_time(incorrect_times, delta)

    return wm_correct_total, wm_miss_total, incorrect, expected_total, incorrect_adj


# Combine incorrect occurrences given the delta value
def combine_incorrect_time(incorrect_times, delta):
    incorrect_sorted = sorted(incorrect_times)
    # Merge incorrect to a bigger window to smooth over data
    if delta == 0 or len(incorrect_sorted) == 0:
        incorrect_adj = len(incorrect_times)
    else:
        last = incorrect_sorted[0]
        incorrect_adj = 0
        incorrect_current = 0
        for i in range(len(incorrect_sorted)):
            if (incorrect_sorted[i] - last) <= delta:
                last = incorrect_sorted[i]
                incorrect_current += 1
            else:
                incorrect_current = np.ceil(incorrect_current / (delta * 2))
                incorrect_adj += incorrect_current
                incorrect_current = 1
                last = incorrect_sorted[i]
        if incorrect_current > 0:
            incorrect_current = np.ceil(incorrect_current / (delta * 2))
            incorrect_adj += incorrect_current

    return incorrect_adj


# Get the expected activity duration given a percentile (result centered around mean)
def get_expected_duration(activity, pct):
    # Hard-coded stats from the real input data
    mean = [0, 15.927, 43.086, 10.553, 38.152, 10.553, 35.542, 3.478975000011945, 33.38133303571937,
            1.7387140496036957, 30.230282142853866, 1.766518000012355, 1.7687680272112853, 8.722998275897634,
            34.815031858422515, 1.7897369564810146, 1.8035833333142914, 1.7429857143035536, 0.0, 1.7369035714293983,
            1.7274778760885676, 5.197146551699853]
    std = [0, 1.352, 9.363, 1.016, 8.668, 0.973, 12.047, 0.5724097746240778, 4.576370287335542, 0.43400412044630904,
           7.6878460794186605, 0.4342020328161984, 0.41127895816118915, 0.8598459123163101, 10.874888589499959,
           0.41721265354207665, 0.40576106925669286, 0.4239521594901133, 0.0, 0.40925221046208177, 0.4422595859454422,
           0.6973014127669434]

    # Calculate the inverse cdf given the percentile
    max_duration = NormalDist(mu=mean[activity], sigma=std[activity]).inv_cdf(pct)

    return max_duration


def start_matching(delta, theta):
    for scenario in ['none', 'RS', 'AL_RS_SC']:
        for act_len in [387, 1494, 2959, 10000]:
            all_acts = sorted(glob.glob(f'data/activities/synth/{act_len}_rand/*_{scenario}.txt'))
            all_events = sorted(glob.glob(f'data/events/synth/{act_len}_rand/*_{scenario}.txt'))

            mapping = f'data/mappings/k_missing/{scenario}_fail.txt'
            aoi = [1, 2, 3, 4, 5, 6]
            theta_dist = [get_expected_duration(act, theta) for act in aoi]

            pool = Pool()
            partial_run = partial(run_matching, mapping, aoi, delta, theta_dist)
            results = list(tqdm(pool.imap_unordered(partial_run, zip(all_acts, all_events)),
                                total=len(all_acts), desc="Processing testcases..."))

            pool.close()
            pool.join()

            wm_correct_total = np.sum([result[0] for result in results])
            wm_missed_total = np.sum([result[1] for result in results])
            wm_incorrect_total = np.sum([result[2] for result in results])
            expected_total = np.sum([result[3] for result in results])
            wm_incorrect_adj_total = np.sum([result[4] for result in results])

            print(f'------{scenario=}, {act_len=}------\n'
                  f'{wm_correct_total=}, {wm_missed_total=}, {wm_incorrect_total=} (adj = {wm_incorrect_adj_total}), '
                  f'{expected_total=}\n')


def start_matching_real(delta, theta):
    mapping = 'data/mappings/k_missing/AL_RS_SC_fail.txt'

    for length in [387, 1494, 2959]:
        aoi = [1, 2, 3, 4, 5, 6]
        activities = f'data/activities/real/{length}.txt'
        events = f'data/events/real/{length}_AL_RS_SC.txt'

        result = run_matching(mapping, aoi, delta, theta, (activities, events))

        wm_correct_total = result[0]
        wm_missed_total = result[1]
        wm_incorrect_total = result[2]
        expected_total = result[3]
        wm_incorrect_adj_total = result[4]

        print(f'scenario=real, {length=}\n'
              f'{wm_correct_total=}, {wm_missed_total=}, {wm_incorrect_total=} (adj = {wm_incorrect_adj_total}), '
              f'{expected_total=}\n')
