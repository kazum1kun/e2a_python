import glob

from utils.FileReader import *
from WindowMatch import WindowMatch
from AkMatch import AkMatch
from statistics import NormalDist


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

    for act in aoi:
        for time in times:
            act_idx = check_activity(activities, act, time, delta)
            if act_idx:
                # Check if WindowMatch found a duplicate
                if act_idx in used_idx:
                    correct += 1
                    duplicate += 1
                else:
                    correct += 1
                    used_idx.add(act_idx)
            else:
                incorrect += 1

    return correct, incorrect, duplicate


# Run the matching algorithms
def run_matching(act_file, event_file, map_file, aoi, delta):
    # Alternative runner which tests AkMatch vs WindowMatch
    activities = read_activities(act_file)
    events = read_events(event_file)
    mappings = read_mappings(map_file)

    wm = WindowMatch(events, mappings)
    ak = AkMatch(events, mappings)

    # Calculate ni values
    ni = [len(mappings[activity]) - 1 for activity in range(1, len(mappings) + 1)]
    # Make array 1-based
    ni.insert(0, 0)

    # Using set to prevent dups
    start_wm = set()
    start_ak = set()

    for act in aoi:
        duration = get_expected_duration(act, 0.9999)
        for k in range(1, ni[act]):
            # Run the matching on each algorithm and add the timestamps of each match found
            result_wm = wm.find_matches(act, k, duration)
            result_ak = ak.find_matches(act, k)
            start_wm.add([x[1] for x in result_wm])
            start_ak.add([x[1] for x in result_ak])

    wm_correct, wm_incorrect, wm_duplicate = check_input(activities, start_wm, aoi, delta)
    ak_correct, ak_incorrect, ak_duplicate = check_input(activities, start_ak, aoi, delta)
    expected = np.count_nonzero(activities[:, 1] in aoi)

    wm_miss = expected - (wm_correct - wm_duplicate)
    ak_miss = expected - (ak_correct - ak_duplicate)

    print(f'{wm_correct=}, {wm_duplicate=}, {wm_miss=}, {wm_incorrect=}\n'
          f'{ak_correct=}, {ak_duplicate=}, {ak_miss=}, {ak_incorrect=}\n'
          f'{expected=}')


# Get the expected activity duration given a percentile (result centered around mean)
def get_expected_duration(activity, pct):
    # Hard-coded data from experiment data
    mean = [0, 15.927, 43.086, 10.553, 38.152, 10.553, 35.542]
    std = [0, 1.352, 9.363, 1.016, 8.668, 0.973, 12.047]

    # Calculate the inverse cdf given the percentile
    max_duration = NormalDist(mu=mean[activity], sigma=std[activity]).inv_cdf(pct)

    return max_duration


# Verify whether specific activities did occur on the timestamps specified in the output file
def verify_matches(output_file, activity_file, aoi):
    times = read_json(output_file)["all_timestamps"]
    activities = read_activities(activity_file)

    return check_input(activities, times, aoi, 5)


def run_verifier():
    for scenario in ['none', 'RS', 'AL_RS_SC']:
        for act_len in [387, 1494, 2959, 10000]:
            all_output = glob.glob(f'data/output/synth/{act_len}/*_{scenario}_new.json')
            all_acts = glob.glob(f'data/activities/synth/{act_len}/*_{scenario}.txt')

            correct_total = 0
            incorrect_total = 0
            duplicate_total = 0

            for output, act in zip(all_output, all_acts):
                correct, incorrect, duplicate = verify_matches(output, act, [1, 2, 3, 4, 5, 6])
                correct_total += correct
                incorrect_total += incorrect
                duplicate_total += duplicate

            print(f'{scenario=}, {act_len=}, correct={correct_total}, total={correct_total + incorrect_total}')
