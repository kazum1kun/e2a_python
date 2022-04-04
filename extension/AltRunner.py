import numpy as np

from utils.FileReader import *
from WindowMatch import WindowMatch
from AkMatch import AkMatch
from statistics import NormalDist

# Alternative runner which tests AkMatch vs WindowMatch
act_file = '../data/activities/real/2959.txt'
event_file = '../data/events/real/2959.txt'
map_file = '../data/mappings/with_q.txt'
# act_file = 'data/activities/examples/test.txt'
# event_file = 'data/events/examples/test.txt'
# map_file = 'data/mappings/examples/test.txt'

activities = read_activities(act_file)
events = read_events(event_file)
mappings = read_mappings(map_file)

wm = WindowMatch(events, mappings)
ak = AkMatch(events[:, 1], mappings)


# Get the expected activity duration given a percentile (result centered around mean)
def get_expected_duration(activity, pct):
    # Hard-coded data from experiment data
    mean = [0, 15.927, 43.086, 10.553, 38.152, 10.553, 35.542]
    std = [0, 1.352, 9.363, 1.016, 8.668, 0.973, 12.047]

    # Calculate the inverse cdf given the percentile
    max_duration = NormalDist(mu=mean[activity], sigma=std[activity]).inv_cdf(pct)

    return max_duration


# Get check if the given activity occurred near the given timestamp
# NOTE: delta will be applied both ways (forward and backward), effectively resulting in 2*delta search zone
def check_activity(activity, time, delta):
    activity_idx = np.where(np.logical_and(activities[:, 1] == activity,
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


# Run the matching algorithms
def run_matching(activities_interested, percentile, time_delta):
    # Calculate ni values
    ni = [len(mappings[activity]) - 1 for activity in range(1, len(mappings) + 1)]
    # Make array 1-based
    ni.insert(0, 0)

    wm_correct_total = 0
    ak_correct_total = 0
    wm_incorrect_total = 0
    ak_incorrect_total = 0
    wm_miss_total = 0
    ak_miss_total = 0
    wm_duplicate_total = 0
    ak_duplicate_total = 0
    expected_total = 0

    # Run the matching on each algorithm and add the results found to their corresponding list
    for act in activities_interested:
        duration = get_expected_duration(act, percentile)

        wm_used_idx = set()
        ak_used_idx = set()

        # Statistics
        wm_correct = 0
        ak_correct = 0
        wm_duplicate = 0
        ak_duplicate = 0
        wm_incorrect = 0
        ak_incorrect = 0

        for k in range(1, ni[act]):
            result_wm = wm.find_matches(act, k, duration)
            result_ak = ak.find_matches(act, k)

            # Since we are interested in the start time only, we extract the first entry (corresponding to the
            # start time of activity). To prevent duplicates, we convert the list to set
            start_wm = [x[0] for x in result_wm]
            start_wm = set(start_wm)
            start_ak = [x[1] for x in result_ak]
            start_ak = set(start_ak)

            # Check if the predicted start time correspond to the activity we wanted
            for event_idx in start_wm:
                event_time = events[:, 0][event_idx]

                act_idx = check_activity(act, event_time, time_delta)
                if act_idx:
                    # Check if WindowMatch found a duplicate
                    if act_idx in wm_used_idx:
                        wm_correct += 1
                        wm_duplicate += 1
                    else:
                        wm_correct += 1
                        wm_used_idx.add(act_idx)
                else:
                    wm_incorrect += 1

            for event_idx in start_ak:
                event_time = events[:, 0][event_idx]

                act_idx = check_activity(act, event_time, time_delta)
                if act_idx:
                    if act_idx in ak_used_idx:
                        ak_correct += 1
                        ak_duplicate += 1
                    else:
                        ak_correct += 1
                        ak_used_idx.add(act_idx)
                else:
                    ak_incorrect += 1

        # Expected # of occurrences of the activity
        expected = np.count_nonzero(activities[:, 1] == act)

        wm_miss = expected - (wm_correct - wm_duplicate)
        ak_miss = expected - (ak_correct - ak_duplicate)

        wm_correct_total += wm_correct
        ak_correct_total += ak_correct
        wm_incorrect_total += wm_incorrect
        ak_incorrect_total += ak_incorrect
        wm_miss_total += wm_miss
        ak_miss_total += ak_miss
        ak_duplicate_total += ak_duplicate
        wm_duplicate_total += wm_duplicate
        expected_total += expected

    print(f'{percentile=}')
    print(f'{wm_correct_total=}, {wm_duplicate_total=}, {wm_miss_total=}, {wm_incorrect_total=}\n'
          f'{ak_correct_total=}, {ak_duplicate_total=}, {ak_miss_total=}, {ak_incorrect_total=}\n'
          f'{expected_total=}')


run_matching([1, 2, 3, 4, 5, 6], percentile=0.999, time_delta=5)
