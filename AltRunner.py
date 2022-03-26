import numpy as np

from utils.FileReader import *
from WindowMatch import WindowMatch
from AkMatch import AkMatch
from statistics import NormalDist


# Alternative runner which tests AkMatch vs WindowMatch
act_file = 'data/activities/real/2959.txt'
event_file = 'data/events/real/2959.txt'
map_file = 'data/mappings/with_q.txt'
# act_file = 'data/activities/examples/test.txt'
# event_file = 'data/events/examples/test.txt'
# map_file = 'data/mappings/examples/test.txt'


# Get the expected activity duration given a percentile (result centered around mean)
def get_expected_duration(activity, pct):
    # Hard-coded data from experiment data
    mean = [0, 15.927, 43.086, 10.553, 38.152, 10.553, 35.542]
    std = [0, 1.352, 9.363, 1.016, 8.668, 0.973, 12.047]

    # Calculate the inverse cdf given the percentile
    max_duration = NormalDist(mu=mean[activity], sigma=std[activity]).inv_cdf(pct)

    return max_duration


# Run the matching algorithms
def run_matching(activities_interested, percentile, confidence):
    activities = read_activities(act_file)
    events = read_events(event_file)
    mappings = read_mappings(map_file)

    wm = WindowMatch(events, mappings)
    ak = AkMatch(events[:, 1], mappings)

    wm_correct_total = 0
    ak_correct_total = 0
    wm_incorrect_total = 0
    ak_incorrect_total = 0
    wm_miss_total = 0
    ak_miss_total = 0
    wm_duplicate_total = 0
    expected_total = 0

    # Run the matching on each algorithm and add the results found to their corresponding list
    for act in activities_interested:
        duration = get_expected_duration(act, percentile)
        result_wm = wm.find_matches(act, duration, confidence)
        result_ak = ak.find_matches(act, 1)

        # Since we are interested in the start time only, we extract the first entry (corresponding to the start time
        # of activity). To prevent duplicates, we convert the list to set
        start_wm = [x[0] for x in result_wm]
        start_wm = set(start_wm)
        start_ak = [x[1] for x in result_ak]
        start_ak = set(start_ak)

        used_idx = set()

        # Statistics
        wm_correct = 0
        ak_correct = 0
        wm_duplicate = 0
        wm_incorrect = 0
        ak_incorrect = 0
        # Expected # of occurrences of the activity
        expected = np.count_nonzero(activities[:, 1] == act)

        # Check if the predicted start time correspond to the activity we wanted
        for event_idx in start_wm:
            event_time = events[:, 0][event_idx]
            acts_before_event = np.where(activities[:, 0] < event_time)
            last_act_idx = acts_before_event[0][-1]
            last_act = activities[:, 1][last_act_idx]

            if last_act == act:
                # Check if WindowMatch found a duplicate
                if last_act_idx in used_idx:
                    wm_correct += 1
                    wm_duplicate += 1
                else:
                    wm_correct += 1
                    used_idx.add(last_act_idx)
            else:
                wm_incorrect += 1

        for event_idx in start_ak:
            event_time = events[:, 0][event_idx]
            acts_before_event = np.where(activities[:, 0] < event_time)
            last_act_idx = acts_before_event[0][-1]
            last_act = activities[:, 1][last_act_idx]

            if last_act == act:
                ak_correct += 1
            else:
                ak_incorrect += 1

        wm_miss = expected - (wm_correct - wm_duplicate)
        ak_miss = expected - ak_correct

        wm_correct_total += wm_correct
        ak_correct_total += ak_correct
        wm_incorrect_total += wm_incorrect
        ak_incorrect_total += ak_incorrect
        wm_miss_total += wm_miss
        ak_miss_total += ak_miss
        wm_duplicate_total += wm_duplicate
        expected_total += expected

    print(f'{wm_correct_total=}, {wm_incorrect_total=}, {wm_miss_total=}, {wm_duplicate_total=}\n'
          f'{ak_correct_total=}, {ak_incorrect_total=}, {ak_miss_total=}\n'
          f'{expected_total=}')


run_matching([1, 2, 3, 4, 5, 6], percentile=0.8, confidence=0.8)
