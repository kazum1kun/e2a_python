from FileReader import read_mappings_0based


def verify_distribution():
    from FileReader import read_activities
    from collections import Counter

    s1 = read_activities('../data/activities/real/activities-real2959.txt')[1:, 1]
    s2 = read_activities('../data/activities/activities-synthtest10000.txt')[1:, 1]

    c1 = Counter(s1)
    c2 = Counter(s2)

    res1 = [(i, c1[i] / len(s1) * 100.0) for i, count in c1.most_common()]
    res2 = [(i, c2[i] / len(s2) * 100.0) for i, count in c2.most_common()]

    print(res1)
    print(res2)


def calc_avg_event_length(folder_names, events_num):
    import numpy as np
    for folder in folder_names:
        len_normal = []
        len_mf = []
        for num in range(events_num):
            with open(f'../data/events/synth/{folder}/{num}.txt') as file:
                len_normal.append(int(file.readline()))
            with open(f'../data/events/synth/{folder}/{num}_aqtcfail.txt') as file:
                len_mf.append(int(file.readline()))

        print(f'Len = {folder}, avg = {np.mean(len_normal)}, max = {np.max(len_normal)}, min = {np.min(len_normal)}\n'
              f'\tavg_mf = {np.mean(len_mf)}, max_mf = {np.max(len_mf)}, min_mf = {np.min(len_mf)}')


# Check how many event sequences are identical in the given mapping
def find_identical_sequences(mapping):
    act_sequences = {}
    act_idx = 1
    for act, sequences in mapping.items():
        ss_idx = 1
        for subsequence in sequences:
            ss = tuple(subsequence)
            if ss in act_sequences:
                act_sequences[ss].append((act_idx, ss_idx))
            else:
                act_sequences[ss] = [(act_idx, ss_idx)]
            ss_idx += 1
        act_idx += 1

    return act_sequences


for device in ['AL', 'AQ', 'DW', 'KM', 'RC', 'RD', 'RS', 'SC', 'TB', 'TC', 'TP', 'AL_RS', 'AL_RS_SC']:
    file = f'../data/mappings/k_missing/{device}_fail.txt'
    results = find_identical_sequences(read_mappings_0based(file))
    # Count # of occurrences of activities for each sequence
    total_count = 0
    counter = {}
    for sequence, activities in results.items():
        activities_count = len(activities)
        if activities_count in counter:
            counter[activities_count] += 1
        else:
            counter[activities_count] = 1
        total_count += activities_count

    print(f'Device: {device}')
    print(f'Avg length = {total_count / len(results)}')
    print(counter)
