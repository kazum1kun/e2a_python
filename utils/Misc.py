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
