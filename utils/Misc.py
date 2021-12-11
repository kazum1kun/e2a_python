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
    for folder in folder_names:
        total = 0
        total_mf = 0
        for num in range(events_num):
            with open(f'../data/events/synth/{folder}/{num}.txt') as file:
                length = int(file.readline())
                total += length
            with open(f'../data/events/synth/{folder}/{num}_aqtcfail.txt') as file:
                length = int(file.readline())
                total_mf += length

        avg = total / events_num
        avg_mf = total_mf / events_num

        print(f'Len = {folder}, avg = {avg}, avg_mf = {avg_mf}')


calc_avg_event_length([387, 1494, 2959, 10000, 100000], 10)
