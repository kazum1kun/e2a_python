import json
import os.path
import itertools
from utils.FileReader import *
from utils.FileWriter import write_list_txt


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


def generate_report(output_folder, suffixes, lengths):
    for suffix in suffixes:
        for length in lengths:
            diff_strict_total = 0
            diff_lax_total = 0
            diff_orig_total = 0

            for i in range(1000000):
                fp = f'{output_folder}/{length}/{i}_{suffix}_new.json'

                if os.path.exists(fp):
                    with open(fp) as res_file:
                        results = json.load(res_file)
                        diff_strict_total += results["diff_strict"]
                        diff_lax_total += results["diff_lax"]
                        diff_orig_total += results["diff_original"]

                else:
                    print(f'Results for {suffix} Length {length}\n'
                          f'Avg Strict Diff: {diff_strict_total / i}\n'
                          f'Avg Lax Diff: {diff_lax_total / i}\n'
                          f'Avg Orig Diff: {diff_orig_total / i}\n'
                          f'Sample size: {i}\n')
                    break


# generate_report('../data/output/synth', ['none', 'RS', 'AL_RS_SC'], [387, 1494, 2959, 10000])
# Remove events related to certain device(s) from an activity sequence
def remove_events(act_file, devices, out_path):
    events = read_events(act_file)
    device_event = read_device_event('data/device_event/e2a.txt')
    removed_events = [ord(event) for device in devices for event in device_event[device]]
    new_events = []

    for event in events:
        if event[1] in removed_events:
            continue
        new_events.append(f'{event[0]} {chr(int(event[1]))}')

    new_events[0] = f'{len(new_events)}'

    write_list_txt(new_events, out_path)

