import logging
import os.path
import random

from FileReader import read_mappings_0based, read_device_event, read_activities


# Generate test cases using the preferences specified by the user
def generate_testcase(normal_file, failed_file, number,
                      generate_partial=False, prob_src=None, rand_seed=None,
                      folder='.', filename='default'):
    # If seed is set, then generate patterns according to this seed
    if rand_seed:
        random.seed(rand_seed)
    mappings_normal = read_mappings_0based(normal_file)
    if failed_file:
        mappings_failed = read_mappings_0based(failed_file)
    n = len(mappings_normal)
    activity_output = [0]
    events_output = [0]
    events_failed_output = [0]
    time_counter = 1
    time_counter_failed = 1

    # Generate a random sequence of activities
    # If prob_src is set, then generate the activity sequence based on the source probability distribution
    # (rather than completely random)
    if prob_src:
        src_activities = read_activities(prob_src)[1:, 1]
        rand_activities = [random.choice(src_activities) for _ in range(number)]
    else:
        rand_activities = [random.randint(1, n - 1) for _ in range(number)]

    for activity in rand_activities:

        activity_output.append(f'{time_counter} {activity}')
        # Generate a partial sequence if the flag is on
        if generate_partial:
            ni = len(mappings_normal[activity])
            sub_sequence = random.randint(0, ni - 1)
        else:
            sub_sequence = 0

        # If generate_fail is on, in addition to normal file it also generate a file where the device malfunctions
        # after half of the activity
        sequence = mappings_normal[activity][sub_sequence]

        if failed_file and len(activity_output) / number > 0.5:
            ni = len(mappings_failed[activity])
            sub_sequence_f = random.randint(0, ni - 1)
            fail_sequence = mappings_failed[activity][sub_sequence_f]

            for i in range(len(fail_sequence)):
                events_failed_output.append(f'{time_counter_failed} {fail_sequence[i]}')
                time_counter_failed += 1

        elif failed_file:
            for i in range(len(sequence)):
                events_failed_output.append(f'{time_counter_failed} {sequence[i]}')
                time_counter_failed += 1

        for i in range(len(sequence)):
            events_output.append(f'{time_counter} {sequence[i]}')
            time_counter += 1

    activity_output[0] = str(len(activity_output) - 1)
    events_output[0] = str(len(events_output) - 1)
    events_failed_output[0] = str(len(events_failed_output) - 1)

    activities_folder = f'../data/activities/synth/{folder}'
    events_folder = f'../data/events/synth/{folder}'

    # Make sure the output folder exists, if not create it
    if not os.path.exists(activities_folder):
        os.mkdir(activities_folder)
    if not os.path.exists(events_folder):
        os.mkdir(events_folder)

    with open(f'{activities_folder}/{filename}.txt', 'w') as activity_file:
        for e in activity_output:
            activity_file.write(e + '\n')

    with open(f'{events_folder}/{filename}.txt', 'w') as events_file:
        for e in events_output:
            events_file.write(e + '\n')

    if failed_file:
        with open(f'{activities_folder}/{filename}_aqtcfail.txt', 'w') as activity_file:
            for e in activity_output:
                activity_file.write(e + '\n')
        with open(f'{events_folder}/{filename}_aqtcfail.txt', 'w') as events_file:
            for e in events_failed_output:
                events_file.write(e + '\n')


# Generate mappings with device failures, that is, events related to failed devices are removed
def generate_mappings(map_file, de_file, extension='', device_failures=()):
    mappings_failed = read_mappings_0based(map_file)
    mappings_combined = read_mappings_0based(map_file)
    device_event = read_device_event(de_file)

    # If device failure is on, remove the events that are related to the device
    if device_failures:
        removed_events = [event for dev in device_failures for event in device_event[dev]]
        failed_mapping = ['0\n']
        combined_mapping = ['0\n']

        for activity, events_list in mappings_failed.items():
            new_list = [[event for event in events if event not in removed_events] for events in events_list]
            # Remove duplicates
            mappings_failed[activity].clear()
            [mappings_failed[activity].append(events) for events in new_list if events not in mappings_failed[activity]]
            [mappings_combined[activity].append(events) for events in new_list if
             events not in mappings_combined[activity]]

            # Ignore the first one, not usable
            if activity != 0:
                sub = 1
                for pattern in mappings_failed[activity]:
                    failed_mapping.append(f'{(activity, sub)} {" ".join(pattern)}\n')
                    sub += 1
                sub = 1
                for pattern in mappings_combined[activity]:
                    combined_mapping.append(f'{(activity, sub)} {" ".join(pattern)}\n')
                    sub += 1

        # Update the counter on top of the file
        failed_mapping[0] = f'{len(failed_mapping) - 1}\n'
        combined_mapping[0] = f'{len(combined_mapping) - 1}\n'

        # Write the updated mapping back to the file
        # with open(f'../data/mappings/k_missing/mappings{extension}.txt', 'w') as mapping_file:
        #     for line in failed_mapping:
        #         mapping_file.write(line)
        with open(f'../data/mappings/k_missing/{extension}_fail.txt', 'w') as mapping_file:
            for line in combined_mapping:
                mapping_file.write(line)
    else:
        logging.warning('No device specified to fail, exiting...')


if __name__ == '__main__':
    # for act_len in [387, 1494, 2959, 10000, 30000]:
    #     for itr in range(100):
    #         seed = act_len * 10000 + itr
    #         generate_testcase(normal_file='../data/mappings/with_q.txt',
    #                           failed_file='../data/mappings/synth_aqtcfail.txt',
    #                           number=act_len, generate_partial=True, generate_fail=True,
    #                           prob_src='../data/activities/real/2959.txt', rand_seed=seed,
    #                           folder=str(act_len), filename=str(itr))

    # generate_mappings('../data/mappings/k_missing/AL_RS_SC_fail.txt', '../data/device_event/e2a.txt',
    #                   extension='AL_AQ_RS_SC', device_failures=('AQ',))
    # for device in ['AL', 'AQ', 'DW', 'KM', 'RC', 'RD', 'RS', 'SC', 'TB', 'TC', 'TP', 'AL_RS', 'AL_RS_SC']:
    #     for length in [10, 387, 1494, 2959]:
    #         generate_testcase(normal_file=f'../data/mappings/k_missing/{device}_fail.txt',
    #                           failed_file=None,
    #                           number=length, generate_partial=True,
    #                           prob_src=None, rand_seed=None,
    #                           folder='dev_fails', filename=f'{device}_{length}')
    #
    for scenario in ['none', 'RS', 'AL_RS_SC']:
        for act_len in [387, 1494, 2959, 10000, 30000]:
            for itr in range(100):
                seed = act_len * 10000 + itr
                generate_testcase(normal_file=f'../data/mappings/k_missing/{scenario}_fail.txt',
                                  failed_file=None,
                                  number=act_len, generate_partial=True,
                                  prob_src='../data/activities/real/2959.txt', rand_seed=seed,
                                  folder=str(act_len), filename=f'{itr}_{scenario}')

    for act_len in [10000]:
        for itr in range(100):
            seed = act_len * 10000 + itr
            generate_testcase(normal_file='../data/mappings/with_q.txt',
                              failed_file='../data/mappings/synth_aqtcfail.txt',
                              number=act_len, generate_partial=True, generate_fail=True,
                              prob_src='../data/activities/real/2959.txt', rand_seed=seed,
                              folder=str(act_len), filename=str(itr))
