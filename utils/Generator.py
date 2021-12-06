import logging
import random

from FileReader import read_mappings_0based, read_device_event


def generate_testcase(normal_file, failed_file, number, extension='', generate_partial=False, generate_fail=False):
    mappings_normal = read_mappings_0based(normal_file)
    mappings_failed = read_mappings_0based(failed_file)
    n = len(mappings_normal)
    activity_output = [0]
    events_output = [0]
    events_failed_output = [0]
    time_counter = 1
    time_counter_failed = 1

    # Generate a random sequence of activities
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

        if generate_fail and len(activity_output) / number > 0.5:
            ni = len(mappings_failed[activity])
            sub_sequence_f = random.randint(0, ni - 1)
            fail_sequence = mappings_failed[activity][sub_sequence_f]

            for i in range(len(fail_sequence)):
                events_failed_output.append(f'{time_counter_failed} {fail_sequence[i]}')
                time_counter_failed += 1

        elif generate_fail:
            for i in range(len(sequence)):
                events_failed_output.append(f'{time_counter_failed} {sequence[i]}')
                time_counter_failed += 1

        for i in range(len(sequence)):
            events_output.append(f'{time_counter} {sequence[i]}')
            time_counter += 1

    activity_output[0] = str(len(activity_output) - 1)
    events_output[0] = str(len(events_output) - 1)
    events_failed_output[0] = str(len(events_failed_output) - 1)

    with open(f'../data/activities/activities{extension}.txt', 'w') as activity_file:
        for e in activity_output:
            activity_file.write(e + '\n')

    with open(f'../data/events/events{extension}.txt', 'w') as events_file:
        for e in events_output:
            events_file.write(e + '\n')

    if generate_fail:
        with open(f'../data/activities/activities{extension}_aqtcfail.txt', 'w') as activity_file:
            for e in activity_output:
                activity_file.write(e + '\n')
        with open(f'../data/events/events{extension}_aqtcfail.txt', 'w') as events_file:
            for e in events_failed_output:
                events_file.write(e + '\n')


def generate_mappings(map_file, de_file, extension='', device_failures=()):
    mappings = read_mappings_0based(map_file)
    device_event = read_device_event(de_file)

    # If device failure is on, remove the events that are related to the device
    if device_failures:
        removed_events = [event for dev in device_failures for event in device_event[dev]]
        new_mapping = ['0\n']

        for activity, events_list in mappings.items():
            new_list = [[event for event in events if event not in removed_events] for events in events_list]
            # Remove duplicates
            mappings[activity].clear()
            [mappings[activity].append(events) for events in new_list if events not in mappings[activity]]

            # Ignore the first one, not usable
            if activity != 0:
                for pattern in mappings[activity]:
                    new_mapping.append(f'{activity} {" ".join(pattern)}\n')

        # Update the counter on top of the file
        new_mapping[0] = f'{len(new_mapping) - 1}\n'

        # Write the updated mapping back to the file
        with open(f'../data/mappings/mappings{extension}.txt', 'w') as mapping_file:
            for line in new_mapping:
                mapping_file.write(line)
    else:
        logging.warning('No device specified to fail, exiting...')


if __name__ == '__main__':
    generate_testcase(normal_file='../data/mappings/mappings-q.txt',
                      failed_file='../data/mappings/mappings-synth_aqtcfail.txt',
                      number=100000, extension='-synth100000', generate_partial=True, generate_fail=True)

    # generate_mappings('../data/mappings/mappings-q.txt', '../data/device_event/e2a.txt',
    #                   extension='-synth_aqtcfail', device_failures=('AQ', 'TC'))
