import random

from FileReader import read_mappings_0based


def main(file, number, extension='', generate_partial=False):
    mappings = read_mappings_0based(file)
    n = len(mappings)
    activity_output = [0]
    events_output = [0]
    time_counter = 0

    # Generate a random sequence of activities
    rand_activities = [random.randint(1, n - 1) for _ in range(number)]

    for activity in rand_activities:
        activity_output.append(f'{time_counter} {activity}')
        # Generate a partial sequence if the flag is on
        if generate_partial:
            ni = len(mappings[activity])
            sub_sequence = random.randint(0, ni - 1)
        else:
            sub_sequence = 0

        sequence = mappings[activity][sub_sequence]

        for i in range(len(sequence)):
            events_output.append(f'{time_counter} {sequence[i]}')
            time_counter += 1

    activity_output[0] = str(len(activity_output) - 1)
    events_output[0] = str(len(events_output) - 1)

    with open(f'activities{extension}.txt', 'w') as activity_file:
        for e in activity_output:
            activity_file.write(e + '\n')

    with open(f'events{extension}.txt', 'w') as events_file:
        for e in events_output:
            events_file.write(e + '\n')


if __name__ == '__main__':
    main('../data/mappings/mappings-reduced.txt', 5)
