# Generate mappings where up to k individual events can be missing
import itertools


def read_file(input_file):
    content = []
    with open(input_file, 'r') as file:
        # Skip the first line
        file.readline()

        # Read in the mappings
        for line in file:
            tokens = line.strip().split(' ')
            content.append(tokens[1:])
    return content


def generate_k_fails(input_file, k, output_file):
    k_missing = []
    original = read_file(input_file)

    # Remove 1...k events from each of the mapping entries
    act_idx = 1
    for entry in original:
        sub_idx = 1
        for i in range(k + 1):
            # We do not want empty lists, however. If the k-value is equal to the length of the token, stop the loop
            if i == len(entry):
                break

            # Remove i events from each entry and add it to the k_missing dictionary
            # This question is equivalent to choosing len(e) - i event combinations from the list w/o replacing
            combinations = itertools.combinations(entry, len(entry) - i)

            # Add the combinations to the list
            for c in combinations:
                k_missing.append([(act_idx, sub_idx), c])
                sub_idx += 1

        act_idx += 1

    # Write the results to file
    with open(output_file, 'w') as file:
        file.write(f'{len(k_missing)}\n')
        for entry in k_missing:
            file.write(f'{entry[0]} {" ".join(entry[1])}\n')

    return k_missing


# Generate mappings where all devices have broken down
def generate_all_dev_fails(input_file, mapping_file, output_file):
    original = read_file(input_file)
    device_events = {}
    modified = []

    # Read the device-event mapping
    with open(mapping_file, 'r') as file:
        for line in file:
            tokens = line.strip().split(' ')
            device_events[tokens[0]] = tokens[1:]

    # Fail the events related to each device by removing them from the original sequence
    seq_num = 1
    for sequence in original:
        subseq_num = 1
        temp = []
        for failed_events in device_events.values():
            result = [event for event in sequence if event not in failed_events]
            if len(result) > 0 and result not in temp:
                modified.append([(seq_num, subseq_num), tuple(result)])
                temp.append(result)
                subseq_num += 1
        seq_num += 1

    with open(output_file, 'w') as file:
        file.write(f'{len(modified)}\n')
        for entry in modified:
            file.write(f'{entry[0][0]} {" ".join(entry[1])}\n')

    return modified


# Generate mappings where one device has broken down
def generate_one_dev_fails(input_file, mapping_file, output_path):
    original = read_file(input_file)
    device_events = {}

    # Read the device-event mapping
    with open(mapping_file, 'r') as file:
        for line in file:
            tokens = line.strip().split(' ')
            device_events[tokens[0]] = tokens[1:]

    # Fail the events related to each device by removing them from the original sequence

    for dev_name, failed_events in device_events.items():
        seq_num = 1
        modified = []
        for sequence in original:
            temp = []
            sub_num = 1
            modified.append([(seq_num, sub_num), sequence])
            temp.append(sequence)
            result = [event for event in sequence if event not in failed_events]
            if len(result) > 0 and result not in temp:
                sub_num += 1
                modified.append([(seq_num, sub_num), tuple(result)])
                temp.append(result)
            seq_num += 1

        with open(f'{output_path}/{dev_name}_fail.txt', 'w') as file:
            file.write(f'{len(modified)}\n')
            for entry in modified:
                file.write(f'{entry[0]} {" ".join(entry[1])}\n')


# Compare the generated sequences and report all sequences with their corresponding identifier(s)
def compare(sequences, output_file=None):
    sequence_mapping = {}

    # Add an identifier to the mapping if the key already exists, otherwise create a new entry
    for sequence in sequences:
        # An entry exists
        if sequence[1] in sequence_mapping:
            sequence_mapping[sequence[1]].append(sequence[0])
        else:
            sequence_mapping[sequence[1]] = [sequence[0]]

    # Sort the mapping-list
    for identifiers in sequence_mapping.values():
        identifiers.sort()

    if output_file:
        with open(output_file, 'w') as file:
            file.write(f'{len(sequence_mapping)}\n')
            # Output according to the number of identifiers attached to a sequence
            for (sequence, identifiers) in sorted(sequence_mapping.items(), key=lambda item: len(item[1]),
                                                  reverse=True):
                file.write(f'{" ".join(sequence)} [len={len(identifiers)}] -> {", ".join(map(str, identifiers))}\n')
    else:
        return sequence_mapping

# result = generate_k_fails('../data/k_missing/original.txt', 1, '../data/k_missing/out_1.txt')
# compare(result, '../data/k_missing/identical_1.txt')
# result = generate_dev_fails('../data/k_missing/original.txt', '../data/device_event/e2a.txt', '../data/k_missing/fails.txt')
# compare(result, '../data/k_missing/identical_fails.txt')
# generate_one_dev_fails('../data/k_missing/original.txt', '../data/device_event/e2a.txt', '../data/mappings/k_missing')
