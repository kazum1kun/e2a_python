def convert_file(file_name, out_name):
    with open(file_name, 'r') as file:
        activities = file.readline()
        tokens = activities.split(', ')

    with open(out_name, 'w') as out:
        for token in tokens:
            out.write(token + '\n')

    print('done')


def print_sequence(activities):
    full = []
    for i in activities[1:, 1]:
        full.append(i)
    print(full)


if __name__ == '__main__':
    convert_file('../data/results/activities-calculated-new.txt', '../data/results/activities-calculated-new-sep.txt')
