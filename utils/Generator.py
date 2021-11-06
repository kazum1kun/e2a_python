from FileReader import read_mappings

def main(file, limit):
    all_tokens = []
    scanned = set()

    with open(file, 'r') as infile:
        infile.readline()

        for line in infile:
            tokens = line.strip().split(' ')

            if int(tokens[0]) >= limit:
                break

            if not tokens[0] in scanned:
                all_tokens.extend(tokens[1:])
                scanned.add(tokens[0])

    counter = 1
    for token in all_tokens:
        print(f'{counter} {token}')
        counter += 1


if __name__ == '__main__':
    main('../data/mappings/mappings-reduced.txt', 16)
