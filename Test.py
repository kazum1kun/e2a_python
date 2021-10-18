from AkMatch import AkMatch


def main():
    E = ['a', 'b', 'a', 'b', 'd', 'c', 'a', 'b', 'c']
    akMatch = AkMatch()
    L = akMatch.find_matches(E, 2, 1)
    print(L)


if __name__ == '__main__':
    main()