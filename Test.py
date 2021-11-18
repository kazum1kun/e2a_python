import logging as log
import editdistance
from tqdm import tqdm

from SLN import SLN
from utils import Timer
from utils.FileReader import *
from OMatch import OMatch


def main():
    log.basicConfig(format='%(message)s', level=log.INFO)

    file_ext = '-revised'
    mapping_ext = '-q'
    activities = read_activities(f'data/activities/activities{file_ext}.txt')
    events = read_events(f'data/events/events{file_ext}.txt')
    mappings = read_mappings(f'data/mappings/mappings{mapping_ext}.txt')
    Timer.lap('Text read finished')

    n = len(mappings)
    # Calculate ni values and the N value
    ni = [len(mappings[activity]) - 1 for activity in range(1, n + 1)]
    # Make array 1-based
    ni.insert(0, 0)
    ni = ni

    oMatch = OMatch(events[:, 1], mappings, n, ni)
    Timer.lap('OMatch initialization done')
    segments, intervals = split_events(oMatch.M, events[:, 1])
    Timer.lap('Segmentation finished')

    f_opt_total = 0
    counter = 0
    Aw_all = []

    # Much faster version that separates input into different segments. Accuracy may suffer for a little bit.
    segments_bar = tqdm(segments)
    for segment, interval in zip(segments_bar, intervals):
        counter += 1
        segments_bar.set_description(f'Processing segments, current f_opt={f_opt_total}')
        sln = SLN(mappings, segment, ni)
        _, Aw, f_opt = sln.sln_nd(1)
        f_opt_total += f_opt
        Aw_all.extend(Aw[1:])

    # Non-separating version
    # sln = SLN(mappings, events[:, 1], ni)
    # _, Aw, f_opt = sln.sln_nd(1)
    # f_opt_total += f_opt
    # Aw_all.extend(Aw[1:])

    Timer.lap('Finished!')
    print(f'\nThe final f_opt={f_opt_total}')
    print('\nThe final activities calculated is')
    activities_calc = [x[0] for x in Aw_all]
    print(activities_calc)
    diff = editdistance.eval(activities_calc, activities[1:, 1])
    print(f'\nThe calculated activities are {diff} edits away from the actual activities')


# Splits the input events into smaller blocks based on the parameter and make sure none of the events are interrupted
def split_events(M, E):
    # Calculate deg, which is then number of crosses a split point makes to matches
    deg = np.array([np.count_nonzero(np.logical_and(M['alpha'] <= i, M['beta'] > i))
                    for i in range(1, len(E))])
    # If any split point has zero deg, it is a suitable candidate to be split
    candidates = np.argwhere(deg == 0).flatten() + 1

    # Create intervals based on the candidates
    intervals = [(candidates[i] + 1, candidates[i+1]) for i in range(0, len(candidates) - 1)]
    intervals.insert(0, (1, candidates[0]))

    # Create segments of the event using the interval
    segments = [np.insert(E[i[0]:i[1] + 1].copy(), 0, 0) for i in intervals]

    return segments, intervals


if __name__ == '__main__':
    main()
