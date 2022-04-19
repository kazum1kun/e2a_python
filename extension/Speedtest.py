import time

from tqdm import tqdm

from Runner import run_e2a

progress_bar = tqdm(range(40), desc='Processing test cases...')
scenario = 'none'

mapping_file = f'data/mappings/k_missing/{scenario}_fail.txt'

for act_len in [387, 1494, 2959]:
    time_multi = []
    time_single = []

    for itr in range(10):
        activity_file = f'data/activities/synth/{act_len}_rand/{itr}_{scenario}.txt'
        event_file = f'data/events/synth/{act_len}_rand/{itr}_{scenario}.txt'

        # Multi-threaded version
        start = time.time()
        run_e2a(activity_file, event_file, mapping_file, aoi=None)
        end = time.time()
        time_multi.append(end - start)

        # Single-threaded version
        start = time.time()
        run_e2a(activity_file, event_file, mapping_file, aoi=None, method='mono')
        end = time.time()
        time_single.append(end - start)

        progress_bar.update(1)

    print(f'act_len = {act_len}, multi avg={sum(time_multi) / len(time_multi)}, '
          f'single avg={sum(time_single) / len(time_single)}')
