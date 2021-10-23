import time

start = time.perf_counter()


def lap(message):
    curr = time.perf_counter()
    print(f'{message}\n'
          f'Time elapsed: {curr - start:.2f}')
