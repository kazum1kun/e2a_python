import time
import logging as log

start = time.perf_counter()


def lap(message):
    curr = time.perf_counter()
    log.info(f'{message}\n'
             f'Time elapsed: {curr - start:.2f}')
