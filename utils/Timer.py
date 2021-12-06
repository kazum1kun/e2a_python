import logging as log
import time


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def lap(self, message):
        curr = time.perf_counter()
        log.info(f'{message}\n'
                 f'Time elapsed: {curr - self.start:.5f}')
