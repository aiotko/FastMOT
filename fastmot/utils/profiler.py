import time
from collections import Counter


class Profiler:
    __call_count = Counter()
    __time_elapsed = Counter()

    def __init__(self, stream_num, name, aggregate=False):
        self.name = name
        self.stream_num = stream_num
        if not aggregate:
            Profiler.__call_count[(stream_num, self.name)] += 1

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
        Profiler.__time_elapsed[(self.stream_num, self.name)] += self.duration

    @classmethod
    def reset(cls):
        cls.__call_count.clear()
        cls.__time_elapsed.clear()

    @classmethod
    def get_millis(cls, stream_num, name):
        return cls.__time_elapsed[(stream_num, name)] * 1000

    @classmethod
    def get_avg_millis(cls, stream_num, name):
        call_count = cls.__call_count[(stream_num, name)]
        if call_count == 0:
            return 0.
        return cls.__time_elapsed[(stream_num, name)] * 1000 / call_count