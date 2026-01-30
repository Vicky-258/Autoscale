from collections import deque

class SlidingWindow:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def add(self, value):
        self.buffer.append(value)

    def get(self):
        return list(self.buffer)

    def is_full(self):
        return len(self.buffer) == self.size
