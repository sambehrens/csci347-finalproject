from heapq import heappush, heappop


class PriorityQueue:
    def __init__(self):
        self.data = []
        self.removed_entries = set()

    def pop(self):
        while len(self.data):
            popped = heappop(self.data)
            if popped in self.removed_entries:
                self.removed_entries.remove(popped)
                continue
            return popped

    def remove(self, item):
        self.removed_entries.add(item)

    def push(self, item):
        heappush(self.data, item)

    def __iadd__(self, other):
        self.push(other)
        return self

    def __len__(self):
        return len(self.data) - len(self.removed_entries)
