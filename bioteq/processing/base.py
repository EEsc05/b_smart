from abc import ABC, abstractmethod
from collections import defaultdict

class Processor(ABC):
    @abstractmethod
    def process(self, readings):
        """Return derived features; timestamps are milliseconds."""

def muscle_groups(readings):
    groups = defaultdict(list)
    for r in readings:
        if r.value is not None:
            groups[r.muscle_site, r.side].append(r)
    return groups
