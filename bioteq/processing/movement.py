from collections import defaultdict
from math import sqrt
from statistics import mean, pstdev
from scipy.signal import find_peaks
from bioteq.processing.base import Processor
from bioteq.schemas.outputs import MovementMetrics

class MovementProcessor(Processor):
    def __init__(self, prominence=0.5, min_spacing_ms=1000, smoothing_samples=1):
        if prominence < 0 or min_spacing_ms <= 0 or smoothing_samples < 1:
            raise ValueError("Invalid movement settings")
        self.prominence, self.min_spacing_ms, self.smoothing_samples = prominence, min_spacing_ms, smoothing_samples

    def process(self, readings):
        frames = defaultdict(dict)
        for r in readings:
            if r.value is not None:
                frames[r.timestamp_ms][r.axis] = r.value
        frames = sorted((t, sqrt(sum(v[a]**2 for a in ("x", "y", "z")))) for t, v in frames.items() if all(a in v for a in ("x", "y", "z")))
        if len(frames) < 3:
            return MovementMetrics(quality="insufficient_complete_xyz")
        times, values = zip(*frames)
        n = self.smoothing_samples
        smooth = [mean(values[max(0, i-n//2):min(len(values), i+n//2+1)]) for i in range(len(values))]
        candidates, props = find_peaks(smooth, prominence=self.prominence)
        # Suppress peaks using actual timestamps, not a presumed sample rate.
        selected = []
        for i in sorted(range(len(candidates)), key=lambda j: props["prominences"][j], reverse=True):
            t = times[candidates[i]]
            if all(abs(t-other) >= self.min_spacing_ms for other in selected):
                selected.append(t)
        selected.sort()
        durations = [(b-a)/1000 for a, b in zip(selected, selected[1:])]
        return MovementMetrics(rep_count=len(selected),
            average_rep_duration_s=mean(durations) if durations else None,
            movement_variability_cv=pstdev(durations)/mean(durations) if durations else None,
            mean_acceleration_m_s2=mean(values), quality="ok" if len(frames) == len({r.timestamp_ms for r in readings}) else "incomplete_xyz_dropped")
