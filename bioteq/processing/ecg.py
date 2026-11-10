from statistics import mean
from bioteq.processing.base import Processor
from bioteq.schemas.outputs import CardiovascularMetrics

class ECGProcessor(Processor):
    def __init__(self, rest_interval_ms=None):
        self.rest_interval_ms = rest_interval_ms
        if rest_interval_ms is not None and not 0 <= rest_interval_ms[0] < rest_interval_ms[1]:
            raise ValueError("Rest interval must have increasing nonnegative timestamps")

    def process(self, readings):
        rates = [(r.timestamp_ms, r.value) for r in readings if r.unit == "bpm" and r.value is not None]
        if not rates:
            peaks = sorted({r.timestamp_ms for r in readings if r.metadata.get("r_peak") is True and r.value is not None})
            rates = [(b, 60000/(b-a)) for a, b in zip(peaks, peaks[1:])]
        if not rates:
            return CardiovascularMetrics(quality="raw_ecg_unsupported" if readings else "unavailable")
        recovery = None
        if self.rest_interval_ms is not None:
            start, end = self.rest_interval_ms
            # Explicit rest observations only; no extrapolation to an unobserved endpoint.
            rest = sorted((t, v) for t, v in rates if start <= t <= end)
            if len(rest) >= 2:
                recovery = rest[0][1] - rest[-1][1]
        return CardiovascularMetrics(heart_rate_bpm=mean(v for _, v in rates),
            heart_rate_recovery_bpm=recovery, quality="ok")
