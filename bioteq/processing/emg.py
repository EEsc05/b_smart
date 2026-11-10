from math import sqrt
from statistics import mean
from bioteq.processing.base import Processor, muscle_groups
from bioteq.schemas.outputs import MuscleMetrics, BilateralMetrics

class EMGProcessor(Processor):
    def __init__(self, references=None):
        self.references = references or {}
        if any(v <= 0 for v in self.references.values()):
            raise ValueError("RMS references must be positive")

    def process(self, readings):
        result = []
        for (site, side), rows in sorted(muscle_groups(readings).items(), key=lambda x: str(x[0])):
            rms = sqrt(mean(r.value ** 2 for r in rows))
            reference = self.references.get((site, side))
            result.append(MuscleMetrics(muscle_site=site, side=side, rms_mv=rms,
                mean_absolute_mv=mean(abs(r.value) for r in rows),
                normalized_activation_pct=100*rms/reference if reference else None))
        return result

    @staticmethod
    def bilateral(muscles):
        result = []
        for site in sorted({m.muscle_site for m in muscles}):
            pair = {m.side: m.rms_mv for m in muscles if m.muscle_site == site}
            if "left" in pair and "right" in pair:
                left, right = pair["left"], pair["right"]
                result.append(BilateralMetrics(muscle_site=site,
                    imbalance_pct=200*abs(right-left)/(right+left) if right+left else None,
                    right_greater_than_left_pct=100*(right-left)/left if left else None))
        return result
