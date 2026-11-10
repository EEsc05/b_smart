from statistics import mean
from bioteq.processing.base import Processor, muscle_groups
from bioteq.schemas.outputs import OxygenationMetrics

class NIRSProcessor(Processor):
    def __init__(self, baselines=None):
        self.baselines = baselines or {}

    def process(self, readings):
        result = []
        for (site, side), rows in sorted(muscle_groups(readings).items(), key=lambda x: str(x[0])):
            values = [r.value for r in rows]
            baseline = self.baselines.get((site, side))
            avg = mean(values)
            result.append(OxygenationMetrics(muscle_site=site, side=side, smo2_pct=avg,
                baseline_smo2_pct=baseline, change_percentage_points=avg-baseline if baseline is not None else None,
                minimum_pct=min(values), maximum_pct=max(values)))
        return result
