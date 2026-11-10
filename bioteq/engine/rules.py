from pathlib import Path
import yaml
from bioteq.schemas.outputs import WarningEvent

class RuleEngine:
    def __init__(self, config_path=None):
        path = config_path or Path(__file__).parents[1] / "config" / "thresholds.yaml"
        self.config = yaml.safe_load(Path(path).read_text())
        rule = self.config["imbalance"]
        if rule["enabled"] and (rule["threshold_pct"] is None or rule["threshold_pct"] < 0 or rule["status"] != "demonstration_only"):
            raise ValueError("Only explicitly labelled demo thresholds are supported in v0.1")

    def evaluate(self, bilateral, start_ms, end_ms):
        rule = self.config["imbalance"]
        if not rule["enabled"]:
            return []
        return [WarningEvent(code="possible_compensation", message=f"Possible compensation: {m.muscle_site} bilateral RMS imbalance exceeds the demonstration threshold.", start_ms=start_ms, end_ms=end_ms)
            for m in bilateral if m.imbalance_pct is not None and m.imbalance_pct > rule["threshold_pct"]]
