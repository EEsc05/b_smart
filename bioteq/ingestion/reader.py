import csv
import json
from collections import defaultdict
from pathlib import Path
from bioteq.schemas.sensor import SensorReading

class SensorReader:
    def validate_readings(self, readings):
        return [SensorReading.model_validate(r) for r in readings]

    def read_csv(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in ("value", "muscle_site", "side", "axis", "rep_number"):
                if row.get(key) == "":
                    row[key] = None
            row["metadata"] = json.loads(row.get("metadata") or "{}")
        return self.validate_readings(rows)

    def read_json(self, path):
        return self.validate_readings(json.loads(Path(path).read_text()))

    def read_jsonl(self, path):
        return self.validate_readings(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())

    def group_by_session(self, readings):
        groups = defaultdict(list)
        for reading in self.validate_readings(readings):
            groups[reading.session_id].append(reading)
        return {key: sorted(value, key=lambda r: r.timestamp_ms) for key, value in groups.items()}
