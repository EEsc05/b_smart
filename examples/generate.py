"""Small deterministic software fixture, not an athlete simulator."""
import csv
from math import sin, pi
from pathlib import Path
from bioteq.schemas.sensor import SensorReading
from bioteq.ingestion.reader import SensorReader
from bioteq.engine.muscle_engine import MuscleEngine

ROOT = Path(__file__).resolve().parent

def generate():
    rows = []
    def add(t, channel, value, unit, **kwargs):
        rows.append(SensorReading(timestamp_ms=t, session_id="session_001", exercise="bicep_curl",
            channel=channel, value=value, unit=unit, source="simulated", **kwargs).model_dump())
    for t in range(0, 30001, 100):
        for axis, value in zip(("x", "y", "z"), (2 + sin(2*pi*t/3000 - pi/2), 0, 0)):
            add(t, "accelerometer", value, "m/s^2", axis=axis)
    for t in range(0, 30001, 250):
        for side, value in (("left", 1.0), ("right", 1.38)):
            add(t, "emg", value, "mV", muscle_site="trapezius", side=side)
        add(t, "emg", 0.35, "mV", muscle_site="biceps", side="right")
    for t in range(0, 30001, 1000):
        add(t, "ecg", 120, "bpm")
        add(t, "nirs", 62, "percent", muscle_site="biceps", side="right")
    add(0, "geolocation", 50.45, "degrees", axis="latitude")
    add(0, "geolocation", -104.61, "degrees", axis="longitude")
    import json
    with (ROOT / "sample_session.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "metadata": json.dumps(row["metadata"])})
    output = MuscleEngine().process_session(SensorReader().read_csv(ROOT / "sample_session.csv"))
    (ROOT / "sample_output.json").write_text(output.model_dump_json(indent=2) + "\n")

if __name__ == "__main__":
    generate()
