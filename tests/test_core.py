import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from bioteq.schemas.sensor import SensorReading, SimulatorLabel
from bioteq.schemas.outputs import EngineOutput
from bioteq.ingestion.reader import SensorReader
from bioteq.engine.muscle_engine import MuscleEngine
from bioteq.processing.emg import EMGProcessor
from bioteq.processing.ecg import ECGProcessor
from bioteq.processing.nirs import NIRSProcessor

ROOT = Path(__file__).resolve().parents[1]

def reading(**kwargs):
    return SensorReading.model_validate(dict(timestamp_ms=0, session_id="s", exercise="bicep_curl", channel="emg",
        value=1, unit="mV", muscle_site="biceps", side="left", source="simulated") | kwargs)

@pytest.mark.parametrize("updates", [{"channel":"other"}, {"unit":"bpm"}, {"value":float("nan")}, {"timestamp_ms":-1}, {"source":"real"}, {"event_type":"injury"}])
def test_invalid(updates):
    with pytest.raises(ValidationError):
        reading(**updates)

def test_emg():
    processor = EMGProcessor({("biceps", "left"): 5})
    result = processor.process([reading(value=3), reading(value=-4, timestamp_ms=1)])
    assert result[0].rms_mv == pytest.approx((12.5)**0.5)
    assert result[0].mean_absolute_mv == 3.5
    assert result[0].normalized_activation_pct == pytest.approx(100*(12.5)**0.5/5)
    assert EMGProcessor().process([reading()])[0].normalized_activation_pct is None

def test_bilateral():
    p = EMGProcessor()
    metrics = p.bilateral(p.process([reading(), reading(side="right", value=1.38)]))[0]
    assert metrics.right_greater_than_left_pct == pytest.approx(38)
    assert metrics.imbalance_pct == pytest.approx(200*.38/2.38)
    zero = p.bilateral(p.process([reading(value=0), reading(side="right", value=0)]))[0]
    assert zero.imbalance_pct is None
    assert zero.right_greater_than_left_pct is None

def test_ecg():
    rows = [reading(channel="ecg", muscle_site=None, side=None, timestamp_ms=t, metadata={"r_peak":True}) for t in (0, 1000, 2000)]
    assert ECGProcessor().process(rows).heart_rate_bpm == 60
    assert ECGProcessor().process([reading(channel="ecg", metadata={})]).heart_rate_bpm is None
    assert ECGProcessor().process([reading(channel="ecg", unit="bpm", value=120)]).heart_rate_bpm == 120

def test_nirs():
    result = NIRSProcessor({("biceps", "left"):70}).process([reading(channel="nirs", unit="percent", value=62)])[0]
    assert result.change_percentage_points == -8
    assert result.minimum_pct == result.maximum_pct == 62

def test_readers(tmp_path):
    row = reading().model_dump()
    (tmp_path / "a.json").write_text(json.dumps([row]))
    (tmp_path / "a.jsonl").write_text(json.dumps(row) + "\n\n")
    reader = SensorReader()
    assert reader.read_json(tmp_path / "a.json") == reader.read_jsonl(tmp_path / "a.jsonl") == [reading()]
    groups = reader.group_by_session([reading(timestamp_ms=2), reading(), reading(session_id="other")])
    assert [r.timestamp_ms for r in groups["s"]] == [0, 2]
    assert len(groups["other"]) == 1

def test_session_isolation():
    engine = MuscleEngine()
    with pytest.raises(ValueError):
        engine.process_session([reading(), reading(session_id="other")])
    with pytest.raises(ValueError):
        engine.process_session([reading(), reading()])
    with pytest.raises(ValueError):
        engine.process_session([])
    result = engine.process_session([reading(value=None)])
    assert result.metrics.muscles == []
    assert result.metrics.movement.rep_count is None
    assert "missing_values:no_interpolation" in result.quality_flags
    assert "missing_channel:ecg" in result.quality_flags
    assert engine.get_session_summary("s") == result
    assert EngineOutput.model_validate_json(result.model_dump_json()) == result

def test_sample_and_demo_rules():
    rows = SensorReader().read_csv(ROOT / "examples/sample_session.csv")
    default = MuscleEngine().process_session(list(reversed(rows)))
    assert default.metrics.movement.rep_count == 10
    assert default.metrics.movement.average_rep_duration_s == pytest.approx(3)
    assert default.metrics.cardiovascular.heart_rate_bpm == 120
    assert default.warnings == []
    assert not default.quality_flags
    assert default.model_dump() == json.loads((ROOT / "examples/sample_output.json").read_text())
    demo = MuscleEngine(ROOT / "bioteq/config/demo_thresholds.yaml").process_session(rows)
    assert len(demo.warnings) == 1
    event = demo.warnings[0]
    assert event.code == "possible_compensation"
    assert (event.start_ms, event.end_ms) == (0, 30000)
    assert event.method == "rule_based_demo"
    with pytest.raises(ValidationError):
        SensorReading.model_validate(SimulatorLabel(session_id="s", start_ms=0, end_ms=1, event_type="shoulder_takeover").model_dump())

def test_incomplete_axes():
    rows = [reading(channel="accelerometer", unit="m/s^2", axis="x", timestamp_ms=t) for t in range(5)]
    result = MuscleEngine().process_window(rows)
    assert result.metrics.movement.rep_count is None
    assert "movement:insufficient_complete_xyz" in result.quality_flags

def test_recovery():
    rows = [reading(channel="ecg", unit="bpm", value=v, timestamp_ms=t) for t, v in ((0, 150), (1000, 140), (2000, 120))]
    assert ECGProcessor((1000, 2000)).process(rows).heart_rate_recovery_bpm == 20
    assert ECGProcessor((3000, 4000)).process(rows).heart_rate_recovery_bpm is None


def test_timestamp_peak_spacing():
    from bioteq.processing.movement import MovementProcessor
    rows = []
    for t, value in ((0, 0), (100, 2), (150, 0), (200, 3), (300, 0), (1800, 2), (1900, 0)):
        for axis in ("x", "y", "z"):
            rows.append(reading(channel="accelerometer", unit="m/s^2", axis=axis, timestamp_ms=t, value=value if axis == "x" else 0))
    assert MovementProcessor(min_spacing_ms=1000).process(rows).rep_count == 2


def test_labels_and_disabled_rules(tmp_path):
    import csv
    labels = list(csv.DictReader((ROOT / "examples/sample_labels.csv").open()))
    assert SimulatorLabel.model_validate(labels[0]).event_type == "shoulder_takeover"
    with pytest.raises(ValidationError):
        SimulatorLabel(session_id="s", start_ms=2, end_ms=1, event_type="test")
    config = tmp_path / "thresholds.yaml"
    config.write_text("imbalance:\n  enabled: true\n  threshold_pct: null\n  status: awaiting_science_review\n")
    with pytest.raises(ValueError):
        MuscleEngine(config)
