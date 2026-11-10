# BIOTEQ Muscle Engine

Phase 1 of `initiation.md`: a modular Python backend for simulated bicep-curl measurements. It validates readings, calculates features, and returns structured JSON. Python 3.11+ is required; tested with Python 3.13. Existing `input.csv`, `input.py`, `output.py`, and `temp.json` are preserved. The legacy CSV has no measurements and is not the new sensor contract.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m examples.generate
```

Process the sample from the repository root:

```python
from bioteq.ingestion.reader import SensorReader
from bioteq.engine.muscle_engine import MuscleEngine

readings = SensorReader().read_csv("examples/sample_session.csv")
engine = MuscleEngine()
output = engine.process_session(readings)
print(output.model_dump_json(indent=2))
summary = engine.get_session_summary("session_001")
```

`examples/sample_output.json` is the full expected response. `examples/generate.py` regenerates this deterministic ten-repetition fixture. All example data is simulated. No public dataset was downloaded and no model was trained.

## Architecture and simulator integration

- `schemas/`: Pydantic input, label, and output contracts.
- `ingestion/`: CSV, JSON array, and JSONL readers, validation, session grouping.
- `processing/`: independent EMG, ECG, NIRS, acceleration, and location processors with a shared base interface.
- `engine/`: orchestration and configurable rules; no simulator or API dependency.
- `config/`: inactive default rules and explicitly labelled demo thresholds.

Shams can emit one `SensorReading` per measurement, then pass validated objects or dictionaries directly to `process_session` or `process_window`. Use one session and exercise per call. `SensorReader.group_by_session` separates batches first. Readings are sorted by timestamp; channels need not share a sampling rate. CSV metadata is a JSON object; blank optional cells become null. JSON/JSONL use ordinary JSON nulls. Duplicate measurement keys are rejected, missing values are omitted from calculations and flagged, and no interpolation occurs. Session summaries are in-memory snapshots of the last `process_session` call; windows do not accumulate or overwrite summaries.

Example input object (JSON files contain arrays of these):

```json
{"timestamp_ms":1250,"session_id":"session_001","exercise":"bicep_curl","channel":"emg","value":0.35,"unit":"mV","muscle_site":"biceps","side":"right","axis":null,"rep_number":2,"source":"simulated","metadata":{}}
```

The required fields are timestamp_ms (nonnegative milliseconds), session_id and exercise (nonempty strings), channel, value (finite number or null), unit, and source. Optional muscle_site, side, axis and rep_number default to null; metadata defaults to an empty object. Sides are left/right and rep numbers start at 1. EMG/NIRS require muscle_site; ECG and location do not require side or muscle_site.

| Channel | Proposed unit | Channel-specific fields |
| --- | --- | --- |
| emg | mV | muscle_site; optional side |
| ecg | mV or bpm | Optional metadata.r_peak=true for externally validated R peaks |
| nirs | percent (0–100) | muscle_site; optional side |
| accelerometer | m/s^2 | axis x, y or z |
| geolocation | degrees | axis latitude or longitude, coordinate bounds validated |

These units are proposed interfaces, not confirmed hardware specifications. Source is `simulated`, `public_dataset`, or `experimental`. Simulated means generated software measurements; public_dataset means externally sourced recordings with separate provenance/licence obligations; experimental means collected research measurements, not clinically validated measurements.

Simulator labels live separately in `SimulatorLabel` and `examples/sample_labels.csv`. They contain session_id, start_ms, end_ms, event_type, optional muscle_site/side, and source=simulator_ground_truth. Do not attach confidential labels to metadata. The engine never uses labels as features; label objects are rejected as readings.

## Output fields and calculations

`EngineOutput` extends `SessionSummary`: session_id, exercise, data_source (single source or explicit mixed sources), start_ms/end_ms (observed window bounds), and metrics. It adds warnings, quality_flags (missing channels/values or unsupported/insufficient processing), and model_version. Unavailable scalars are null and absent muscle groups are empty lists. Output models are in `bioteq/schemas/outputs.py`.

| metrics field | Meaning / units |
| --- | --- |
| muscles | MuscleMetrics: muscle_site, side, rms_mv, mean_absolute_mv, normalized_activation_pct, quality |
| bilateral | Per-muscle imbalance_pct and signed right_greater_than_left_pct |
| cardiovascular | Mean heart_rate_bpm, heart_rate_recovery_bpm, quality |
| oxygenation | Per-site/side mean smo2_pct, baseline_smo2_pct, change_percentage_points, minimum_pct, maximum_pct, quality |
| movement | rep_count, average_rep_duration_s, movement_variability_cv (dimensionless), mean_acceleration_m_s2, quality |
| location | Last complete same-timestamp latitude/longitude pair and timestamp_ms; otherwise null |

EMG RMS is sqrt(mean(x²)); mean absolute value is mean(abs(x)). Normalization is 100 * window RMS / caller-provided RMS reference (`emg_references={("biceps", "right"): reference}`). It is percent MVC only if that reference comes from an appropriate MVC protocol. No reference means null. No filtering is applied in this baseline. EMG is not a direct muscle force or hypertrophy measurement.

Bilateral imbalance is 200*abs(right-left)/(right+left). Right-greater-than-left is 100*(right-left)/left. Zero denominators yield null. The fixture's trapezius is 38% greater on the right using the left denominator, corresponding to about 31.93% symmetric imbalance. Comparisons use whole-window RMS; callers must provide comparable bilateral windows and calibration.

ECG uses supplied bpm, or 60000/RR_ms from unique externally validated R-peak timestamps if bpm is unavailable. Raw ECG alone returns unsupported quality. Mean heart rate is an arithmetic sample mean. With `rest_interval_ms=(start, end)`, recovery is first minus last observed heart rate within that explicit interval, requiring two observations. It does not extrapolate to a fixed recovery time. NIRS baseline is caller-defined via `nirs_baselines={("biceps", "right"): 70}`; changes are percentage points, not relative percentages or fatigue diagnoses.

Movement pairs x/y/z at exact timestamps, computes magnitude, and uses scipy peak prominence plus timestamp-based minimum peak spacing. `movement_settings` accepts prominence (default 0.5 m/s²), min_spacing_ms (1000), and smoothing_samples (1). Duration is the average interval between detected peaks; variability is population standard deviation / mean of those intervals. Partial xyz frames are dropped and flagged. Gravity, orientation, placement, gaps and exercise affect results; counts are a demonstration. Location is optional metadata and does not inform physiological rules.

Warnings contain code, severity, category, message, confidence (null), method, start_ms and end_ms. Demo flags use informational severity, technique category, method=rule_based_demo, and the evaluated window's full bounds. These bounds identify the evaluated window, not a localized event onset. There are no ML predictions in this phase.

To exercise the planted case explicitly:

```python
engine = MuscleEngine(config_path="bioteq/config/demo_thresholds.yaml")
output = engine.process_session(readings)
assert output.warnings[0].code == "possible_compensation"
```

Default thresholds are inactive and await Yara's review. The separate demo uses an arbitrary 30% symmetric imbalance threshold for software testing. Rules do not diagnose strain or injury. Yara must review normalization/MVC protocols, bilateral comparability, compensatory activation criteria, movement consistency thresholds, oxygenation changes, heart-rate/recovery protocols, and any future physiological caution thresholds.

## Scope and next steps

Phase 2 adds FastAPI/OpenAPI endpoints, an incremental session buffer, and day/week history including the four-week scenario. No API server exists yet. Phase 3 adds independent scikit-learn pipelines and participant-separated evaluation. Public-data use requires documentation/licence verification and Yara's explicit approval before download or training; no licence approval is assumed. Synthetic test scores must never be reported as real-world validation.

Future integration includes authentication/Coach connect permissions, persistent summaries, streaming transport, other exercises and conservative workout recommendations. This phase provides no diagnosis, personalized readiness score, or workout prescription. Confirm simulator calibration/reference protocols, sampling characteristics, missing-data conventions and recording provenance before integration with real sensors.

## Colab notebook

Open [notebooks/bioteq_colab.ipynb](notebooks/bioteq_colab.ipynb) in Google Colab and run all cells. It installs dependencies, loads the simulated sample, runs the rule-based engine, displays metric tables and sensor/RMS plots, and exports default/demo JSON outputs. Push the code to GitHub first, or select the repository ZIP upload option for local changes. No trained ML model is included.
