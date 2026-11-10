
# BIOTEQ Muscle Engine — Implementation Plan

## 1. Project Objective

Build a modular Python backend that receives simulated wearable
sensor measurements, processes them into physiological and
movement-related features, and returns structured outputs for
BIOTEQ's athlete and consumer applications.

The system must initially support:

1. EMG — muscle activation and left/right asymmetry.
2. ECG — heart rate and recovery measurements.
3. NIRS — muscle oxygenation.
4. Accelerometer — repetition counting and movement timing.
5. Geolocation — workout location metadata.

The first exercise is a bicep curl.

The initial implementation should use deterministic rules and
signal-processing algorithms. Add a small scikit-learn model
as a separate proof of concept.

The project is software-only. There is no physical hardware
integration at this stage.

IMPORTANT:
- Do not claim to diagnose muscle strain or injury.
- Do not treat simulated labels as real injury labels.
- Keep physiological thresholds configurable.
- Clearly distinguish simulated and experimental data.
- Avoid unnecessary complexity and dependencies.
- Do not modify unrelated repository files.

## 2. Inspect the Repository First

Before implementing anything:

1. Inspect the existing repository structure.
2. Identify the Python version and dependency manager.
3. Check whether an API framework or existing data schema exists.
4. Reuse existing conventions where practical.
5. Do not overwrite Shams's simulator or existing application code.
6. Document any assumptions or integration questions.

If the repository is empty, create the structure described below.

## 3. Proposed Architecture

Use a modular, class-based design.

Suggested structure:

bioteq/
    __init__.py

    schemas/
        __init__.py
        sensor.py
        outputs.py

    ingestion/
        __init__.py
        reader.py
        validation.py
        session_buffer.py

    processing/
        __init__.py
        emg.py
        ecg.py
        nirs.py
        movement.py
        location.py

    engine/
        __init__.py
        muscle_engine.py
        feature_extractor.py
        rules.py
        warnings.py
        trends.py
        recommendations.py

    ml/
        __init__.py
        dataset.py
        train.py
        predict.py

    api/
        __init__.py
        main.py

    config/
        thresholds.yaml

examples/
    sample_session.csv
    sample_labels.csv
    sample_output.json

tests/
    test_schemas.py
    test_ingestion.py
    test_processing.py
    test_engine.py
    test_api.py
    test_ml.py

README.md
requirements.txt

This is a suggested structure, not a strict requirement.
Prefer adapting an existing project to creating duplicates.

## 4. Input Data Contract

### 4.1 SensorReading

Create a Pydantic model named `SensorReading`.

Fields:

- timestamp_ms: int
- session_id: str
- exercise: str
- channel: str
- value: float | None
- unit: str
- muscle_site: str | None
- side: str | None
- axis: str | None
- rep_number: int | None
- source: str
- metadata: dict

Use explicit optional fields where measurements do not apply.

Supported channels:

- emg
- ecg
- nirs
- accelerometer
- geolocation

Use `source` to distinguish:
- simulated
- public_dataset
- experimental

Validate channel names, units and required fields.

Do not require muscle_site or side for ECG, geolocation,
or other channels where those fields are inapplicable.

Recommended initial units:

EMG:
- mV for electrical amplitude
- normalized activation as a separate derived feature

ECG:
- mV for raw ECG
- bpm for derived heart rate

NIRS:
- percent for muscle oxygen saturation

Accelerometer:
- m/s^2 for acceleration
- separate readings for x, y and z axes

Geolocation:
- degrees for latitude and longitude

Do not assume every channel has the same sampling rate.

Document that these are proposed interface units, not
confirmed specifications of BIOTEQ's future hardware.

### 4.2 Example Input

Example EMG measurement:

{
  "timestamp_ms": 1250,
  "session_id": "session_001",
  "exercise": "bicep_curl",
  "channel": "emg",
  "value": 0.35,
  "unit": "mV",
  "muscle_site": "biceps",
  "side": "right",
  "axis": null,
  "rep_number": 2,
  "source": "simulated",
  "metadata": {}
}

Example accelerometer measurement:

{
  "timestamp_ms": 1250,
  "session_id": "session_001",
  "exercise": "bicep_curl",
  "channel": "accelerometer",
  "value": 1.25,
  "unit": "m/s^2",
  "muscle_site": null,
  "side": null,
  "axis": "x",
  "rep_number": 2,
  "source": "simulated",
  "metadata": {}
}

Geolocation may use a separate session metadata schema
with latitude and longitude as distinct fields.

### 4.3 Ground Truth Labels

Create a separate model for simulator labels.

Example:

{
  "session_id": "session_001",
  "start_ms": 5000,
  "end_ms": 12000,
  "event_type": "shoulder_takeover",
  "muscle_site": "trapezius",
  "side": "right",
  "source": "simulator_ground_truth"
}

IMPORTANT:
Ground-truth labels must never be included as input
features for the classifier or rule-based engine.

They are only for testing and evaluation.

## 5. Input Processing

Create a `SensorReader` class.

Required methods:

- read_csv(path)
- read_json(path)
- read_jsonl(path)
- validate_readings(readings)
- group_by_session(readings)

Return validated SensorReading objects.

Create a `SessionBuffer` class.

Responsibilities:

- Receive readings incrementally.
- Group measurements by session and channel.
- Maintain a configurable rolling window.
- Handle missing or delayed measurements.
- Prevent accidental mixing of different sessions.
- Return synchronized windows when possible.

Support both:

1. Batch processing from CSV files.
2. Incremental processing of incoming readings.

Do not implement real network streaming yet.
The buffer should be compatible with future streaming.

Use timestamps rather than assuming that all channels
arrive at identical intervals.

Do not silently interpolate missing values.

## 6. Feature Extraction

Create a base processing interface.

Each processor should accept validated readings
and return derived features with units and quality flags.

### 6.1 EMGProcessor

Implement:

- Root mean square (RMS) amplitude.
- Mean absolute value.
- Optional basic filtering when sampling frequency
  is known and sufficiently high.
- Normalized activation if a reference value exists.
- Left/right activation comparison.

RMS:

RMS = sqrt(mean(x_i^2))

Example output:

{
  "muscle_site": "biceps",
  "side": "right",
  "rms_mv": 0.18,
  "normalized_activation_pct": null,
  "quality": "ok"
}

Do not calculate percent MVC without an MVC reference.

Left/right imbalance:

imbalance_pct =
    100 * abs(right - left) / ((right + left) / 2)

Handle zero denominators.

Document the chosen normalization.

Do not interpret EMG amplitude directly as muscle force.

### 6.2 ECGProcessor

Implement:

- Heart rate from validated R-peak intervals.
- Mean heart rate over a time window.
- Heart-rate recovery if a rest interval exists.

Heart rate:

HR_bpm = 60000 / RR_interval_ms

For the first version, support either:

1. Precomputed R-peak timestamps, or
2. Heart-rate readings supplied by the simulator.

Do not pretend a generic ECG waveform can be reliably
processed without a tested peak-detection algorithm.

If raw ECG peak detection is implemented, make it
an optional, independently tested component.

### 6.3 NIRSProcessor

Implement:

- Mean muscle oxygen saturation.
- Change from a defined baseline.
- Minimum and maximum saturation within a window.

Example:

{
  "smo2_pct": 62.0,
  "baseline_smo2_pct": 70.0,
  "change_percentage_points": -8.0
}

Distinguish percentage points from relative percent change.

Do not assume NIRS measurements directly diagnose fatigue.

### 6.4 MovementProcessor

Implement:

- Acceleration magnitude.
- Basic peak-based repetition counting.
- Average repetition duration.
- Movement variability.
- Optional movement asymmetry if paired measurements exist.

Acceleration magnitude:

a = sqrt(ax^2 + ay^2 + az^2)

Use scipy.signal.find_peaks where appropriate.

Make peak prominence, minimum spacing and smoothing
parameters configurable.

Include tests with ten simulated repetitions.

Document that gravity, sensor orientation and exercise
type affect acceleration-based rep counting.

### 6.5 LocationProcessor

Implement:

- Session location storage.
- Optional distance calculation for multiple GPS points.

Do not use geolocation to estimate muscle strain.

Do not infer indoor gym location from unreliable GPS data.

Treat location as optional metadata.

## 7. Muscle Engine

Create a `MuscleEngine` class that coordinates the
processors and produces one consistent output schema.

Suggested interface:

class MuscleEngine:
    def process_session(self, readings):
        ...

    def process_window(self, readings):
        ...

    def get_session_summary(self, session_id):
        ...

Responsibilities:

1. Validate inputs.
2. Group readings by channel.
3. Call the relevant processors.
4. Assemble derived features.
5. Evaluate configurable rules.
6. Return structured outputs.
7. Record missing inputs and quality warnings.

Avoid tightly coupling the engine to a particular
simulator or API framework.

## 8. Output Schema

Create Pydantic models for:

- MuscleMetrics
- CardiovascularMetrics
- MovementMetrics
- OxygenationMetrics
- WarningEvent
- SessionSummary
- EngineOutput

Example output:

{
  "session_id": "session_001",
  "exercise": "bicep_curl",
  "data_source": "simulated",
  "metrics": {
    "rep_count": 10,
    "average_rep_duration_s": 3.2,
    "heart_rate_bpm": 142,
    "heart_rate_recovery_bpm": null,
    "muscles": [
      {
        "muscle_site": "biceps",
        "side": "right",
        "rms_mv": 0.18,
        "normalized_activation_pct": null
      }
    ],
    "muscle_oxygenation_pct": 62.0,
    "left_right_imbalance_pct": 18.75
  },
  "warnings": [
    {
      "code": "possible_compensation",
      "severity": "informational",
      "message": "Elevated right-side activation detected",
      "confidence": null,
      "method": "rule_based"
    }
  ],
  "quality_flags": [],
  "model_version": "rules_v0.1"
}

All fields must be documented.

Use null for unavailable measurements.

Do not fabricate values when a channel is missing.

Distinguish:
- Observed measurements
- Derived features
- Rule-based warnings
- ML predictions

## 9. Rule-Based Baseline

Create a `RuleEngine`.

Use thresholds from config/thresholds.yaml.

Possible rules:

- Left/right activation imbalance.
- Elevated compensatory muscle activation.
- Declining movement consistency.
- Large oxygenation changes.
- Unusual heart-rate response.

Do not hardcode medically meaningful thresholds.

If thresholds have not been approved by Yara,
mark them as demonstration-only.

Example:

imbalance:
  enabled: true
  threshold_pct: null
  status: "awaiting_science_review"

Provide an optional demo configuration with clearly
labelled arbitrary values for testing the software.

The production/default configuration must not issue
physiological danger warnings based on arbitrary values.

Separate:
- Measurement quality warnings
- Movement/technique flags
- Physiological caution flags

Use the phrase "possible compensation" rather than
diagnosing injury.

## 10. Required Simulator Test Cases

Support the two planted cases from Shams's syllabus.

### Case A: Shoulder Takeover

During a bicep curl, the right trapezius shows
38% greater activation than the left.

Requirements:

- Calculate bilateral activation differences.
- Preserve the simulator's event labels separately.
- Detect the event when an appropriate demo threshold
  is enabled.
- Report the event time window.
- Avoid interpreting the event as a confirmed injury.

Note:
Define whether "38% greater" uses the left side
as denominator. Do not confuse this with the
symmetric imbalance formula.

### Case B: Four-Week Imbalance

Over four weeks, the right side improves while
the left side remains relatively unchanged.

Requirements:

- Accept session history.
- Aggregate comparable measurements by week.
- Calculate left/right trends.
- Return changes over time.
- Do not claim muscle hypertrophy from EMG alone.

Create tests for both scenarios.

## 11. Historical Trends

Create a `TrendAnalyzer`.

Required functionality:

- Aggregate sessions by day or week.
- Calculate changes in comparable features.
- Track bilateral differences over time.
- Return historical values for graphing.
- Handle missing sessions.

Example output:

{
  "metric": "left_right_imbalance_pct",
  "unit": "%",
  "values": [
    {"week": 1, "value": 5.0},
    {"week": 2, "value": 8.0},
    {"week": 3, "value": 12.0},
    {"week": 4, "value": 17.0}
  ]
}

No forecasting model is required initially.

## 12. Next-Workout Recommendation Interface

Create a `WorkoutRecommender` class.

For now, implement the interface and a conservative
rule-based demonstration.

Inputs may include:

- Recent exercise history
- Exercise type
- Movement quality flags
- Muscle activation trends
- User goals
- Available exercises
- Rest intervals

Outputs:

- Suggested exercise
- Suggested sets and repetitions
- Reason for recommendation
- Missing information
- Whether recommendation is demo-only

Do not automatically prescribe exercises solely
because one muscle has high activation.

If insufficient information is available, return
"insufficient_data" rather than inventing a plan.

## 13. Small scikit-learn Proof of Concept

Build one small supervised classification experiment.

This must be independent from the production
rule-based engine.

### Candidate dataset

UCI Machine Learning Repository:

Physical Therapy Exercises Dataset

https://archive.ics.uci.edu/dataset/730/physical+therapy+exercises+dataset

This dataset contains wearable motion recordings
with exercise execution categories.

Before downloading or training:

1. Verify the dataset URL and availability.
2. Review the current dataset documentation.
3. Verify the licence and attribution requirements.
4. Record the licence in a dataset manifest.
5. Require confirmation that Yara has approved its use.
6. Do not commit the dataset itself to Git.
7. Do not silently substitute another dataset.

If approval is not available, implement the training
pipeline and unit tests using a small synthetic
fixture, but do not report it as public-data validation.

### Classification Objective

For the initial experiment, predict the available
exercise execution category from motion features.

Do not relabel these categories as injuries.

If the dataset contains correct, fast and
low-amplitude execution labels, use those labels
as documented.

Inspect the actual dataset schema before coding
the preprocessing.

### Feature Extraction

Depending on the dataset structure, consider:

- Mean acceleration per axis
- Standard deviation
- RMS acceleration
- Peak acceleration
- Signal magnitude area
- Window duration
- Optional frequency-domain features

Use fixed-size windows or complete repetitions,
depending on the available annotations.

Keep windowing consistent across train and test.

### Models

Implement:

1. DummyClassifier as a baseline.
2. LogisticRegression.
3. RandomForestClassifier.

Use scikit-learn pipelines.

Standardize features for logistic regression.

Use a fixed random seed.

### Evaluation

Report:

- Accuracy
- Macro F1 score
- Per-class precision and recall
- Confusion matrix
- Number of participants
- Number of train/test samples

IMPORTANT:

Split by participant where participant IDs are
available to avoid leakage.

If participant IDs are unavailable, investigate
session-based splitting and document limitations.

Never randomly distribute overlapping windows
from the same recording across training and test.

Compare all models on the same held-out data.

Do not present a simulator score as real-world
model accuracy.

### Saving the Model

Save the fitted preprocessing and classifier
pipeline using joblib.

Record:

- Model version
- Dataset identifier
- Training date
- Feature names
- Label mapping
- Validation metrics
- Dataset licence
- Known limitations

Only load trusted local model files.

### ML Integration

Create an `MLPredictor` class.

Suggested methods:

- load_model(path)
- predict(features)
- predict_proba(features)

Only run inference when incoming features match
the model's training feature schema.

Do not apply a model trained on accelerometer data
directly to EMG, ECG or NIRS features.

If input features are incompatible, return an
explicit error or unsupported status.

Do not combine unrelated public datasets into
synthetic synchronized multimodal observations
and present them as real recordings.

## 14. API

If no API framework exists, use FastAPI.

Implement:

GET /health

POST /sessions/process

POST /sessions/window

GET /sessions/{session_id}/summary

POST /models/predict

Use Pydantic request and response schemas.

Provide OpenAPI documentation.

Use HTTP 422 for invalid inputs.

Do not implement authentication or Coach connect
access control in this first version.

Document these as future requirements.

The API must not expose confidential simulator labels
as though they were sensor measurements.

## 15. Example Files

Create small deterministic example files:

examples/sample_session.csv

Contains a simulated ten-repetition bicep curl.

Include:
- EMG
- Accelerometer
- ECG or derived heart-rate measurements
- NIRS
- Optional session geolocation

examples/sample_labels.csv

Contains planted event annotations.

examples/sample_output.json

Contains the expected engine response.

Clearly label all example measurements as simulated.

Use a fixed random seed if generating fixtures.

Do not build a full athlete simulator.
That is Shams's responsibility.

## 16. Testing

Use pytest.

Test at minimum:

1. Valid input parsing.
2. Invalid channel rejection.
3. Missing channel handling.
4. Mixed sampling rates.
5. Out-of-order timestamps.
6. Session separation.
7. EMG RMS calculation.
8. Left/right imbalance calculation.
9. Zero denominator handling.
10. Heart-rate calculation.
11. NIRS baseline change.
12. Ten-repetition counting.
13. Rule evaluation.
14. Planted shoulder takeover.
15. Four-week trend aggregation.
16. Output JSON serialization.
17. API request validation.
18. ML feature compatibility.
19. Participant-level train/test separation.

All tests must be runnable locally.

## 17. README

Document:

- Project purpose
- Architecture
- Installation
- Dependencies
- Running the tests
- Processing a sample session
- Starting the API
- Example request and response
- Running the ML experiment
- Dataset licence requirements
- Current limitations
- Integration instructions for Shams
- Metrics requiring Yara's review

Include a short explanation of the difference between
simulated, public experimental and real sensor data.

## 18. Implementation Phases

### Phase 1 — Core Interfaces and Baseline

Implement first:

- Pydantic schemas
- CSV/JSON ingestion
- Sensor processors
- MuscleEngine
- Basic rules
- Structured outputs
- Example data
- Unit tests

Acceptance criteria:

- A simulated session can be loaded from CSV.
- The engine calculates available features.
- It returns valid JSON.
- Missing channels do not crash processing.
- The planted shoulder-takeover case can be evaluated
  with explicitly enabled demonstration thresholds.
- All Phase 1 tests pass.

### Phase 2 — API and History

Implement:

- FastAPI endpoints
- Session buffer
- Four-week trend analysis
- Additional tests
- API examples

Acceptance criteria:

- A client can submit a session and receive metrics.
- Incremental windows can be processed.
- Four-week trends are returned in structured form.
- API documentation is generated.

### Phase 3 — Small ML Experiment

Implement:

- Dataset manifest and licence gate
- Public dataset loader
- Feature extraction
- Dummy baseline
- Logistic regression
- Random forest
- Evaluation
- Model persistence
- MLPredictor interface

Acceptance criteria:

- The training script runs on an approved dataset.
- Results include macro F1 and a confusion matrix.
- Train/test separation is documented.
- The model can be reloaded for inference.
- No real-world injury detection claim is made.

### Phase 4 — Future Integration

Leave interfaces or TODOs for:

- Coach connect permissions
- Athlete and consumer views
- Real-time streaming transport
- Personalized readiness
- Workout recommendations
- Additional exercises
- Multimodal ML models

Do not implement these prematurely.

## 19. Final Deliverables

When implementation is complete, provide:

1. Summary of files created or modified.
2. Architecture explanation.
3. Input schema.
4. Output schema.
5. Example JSON request and response.
6. Instructions for Shams to connect the simulator.
7. List of metrics needing Yara's approval.
8. ML experiment results, if actually run.
9. Test results.
10. Known limitations and next steps.

Do not claim that tests passed unless they were run.

Do not claim a model was trained unless training
actually completed.

Do not claim dataset licensing was approved without
confirmation.

Start with Phase 1. Prefer a small, working system
over a large incomplete implementation.
