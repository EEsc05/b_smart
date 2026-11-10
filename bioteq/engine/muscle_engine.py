from bioteq.ingestion.reader import SensorReader
from bioteq.processing.emg import EMGProcessor
from bioteq.processing.ecg import ECGProcessor
from bioteq.processing.nirs import NIRSProcessor
from bioteq.processing.movement import MovementProcessor
from bioteq.processing.location import LocationProcessor
from bioteq.engine.rules import RuleEngine
from bioteq.schemas.outputs import EngineOutput, Metrics

class MuscleEngine:
    def __init__(self, config_path=None, emg_references=None, nirs_baselines=None, movement_settings=None, rest_interval_ms=None):
        self.reader = SensorReader()
        self.emg = EMGProcessor(emg_references)
        self.ecg = ECGProcessor(rest_interval_ms)
        self.nirs = NIRSProcessor(nirs_baselines)
        self.movement = MovementProcessor(**(movement_settings or {}))
        self.location = LocationProcessor()
        self.rules = RuleEngine(config_path)
        self.summaries = {}

    def process_session(self, readings):
        output = self.process_window(readings)
        self.summaries[output.session_id] = output.model_copy(deep=True)
        return output

    def process_window(self, readings):
        rows = sorted(self.reader.validate_readings(readings), key=lambda r: r.timestamp_ms)
        if not rows:
            raise ValueError("At least one reading required")
        if len({r.session_id for r in rows}) != 1 or len({r.exercise for r in rows}) != 1:
            raise ValueError("Process one session and exercise at a time")
        # Reject ambiguous duplicates instead of silently overwriting measurements.
        keys = [(r.timestamp_ms, r.channel, r.unit, r.muscle_site, r.side, r.axis) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate sensor measurement")
        channels = {c: [r for r in rows if r.channel == c] for c in ("emg", "ecg", "nirs", "accelerometer", "geolocation")}
        flags = [f"missing_channel:{c}" for c, rs in channels.items() if not rs]
        if any(r.value is None for r in rows):
            flags.append("missing_values:no_interpolation")
        muscles = self.emg.process(channels["emg"])
        bilateral = self.emg.bilateral(muscles)
        metrics = Metrics(muscles=muscles, bilateral=bilateral,
            cardiovascular=self.ecg.process(channels["ecg"]), oxygenation=self.nirs.process(channels["nirs"]),
            movement=self.movement.process(channels["accelerometer"]), location=self.location.process(channels["geolocation"]))
        for name in ("cardiovascular", "movement"):
            quality = getattr(metrics, name).quality
            if quality != "ok":
                flags.append(f"{name}:{quality}")
        sources = sorted({r.source for r in rows})
        return EngineOutput(session_id=rows[0].session_id, exercise=rows[0].exercise,
            data_source=sources[0] if len(sources) == 1 else "mixed:" + ",".join(sources),
            start_ms=rows[0].timestamp_ms, end_ms=rows[-1].timestamp_ms, metrics=metrics,
            warnings=self.rules.evaluate(bilateral, rows[0].timestamp_ms, rows[-1].timestamp_ms), quality_flags=flags)

    def get_session_summary(self, session_id):
        return self.summaries[session_id].model_copy(deep=True)
