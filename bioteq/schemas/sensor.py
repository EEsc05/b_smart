from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class SensorReading(BaseModel):
    """Proposed simulator interface; units are not hardware specifications."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp_ms: int = Field(ge=0)
    session_id: str = Field(min_length=1)
    exercise: str = Field(min_length=1)
    channel: Literal["emg", "ecg", "nirs", "accelerometer", "geolocation"]
    value: float | None
    unit: str
    muscle_site: str | None = None
    side: Literal["left", "right"] | None = None
    axis: str | None = None
    rep_number: int | None = Field(default=None, ge=1)
    source: Literal["simulated", "public_dataset", "experimental"]
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_channel(self):
        units = {"emg": {"mV"}, "ecg": {"mV", "bpm"}, "nirs": {"percent"},
                 "accelerometer": {"m/s^2"}, "geolocation": {"degrees"}}
        if self.unit not in units[self.channel]:
            raise ValueError("Invalid unit for channel")
        if self.channel in {"emg", "nirs"} and not self.muscle_site:
            raise ValueError("Muscle channel requires muscle_site")
        if self.channel == "accelerometer" and self.axis not in {"x", "y", "z"}:
            raise ValueError("Acceleration requires x, y or z axis")
        if self.channel == "geolocation":
            if self.axis not in {"latitude", "longitude"}:
                raise ValueError("Location requires latitude or longitude axis")
            limit = 90 if self.axis == "latitude" else 180
            if self.value is not None and abs(self.value) > limit:
                raise ValueError("Invalid coordinate")
        if self.channel == "nirs" and self.value is not None and not 0 <= self.value <= 100:
            raise ValueError("NIRS must be between 0 and 100")
        if self.channel == "ecg" and self.unit == "bpm" and self.value is not None and self.value <= 0:
            raise ValueError("Heart rate must be positive")
        return self

class SimulatorLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    event_type: str
    muscle_site: str | None = None
    side: Literal["left", "right"] | None = None
    source: Literal["simulator_ground_truth"] = "simulator_ground_truth"

    @model_validator(mode="after")
    def ordered(self):
        if self.end_ms < self.start_ms:
            raise ValueError("Label end precedes start")
        return self
