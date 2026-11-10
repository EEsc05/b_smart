from pydantic import BaseModel, Field

class MuscleMetrics(BaseModel):
    muscle_site: str
    side: str | None
    rms_mv: float
    mean_absolute_mv: float
    normalized_activation_pct: float | None = None
    quality: str = "ok"

class BilateralMetrics(BaseModel):
    muscle_site: str
    imbalance_pct: float | None
    right_greater_than_left_pct: float | None

class CardiovascularMetrics(BaseModel):
    heart_rate_bpm: float | None = None
    heart_rate_recovery_bpm: float | None = None
    quality: str = "unavailable"

class OxygenationMetrics(BaseModel):
    muscle_site: str
    side: str | None
    smo2_pct: float
    baseline_smo2_pct: float | None = None
    change_percentage_points: float | None = None
    minimum_pct: float
    maximum_pct: float
    quality: str = "ok"

class MovementMetrics(BaseModel):
    rep_count: int | None = None
    average_rep_duration_s: float | None = None
    movement_variability_cv: float | None = None
    mean_acceleration_m_s2: float | None = None
    quality: str = "unavailable"

class WarningEvent(BaseModel):
    code: str
    severity: str = "informational"
    category: str = "technique"
    message: str
    confidence: float | None = None
    method: str = "rule_based_demo"
    start_ms: int
    end_ms: int

class Metrics(BaseModel):
    muscles: list[MuscleMetrics] = Field(default_factory=list)
    bilateral: list[BilateralMetrics] = Field(default_factory=list)
    cardiovascular: CardiovascularMetrics = Field(default_factory=CardiovascularMetrics)
    oxygenation: list[OxygenationMetrics] = Field(default_factory=list)
    movement: MovementMetrics = Field(default_factory=MovementMetrics)
    location: dict | None = None

class SessionSummary(BaseModel):
    session_id: str
    exercise: str
    data_source: str
    start_ms: int
    end_ms: int
    metrics: Metrics

class EngineOutput(SessionSummary):
    warnings: list[WarningEvent] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    model_version: str = "rules_v0.1"
