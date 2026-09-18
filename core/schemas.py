"""Typed data contracts used across the pipeline."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TimeHorizon(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SiteProfile(BaseModel):
    """What we know about the user's land so far. Every field optional -
    that's how we detect what's still missing."""

    soil_organic_carbon_pct: Optional[float] = Field(None, ge=0, le=25)
    soil_ph: Optional[float] = Field(None, ge=0, le=14)
    rainfall_pattern: Optional[str] = None   # low | moderate | high | erratic
    rainfall_mm: Optional[float] = None
    land_use: Optional[str] = None           # monoculture | polyculture | grassland | agroforestry | fallow
    crop: Optional[str] = None
    region_type: Optional[str] = None        # arid | semi_arid | subhumid | humid | coastal
    area_hectares: Optional[float] = None
    observed_problem: Optional[str] = None
    human_pressures: List[str] = Field(default_factory=list)

    def filled_fields(self) -> List[str]:
        return [k for k, v in self.model_dump().items() if v not in (None, [], "")]

    def to_context_string(self) -> str:
        parts = [f"{k.replace('_', ' ')}: {v}" for k, v in self.model_dump().items() if v not in (None, [], "")]
        return "; ".join(parts) if parts else "no site data supplied yet"


class Evidence(BaseModel):
    card_id: str
    source: str
    claim: str


class Recommendation(BaseModel):
    action: str
    scientific_reasoning: str
    metrics_improved: List[str] = Field(default_factory=list)
    expected_change: str = ""
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM
    confidence: float = Field(0.5, ge=0, le=1)
    evidence: List[Evidence] = Field(default_factory=list)
    grounded: bool = True
    grounding_note: str = ""


class AdvisoryResponse(BaseModel):
    mode: str = "advisory"     # advisory | clarifying
    summary: str = ""
    diagnosis: str = ""
    recommendations: List[Recommendation] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)