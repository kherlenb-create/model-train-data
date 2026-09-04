from __future__ import annotations

import datetime
from pydantic import BaseModel
from enum import Enum
import numpy as np
from typing import Any

class CandleModel(BaseModel):
    o: float
    h: float
    l: float
    c: float
    v: float | None = None
    timestamp: int

class GroupLabel(str, Enum):
    UPWARD = "UPWARD"
    DOWNWARD = "DOWNWARD"

class SimilarPatternModel(BaseModel):
    pattern_candles: list[CandleModel]
    reaction_candles: list[CandleModel]
    similarity_score: float
    end_timestamp: float

class GenerateModel(BaseModel):
    scanned_candles: list[CandleModel]
    similar_patterns: list[SimilarPatternModel]
    q_anchor: float
    forward_candle_data: list[CandleModel] | None = None

class ReturnCandleModel(BaseModel):
    o: float
    h: float
    l: float
    c: float

class ReturnCandle(BaseModel):
    o: float
    h: float
    l: float
    c: float

class CandleGroup(BaseModel):
    label: GroupLabel
    candles: list[ReturnCandle]

class CandleGroupList(BaseModel):
    groups: list[CandleGroup]
