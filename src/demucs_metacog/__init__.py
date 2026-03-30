"""
demucs-metacog: Demucs + Intent-Aware Temporal Metacognition Engine
"""
from .engine import EngineConfig, EngineResult, MetaCogEngine
from .intent import IntentProfile, SeparationIntent, get_intent_profile
from .quality import QualityReport, evaluate_stems
from .separator import DemucsBase

__all__ = [
    "MetaCogEngine",
    "EngineConfig",
    "EngineResult",
    "DemucsBase",
    "evaluate_stems",
    "QualityReport",
    "SeparationIntent",
    "IntentProfile",
    "get_intent_profile",
]
