"""
demucs-metacog: Demucs + Temporal Metacognition Engine
"""
from .engine import EngineConfig, EngineResult, MetaCogEngine
from .quality import QualityReport, evaluate_stems
from .separator import DemucsBase

__all__ = [
    "MetaCogEngine",
    "EngineConfig",
    "EngineResult",
    "DemucsBase",
    "evaluate_stems",
    "QualityReport",
]
