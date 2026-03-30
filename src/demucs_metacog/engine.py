"""
engine.py — Temporal Metacognition Engine
「分離 → 品質監査 → 再分離ループ」のコアオーケストレーター。

ARC-AGIメタ認知エンジンの「直感（LLM生成）→実行（Sandbox）→修正（Meta-Cognition）」
ループを音声分離に移植した設計。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from .quality import QualityReport, evaluate_stems
from .separator import DemucsBase

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3  # 無限ループ防止上限


@dataclass
class EngineConfig:
    """エンジン設定。"""
    model_name: str = "htdemucs"
    device: Optional[str] = None
    max_iterations: int = MAX_ITERATIONS
    quality_thresholds: dict = field(default_factory=dict)
    # 再分離時のストラテジー
    retry_strategy: str = "shifts"   # "shifts" | "overlap" | "model_upgrade"
    save_iterations: bool = False     # 中間イテレーションも保存するか
    verbose: bool = True


@dataclass
class IterationResult:
    iteration: int
    stems: dict[str, torch.Tensor]
    report: QualityReport
    elapsed_sec: float


@dataclass
class EngineResult:
    """最終結果。"""
    final_stems: dict[str, torch.Tensor]
    sample_rate: int
    iterations: list[IterationResult]
    total_elapsed_sec: float

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def final_report(self) -> QualityReport:
        return self.iterations[-1].report

    def summary(self) -> str:
        lines = [
            f"=== MetaCog Engine Result ===",
            f"Total iterations : {self.n_iterations}",
            f"Total time       : {self.total_elapsed_sec:.1f}s",
            f"Final quality    : {'✅ PASS' if self.final_report.overall_passed else '⚠️ Best-effort'}",
            "",
        ]
        for r in self.iterations:
            lines.append(r.report.summary())
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 再分離ストラテジー（イテレーション毎にパラメータを変化させる）
# ──────────────────────────────────────────────

def _get_apply_kwargs(iteration: int, strategy: str) -> dict:
    """
    イテレーションと戦略に応じてDemucs apply_modelのパラメータを調整する。
    - shifts: ランダムシフトの回数を増やす → 位相揺らぎの平均化でアーティファクト低減
    - overlap: オーバーラップ率を上げる → セグメント境界ノイズを低減
    """
    if strategy == "shifts":
        return {"shifts": 1 + iteration, "overlap": 0.25}
    elif strategy == "overlap":
        overlap = min(0.25 + iteration * 0.1, 0.5)
        return {"shifts": 1, "overlap": overlap}
    else:
        return {"shifts": 1, "overlap": 0.25}


class MetaCogEngine:
    """
    Temporal Metacognition Engine for stem separation.

    ループ構造:
        for i in range(max_iterations):
            stems = demucs.separate(audio, **strategy_params(i))
            report = quality.evaluate(original, stems)
            if report.overall_passed:
                break  # 合格したら終了
            # 合格しなければ次のイテレーションへ（パラメータを変えて再試行）
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._separator = DemucsBase(
            model_name=self.config.model_name,
            device=self.config.device,
        )

    def _separate_with_strategy(
        self, waveform: torch.Tensor, sample_rate: int, iteration: int
    ) -> dict[str, torch.Tensor]:
        """分離実行（イテレーションに応じてパラメータを調整）。"""
        from demucs.apply import apply_model

        self._separator._load()
        model = self._separator._model

        target_sr = model.samplerate
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        kwargs = _get_apply_kwargs(iteration, self.config.retry_strategy)
        if self.config.verbose:
            logger.info(f"[Iter {iteration}] apply_model params: {kwargs}")

        wav_batch = waveform.unsqueeze(0).to(self._separator.device)
        with torch.no_grad():
            sources = apply_model(model, wav_batch, device=self._separator.device,
                                  split=True, progress=False, **kwargs)

        stems: dict[str, torch.Tensor] = {}
        for i, name in enumerate(model.sources):
            stems[name] = sources[0, i].cpu()

        return stems

    def run(self, waveform: torch.Tensor, sample_rate: int) -> EngineResult:
        """
        メタ認知ループを実行する。

        Args:
            waveform: (channels, samples) float tensor
            sample_rate: 入力のサンプルレート

        Returns:
            EngineResult
        """
        total_start = time.perf_counter()
        iteration_results: list[IterationResult] = []
        best_stems: Optional[dict[str, torch.Tensor]] = None

        logger.info(f"MetaCogEngine starting. max_iter={self.config.max_iterations}, "
                    f"model={self.config.model_name}")

        for i in range(self.config.max_iterations):
            iter_start = time.perf_counter()
            logger.info(f"\n--- Iteration {i} ---")

            # 1. 分離（直感パス）
            stems = self._separate_with_strategy(waveform, sample_rate, iteration=i)

            # 2. 品質監査（監査層）
            report = evaluate_stems(
                original=waveform,
                stems=stems,
                thresholds=self.config.quality_thresholds,
            )
            report.iteration = i
            elapsed = time.perf_counter() - iter_start

            iteration_results.append(IterationResult(
                iteration=i,
                stems=stems,
                report=report,
                elapsed_sec=round(elapsed, 2),
            ))

            if self.config.verbose:
                print(report.summary())

            # ベストを更新（最後に合格したイテレーション、または最終イテレーション）
            best_stems = stems

            # 3. 合格判定 → ループ脱出
            if report.overall_passed:
                logger.info(f"✅ Quality passed at iteration {i}")
                break
            else:
                if i < self.config.max_iterations - 1:
                    logger.info(f"⚠️  Quality check failed. Retrying with adjusted params...")
                else:
                    logger.info(f"⚠️  Max iterations reached. Returning best-effort result.")

        total_elapsed = time.perf_counter() - total_start

        result = EngineResult(
            final_stems=best_stems,
            sample_rate=self._separator.sample_rate,
            iterations=iteration_results,
            total_elapsed_sec=round(total_elapsed, 2),
        )

        return result
