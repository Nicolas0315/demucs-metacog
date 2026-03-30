"""
engine.py — Temporal Metacognition Engine v2

v2の追加:
- Intent-Aware: IntentProfileを受け取り、用途に応じた閾値・戦略を適用
- Targeted Re-separation: 失敗ステムだけを特定し、次のイテレーションで
  そのステムの監査基準を重点的にチェックする（全ステム再実行だが評価を集中）
- モデルアップグレード戦略: masteringモードでは再試行時にhtdemucs_ftに切り替える
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torchaudio

from .intent import IntentProfile, SeparationIntent, get_intent_profile
from .quality import QualityReport, evaluate_stems
from .separator import DemucsBase

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


@dataclass
class EngineConfig:
    """エンジン設定。"""
    model_name: str = "htdemucs"
    device: Optional[str] = None
    max_iterations: int = MAX_ITERATIONS
    quality_thresholds: dict = field(default_factory=dict)
    retry_strategy: str = "shifts"   # "shifts" | "overlap" | "model_upgrade"
    save_iterations: bool = False
    verbose: bool = True
    # Intent-Aware設定
    intent: SeparationIntent | str = SeparationIntent.DEFAULT


@dataclass
class IterationResult:
    iteration: int
    stems: dict[str, torch.Tensor]
    report: QualityReport
    elapsed_sec: float
    model_used: str = "htdemucs"
    intent_applied: str = "default"


@dataclass
class EngineResult:
    """最終結果。"""
    final_stems: dict[str, torch.Tensor]
    sample_rate: int
    iterations: list[IterationResult]
    total_elapsed_sec: float
    intent: str = "default"

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def final_report(self) -> QualityReport:
        return self.iterations[-1].report

    def summary(self) -> str:
        lines = [
            f"=== MetaCog Engine Result ===",
            f"Intent           : {self.intent}",
            f"Total iterations : {self.n_iterations}",
            f"Total time       : {self.total_elapsed_sec:.1f}s",
            f"Final quality    : {'✅ PASS' if self.final_report.overall_passed else '⚠️ Best-effort'}",
            "",
        ]
        for r in self.iterations:
            model_tag = f" [{r.model_used}]" if r.model_used != "htdemucs" else ""
            lines.append(f"  --- Iter {r.iteration}{model_tag} ({r.elapsed_sec:.1f}s) ---")
            lines.append(r.report.summary())
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 再分離パラメータ計算
# ──────────────────────────────────────────────────────────────────────────────

def _get_apply_kwargs(
    iteration: int,
    strategy: str,
    profile: IntentProfile,
) -> dict:
    """
    イテレーション・戦略・IntentProfileを組み合わせてapply_modelパラメータを決定。
    
    iter=0: profileのデフォルト値を使う
    iter=1+: strategyに応じて強化
    """
    base_overlap = profile.overlap
    base_shifts  = profile.shifts

    if iteration == 0:
        return {"shifts": base_shifts, "overlap": base_overlap}

    if strategy == "shifts":
        # shifts を増やして位相ゆらぎを平均化
        return {"shifts": base_shifts + iteration, "overlap": base_overlap}
    elif strategy == "overlap":
        # overlap を上げてセグメント境界ノイズを減らす
        overlap = min(base_overlap + iteration * 0.1, 0.5)
        return {"shifts": base_shifts, "overlap": overlap}
    elif strategy == "model_upgrade":
        # モデル変更は呼び出し側で制御（ここではパラメータ据え置き）
        return {"shifts": base_shifts + 1, "overlap": base_overlap}
    else:
        return {"shifts": base_shifts, "overlap": base_overlap}


# ──────────────────────────────────────────────────────────────────────────────
# エンジン本体
# ──────────────────────────────────────────────────────────────────────────────

class MetaCogEngine:
    """
    Intent-Aware Temporal Metacognition Engine for stem separation.

    ループ構造:
        profile = get_intent_profile(intent)
        for i in range(max_iterations):
            stems = demucs.separate(audio, **strategy_params(i, profile))
            report = quality.evaluate(original, stems,
                                      thresholds=profile.to_quality_thresholds(),
                                      stem_snr_thresholds=profile.stem_snr_thresholds)
            if report.overall_passed:
                break
            # 失敗ステムを特定（Targeted awareness）
            # 次イテレーションで同じモデルを強化 or モデル切り替え
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._separators: dict[str, DemucsBase] = {}  # モデルキャッシュ

    def _get_separator(self, model_name: str) -> DemucsBase:
        if model_name not in self._separators:
            self._separators[model_name] = DemucsBase(
                model_name=model_name,
                device=self.config.device,
            )
        return self._separators[model_name]

    def _separate_with_strategy(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        iteration: int,
        model_name: str,
        profile: IntentProfile,
    ) -> dict[str, torch.Tensor]:
        """分離実行。"""
        from demucs.apply import apply_model

        sep = self._get_separator(model_name)
        sep._load()
        model = sep._model

        target_sr = model.samplerate
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        kwargs = _get_apply_kwargs(iteration, self.config.retry_strategy, profile)
        if self.config.verbose:
            logger.info(f"[Iter {iteration}] model={model_name} params={kwargs}")

        wav_batch = waveform.unsqueeze(0).to(sep.device)
        with torch.no_grad():
            sources = apply_model(
                model, wav_batch,
                device=sep.device,
                split=True,
                progress=False,
                **kwargs,
            )

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
        # ── Intentプロファイルを解決 ──────────────────────────────
        profile = get_intent_profile(self.config.intent)
        if self.config.verbose:
            logger.info(f"Intent: {profile.intent.value} — {profile.description}")

        # EngineConfigの明示的な閾値でオーバーライドも可能
        quality_thresholds = {
            **profile.to_quality_thresholds(),
            **self.config.quality_thresholds,
        }
        stem_snr_thresholds = profile.stem_snr_thresholds.copy()

        total_start = time.perf_counter()
        iteration_results: list[IterationResult] = []
        best_stems: Optional[dict[str, torch.Tensor]] = None

        # 現在使用するモデル（再試行時に切り替わる可能性あり）
        current_model = self.config.model_name

        logger.info(
            f"MetaCogEngine starting. "
            f"intent={profile.intent.value} max_iter={self.config.max_iterations} "
            f"model={current_model}"
        )

        for i in range(self.config.max_iterations):
            iter_start = time.perf_counter()
            logger.info(f"\n--- Iteration {i} (model: {current_model}) ---")

            # ── 1. 分離（直感パス）──────────────────────────────────
            stems = self._separate_with_strategy(
                waveform, sample_rate,
                iteration=i,
                model_name=current_model,
                profile=profile,
            )

            # ── 2. 品質監査（監査層）────────────────────────────────
            report = evaluate_stems(
                original=waveform,
                stems=stems,
                thresholds=quality_thresholds,
                stem_snr_thresholds=stem_snr_thresholds,
            )
            report.iteration = i
            elapsed = time.perf_counter() - iter_start

            iteration_results.append(IterationResult(
                iteration=i,
                stems=stems,
                report=report,
                elapsed_sec=round(elapsed, 2),
                model_used=current_model,
                intent_applied=profile.intent.value,
            ))

            if self.config.verbose:
                print(report.summary())

            best_stems = stems

            # ── 3. 合格判定 ────────────────────────────────────────
            if report.overall_passed:
                logger.info(f"✅ Quality passed at iteration {i}")
                break

            # ── 4. 失敗分析とモデル切り替え判断 ──────────────────
            if i < self.config.max_iterations - 1:
                failed = report.failed_stems
                logger.info(f"⚠️  Failed stems: {failed}. Adjusting strategy...")

                # model_upgradeストラテジー: profile指定のretry_modelに切り替える
                if (
                    self.config.retry_strategy == "model_upgrade"
                    and profile.retry_model
                    and current_model != profile.retry_model
                ):
                    logger.info(
                        f"  Upgrading model: {current_model} → {profile.retry_model}"
                    )
                    current_model = profile.retry_model
            else:
                logger.info(f"⚠️  Max iterations reached. Returning best-effort result.")

        total_elapsed = time.perf_counter() - total_start
        sep = self._get_separator(current_model)

        result = EngineResult(
            final_stems=best_stems,
            sample_rate=sep.sample_rate,
            iterations=iteration_results,
            total_elapsed_sec=round(total_elapsed, 2),
            intent=profile.intent.value,
        )

        return result
