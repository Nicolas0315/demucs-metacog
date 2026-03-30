"""
engine.py — Temporal Metacognition Engine v3

v3の追加:
- Dual-Model並列実行: htdemucs と htdemucs_ft を同時に実行し、
  PerceptualScores（MAPSS近似）で良い方を自動選択する。
  これがHuang Limitへの直接的な回答: 外部評価器（perceptual.py）が
  同一モデルの内部バイアスから独立した判断を下す。
- VAD統合: vocalsステムに音声活動がなければ早期に警告し、
  無音ステムへの再試行リソースを無駄にしない。
- PerceptualScores をEngineResultに含め、CLI出力に反映する。

v2:
- Intent-Aware: IntentProfileを受け取り、用途に応じた閾値・戦略を適用
- Targeted Re-separation: 失敗ステムのリスト化
- モデルアップグレード戦略
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import torch
import torchaudio

from .intent import IntentProfile, SeparationIntent, get_intent_profile
from .perceptual import PerceptualScores, evaluate_perceptual
from .quality import QualityReport, evaluate_stems
from .separator import DemucsBase
from .vad import VoiceActivity, audit_stems_vad

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


@dataclass
class EngineConfig:
    """エンジン設定。"""
    model_name: str = "htdemucs"
    device: Optional[str] = None
    max_iterations: int = MAX_ITERATIONS
    quality_thresholds: dict = field(default_factory=dict)
    retry_strategy: str = "shifts"   # "shifts" | "overlap" | "model_upgrade" | "dual_model"
    save_iterations: bool = False
    verbose: bool = True
    # Intent-Aware設定
    intent: SeparationIntent | str = SeparationIntent.DEFAULT
    # Dual-Model設定
    dual_model_name: str = "htdemucs_ft"   # dual_model戦略で使う対抗モデル
    use_perceptual_eval: bool = True        # MAPSS近似評価を使う
    use_vad: bool = True                    # VAD評価を使う


@dataclass
class IterationResult:
    iteration: int
    stems: dict[str, torch.Tensor]
    report: QualityReport
    elapsed_sec: float
    model_used: str = "htdemucs"
    intent_applied: str = "default"
    perceptual_scores: dict[str, PerceptualScores] = field(default_factory=dict)
    vad_results: dict[str, VoiceActivity] = field(default_factory=dict)


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
            # Perceptualスコアがあれば表示
            if r.perceptual_scores:
                method = next(iter(r.perceptual_scores.values())).method
                lines.append(f"  [Perceptual eval — {method}]")
                for name, ps in r.perceptual_scores.items():
                    lines.append(
                        f"    {name:8s}  PS={ps.ps_score:.3f}  PM={ps.pm_score:.3f}  "
                        f"combined={ps.combined:.3f}"
                    )
            # VAD結果があれば表示
            if r.vad_results:
                lines.append("  [VAD]")
                for name, vad in r.vad_results.items():
                    activity = f"{vad.activity_ratio:.1%}"
                    flag = " ⚠️ LOW" if not vad.has_activity else ""
                    lines.append(f"    {name:8s}  activity={activity}{flag}")
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

    def _dual_model_run(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        iteration: int,
        profile: IntentProfile,
    ) -> tuple[dict[str, torch.Tensor], str]:
        """
        Dual-Model並列実行 (Huang Limit突破戦略)。

        model_name と dual_model_name を ThreadPoolExecutor で並列実行し、
        PerceptualScores（MAPSS近似）で combined スコアが高い方を選択する。
        同一モデルの内部バイアスに依存しない「外部評価器」として機能する。
        """
        model_a = self.config.model_name
        model_b = self.config.dual_model_name
        logger.info(f"[Dual-Model] Parallel: {model_a} vs {model_b}")

        def run_model(name: str) -> tuple[str, dict[str, torch.Tensor]]:
            stems = self._separate_with_strategy(
                waveform, sample_rate, iteration=iteration,
                model_name=name, profile=profile,
            )
            return name, stems

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(run_model, m): m for m in [model_a, model_b]}
            results: dict[str, dict[str, torch.Tensor]] = {}
            for fut in as_completed(futures):
                name, stems = fut.result()
                results[name] = stems

        # ── PerceptualScoreで比較 ─────────────────────────────
        sep_a = results[model_a]
        sep_b = results[model_b]

        score_a = evaluate_perceptual(waveform, sep_a, sample_rate, use_wavlm=False)
        score_b = evaluate_perceptual(waveform, sep_b, sample_rate, use_wavlm=False)

        avg_a = sum(s.combined for s in score_a.values()) / max(len(score_a), 1)
        avg_b = sum(s.combined for s in score_b.values()) / max(len(score_b), 1)

        winner = model_a if avg_a >= avg_b else model_b
        winner_stems = results[winner]

        logger.info(
            f"[Dual-Model] {model_a}={avg_a:.3f}  {model_b}={avg_b:.3f}  "
            f"→ winner: {winner}"
        )
        return winner_stems, winner

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

        quality_thresholds = {
            **profile.to_quality_thresholds(),
            **self.config.quality_thresholds,
        }
        stem_snr_thresholds = profile.stem_snr_thresholds.copy()

        total_start = time.perf_counter()
        iteration_results: list[IterationResult] = []
        best_stems: Optional[dict[str, torch.Tensor]] = None
        current_model = self.config.model_name

        logger.info(
            f"MetaCogEngine v3 starting. "
            f"intent={profile.intent.value} max_iter={self.config.max_iterations} "
            f"model={current_model} strategy={self.config.retry_strategy}"
        )

        for i in range(self.config.max_iterations):
            iter_start = time.perf_counter()
            logger.info(f"\n--- Iteration {i} (model: {current_model}) ---")

            # ── 1. 分離（直感パス）──────────────────────────────────
            if self.config.retry_strategy == "dual_model":
                stems, current_model = self._dual_model_run(
                    waveform, sample_rate, iteration=i, profile=profile,
                )
            else:
                stems = self._separate_with_strategy(
                    waveform, sample_rate,
                    iteration=i, model_name=current_model, profile=profile,
                )

            # ── 2. 品質監査（SNR/Leakage）──────────────────────────
            report = evaluate_stems(
                original=waveform,
                stems=stems,
                thresholds=quality_thresholds,
                stem_snr_thresholds=stem_snr_thresholds,
            )
            report.iteration = i

            # ── 3. Perceptual評価（MAPSS近似）──────────────────────
            perceptual: dict[str, PerceptualScores] = {}
            if self.config.use_perceptual_eval:
                try:
                    perceptual = evaluate_perceptual(
                        waveform, stems, sample_rate, use_wavlm=False
                    )
                except Exception as e:
                    logger.warning(f"Perceptual eval failed: {e}")

            # ── 4. VAD（音声活動検出）──────────────────────────────
            vad_results: dict[str, VoiceActivity] = {}
            if self.config.use_vad:
                try:
                    vad_results = audit_stems_vad(stems, sample_rate)
                    # vocalsが無声なら注意ノートを追加
                    if "vocals" in vad_results and not vad_results["vocals"].has_activity:
                        report.notes.append(
                            "VAD: vocals has no detectable activity — "
                            "may be instrumental or separation failed"
                        )
                except Exception as e:
                    logger.warning(f"VAD eval failed: {e}")

            elapsed = time.perf_counter() - iter_start

            iteration_results.append(IterationResult(
                iteration=i,
                stems=stems,
                report=report,
                elapsed_sec=round(elapsed, 2),
                model_used=current_model,
                intent_applied=profile.intent.value,
                perceptual_scores=perceptual,
                vad_results=vad_results,
            ))

            if self.config.verbose:
                print(report.summary())

            best_stems = stems

            # ── 5. 合格判定 ────────────────────────────────────────
            if report.overall_passed:
                logger.info(f"✅ Quality passed at iteration {i}")
                break

            # ── 6. 失敗分析とモデル切り替え判断 ──────────────────
            if i < self.config.max_iterations - 1:
                failed = report.failed_stems
                logger.info(f"⚠️  Failed stems: {failed}. Adjusting strategy...")

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
