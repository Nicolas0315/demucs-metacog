"""
quality.py — ステム品質評価モジュール v2

v1の問題点を修正:
- クロストーク計算が「他ステムより強いか」を測っていただけで意味がなかった
- 真のクロストークはリファレンスなしには計算不可能

v2の方針:
- クロストークを廃止し、代わりに以下の2指標を使う
  1. SNR (自己SNR近似): ステムのエネルギー vs 全ステムの残差エネルギー
  2. Leakage ratio: 「このステムに含まれる他ステム由来成分の割合」を
                    スペクトル差分で近似する
- IntentProfileによるper-stem閾値を受け取れるように拡張

Katala 2-3-1モデルのうち「冷徹な監査役」に相当。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class StemQuality:
    """1ステムの品質スコアセット。"""
    stem: str
    snr_db: float            # Signal-to-Noise Ratio（高いほど良い）
    energy_ratio: float      # この stem の全体エネルギー比
    leakage_ratio: float     # 他ステム由来と推定される成分の割合（低いほど良い）
    passed: bool = False
    reason: str = ""


@dataclass
class QualityReport:
    """全ステムの品質レポート。"""
    stems: dict[str, StemQuality] = field(default_factory=dict)
    overall_passed: bool = False
    retry_needed: bool = False
    iteration: int = 0
    notes: list[str] = field(default_factory=list)
    # どのステムが失敗したか（Targeted Re-separationの判断材料）
    failed_stems: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"[Iter {self.iteration}] Overall: {'✅ PASS' if self.overall_passed else '❌ RETRY'}"]
        for name, q in self.stems.items():
            icon = "✅" if q.passed else "⚠️"
            lines.append(
                f"  {icon} {name:8s}  SNR={q.snr_db:+.1f}dB  "
                f"energy={q.energy_ratio:.2%}  leakage={q.leakage_ratio:.2%}"
            )
        if self.notes:
            lines.append("  Notes: " + " | ".join(self.notes))
        if self.failed_stems:
            lines.append(f"  Failed: {', '.join(self.failed_stems)}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# デフォルト閾値
# ──────────────────────────────────────────────
SNR_THRESHOLD_DB = 8.0
MAX_LEAKAGE_RATIO = 0.35     # 他ステム由来が35%以下ならOK
MIN_ENERGY_RATIO = 0.01


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)) + 1e-10)


def _db(ratio: float) -> float:
    return 20.0 * math.log10(max(ratio, 1e-10))


def _leakage(stem_arr: np.ndarray, other_stems: list[np.ndarray]) -> float:
    """
    Leakage ratio: このステムに「他ステムの成分が混入している割合」の近似。

    手法: 「sum of |other| が stem に対してどれだけ大きいか」をスペクトル的に推定。
    真の計算にはリファレンスが必要だが、ここでは以下の代理指標を使う:
    
    leakage ≈ min(1, RMS(Σ|other|) / (RMS(stem) + RMS(Σ|other|)))
    
    = 「他ステムの合計エネルギーが全体に占める割合」
    これはゼロに近いほど干渉が少ない（クリーンな分離）。
    """
    if not other_stems:
        return 0.0

    other_mix = np.zeros_like(stem_arr)
    for o in other_stems:
        # 長さを揃える
        length = min(len(stem_arr), len(o))
        other_mix[:length] += np.abs(o[:length])

    stem_rms = _rms(stem_arr)
    other_rms = _rms(other_mix)
    return float(other_rms / (stem_rms + other_rms + 1e-10))


def evaluate_stems(
    original: torch.Tensor,
    stems: dict[str, torch.Tensor],
    thresholds: Optional[dict] = None,
    # IntentProfileからのper-stem SNR閾値（任意）
    stem_snr_thresholds: Optional[dict[str, float]] = None,
) -> QualityReport:
    """
    Args:
        original: ミックス前のオリジナル (channels, samples)
        stems:    分離後の各ステム dict
        thresholds: グローバル閾値オーバーライド
        stem_snr_thresholds: ステム固有SNR閾値（IntentProfile用）

    Returns:
        QualityReport
    """
    thr_snr     = (thresholds or {}).get("snr_db", SNR_THRESHOLD_DB)
    thr_leakage = (thresholds or {}).get("max_leakage_ratio", MAX_LEAKAGE_RATIO)
    thr_energy  = (thresholds or {}).get("min_energy_ratio", MIN_ENERGY_RATIO)

    orig_np = _to_numpy(original.mean(0))
    orig_rms = _rms(orig_np)

    stems_np: dict[str, np.ndarray] = {
        k: _to_numpy(v.mean(0)) for k, v in stems.items()
    }

    stem_qualities: dict[str, StemQuality] = {}
    all_passed = True
    failed_stems: list[str] = []

    for name, arr in stems_np.items():
        # ── エネルギー比 ───────────────────────────────────────
        stem_rms = _rms(arr)
        energy_ratio = stem_rms / (orig_rms + 1e-10)

        # ── SNR: 元信号との差分をノイズとして扱う近似 ────────────
        length = min(len(orig_np), len(arr))
        residual = orig_np[:length] - arr[:length]
        snr = _db(stem_rms / (_rms(residual) + 1e-10))

        # ── Leakage ratio ────────────────────────────────────
        others = [v for k, v in stems_np.items() if k != name]
        leakage = _leakage(arr, others)

        # ── ステム固有SNR閾値を取得 ───────────────────────────
        effective_snr_thr = (stem_snr_thresholds or {}).get(name, thr_snr)

        # ── 判定 ──────────────────────────────────────────────
        fail_reasons = []
        if snr < effective_snr_thr:
            fail_reasons.append(f"SNR low ({snr:.1f}dB < {effective_snr_thr}dB)")
        if leakage > thr_leakage:
            fail_reasons.append(f"leakage high ({leakage:.1%} > {thr_leakage:.1%})")
        if energy_ratio < thr_energy:
            fail_reasons.append(f"energy too low ({energy_ratio:.2%})")

        passed = len(fail_reasons) == 0
        if not passed:
            all_passed = False
            failed_stems.append(name)

        stem_qualities[name] = StemQuality(
            stem=name,
            snr_db=round(snr, 2),
            energy_ratio=round(energy_ratio, 4),
            leakage_ratio=round(leakage, 4),
            passed=passed,
            reason="; ".join(fail_reasons),
        )

    notes = []
    if "vocals" in stem_qualities:
        if stem_qualities["vocals"].energy_ratio < 0.05:
            notes.append("vocals energy very low — may be instrumental track")

    # 全ステムのleakageが高い場合 = モデル自体が苦手な素材
    avg_leakage = np.mean([q.leakage_ratio for q in stem_qualities.values()])
    if avg_leakage > 0.45:
        notes.append(f"avg leakage {avg_leakage:.1%} — consider htdemucs_ft or different model")

    report = QualityReport(
        stems=stem_qualities,
        overall_passed=all_passed,
        retry_needed=not all_passed,
        notes=notes,
        failed_stems=failed_stems,
    )
    return report
