"""
quality.py — ステム品質評価モジュール
分離後のステムに対してSNR・クロストーク・エネルギー比を計算し、
再分離が必要かを判定する「監査層」。

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
    crosstalk_db: float      # 他ステムとのクロストーク（低いほど良い）
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

    def summary(self) -> str:
        lines = [f"[Iter {self.iteration}] Overall: {'✅ PASS' if self.overall_passed else '❌ RETRY'}"]
        for name, q in self.stems.items():
            icon = "✅" if q.passed else "⚠️"
            lines.append(
                f"  {icon} {name:8s}  SNR={q.snr_db:+.1f}dB  "
                f"energy={q.energy_ratio:.2%}  xtalk={q.crosstalk_db:+.1f}dB"
            )
        if self.notes:
            lines.append("  Notes: " + " | ".join(self.notes))
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 閾値設定（チューニング可能）
# ──────────────────────────────────────────────
SNR_THRESHOLD_DB = 8.0          # これ以上ならOK
CROSSTALK_THRESHOLD_DB = -12.0  # これ以下（より負）ならOK
MIN_ENERGY_RATIO = 0.01         # 各ステムが全体の1%以上エネルギーを持っていること


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _rms(arr: np.ndarray) -> float:
    """Root Mean Square（エネルギー指標）。"""
    return float(np.sqrt(np.mean(arr ** 2)) + 1e-10)


def _db(ratio: float) -> float:
    return 20.0 * math.log10(max(ratio, 1e-10))


def evaluate_stems(
    original: torch.Tensor,
    stems: dict[str, torch.Tensor],
    thresholds: Optional[dict] = None,
) -> QualityReport:
    """
    Args:
        original: ミックス前のオリジナル (channels, samples)
        stems:    分離後の各ステム dict
        thresholds: 閾値オーバーライド（任意）

    Returns:
        QualityReport
    """
    thr_snr = (thresholds or {}).get("snr_db", SNR_THRESHOLD_DB)
    thr_xtalk = (thresholds or {}).get("crosstalk_db", CROSSTALK_THRESHOLD_DB)
    thr_energy = (thresholds or {}).get("min_energy_ratio", MIN_ENERGY_RATIO)

    orig_np = _to_numpy(original.mean(0))  # モノ化
    orig_rms = _rms(orig_np)

    # 各ステムをモノ化
    stems_np: dict[str, np.ndarray] = {
        k: _to_numpy(v.mean(0)) for k, v in stems.items()
    }
    total_stem_energy = sum(_rms(v) ** 2 for v in stems_np.values()) ** 0.5

    stem_qualities: dict[str, StemQuality] = {}
    all_passed = True

    for name, arr in stems_np.items():
        stem_rms = _rms(arr)

        # エネルギー比
        energy_ratio = (stem_rms / (orig_rms + 1e-10))

        # SNR: 元の信号とステムの差分をノイズとみなす
        # ここでは「元音全体 vs このステム」の簡易近似
        # 真のSNRにはリファレンスが必要だが、ここは相対的な自己チェック
        residual = orig_np - arr
        snr = _db(stem_rms / (_rms(residual) + 1e-10))

        # クロストーク: 他のステム群との相関パワー
        other_mix = sum(
            other_arr for k, other_arr in stems_np.items() if k != name
        )
        xtalk = _db(stem_rms / (_rms(other_mix) + 1e-10)) - 6.0  # 補正係数

        # 各ステム判定
        fail_reasons = []
        if snr < thr_snr:
            fail_reasons.append(f"SNR low ({snr:.1f}dB < {thr_snr}dB)")
        if xtalk > thr_xtalk:
            fail_reasons.append(f"crosstalk high ({xtalk:.1f}dB > {thr_xtalk}dB)")
        if energy_ratio < thr_energy:
            fail_reasons.append(f"energy too low ({energy_ratio:.2%})")

        passed = len(fail_reasons) == 0
        if not passed:
            all_passed = False

        stem_qualities[name] = StemQuality(
            stem=name,
            snr_db=round(snr, 2),
            energy_ratio=round(energy_ratio, 4),
            crosstalk_db=round(xtalk, 2),
            passed=passed,
            reason="; ".join(fail_reasons),
        )

    notes = []
    # ボーカルが異常に低いエネルギーなら特記
    if "vocals" in stem_qualities:
        if stem_qualities["vocals"].energy_ratio < 0.05:
            notes.append("vocals energy very low — may be instrumental track")

    report = QualityReport(
        stems=stem_qualities,
        overall_passed=all_passed,
        retry_needed=not all_passed,
        notes=notes,
    )
    return report
