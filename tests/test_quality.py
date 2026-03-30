"""
test_quality.py — 品質評価モジュールのユニットテスト (v2)
実際の音声ファイルなしで動作する（合成信号で検証）。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from demucs_metacog.quality import evaluate_stems, QualityReport, StemQuality


def _make_sine(freq: float, duration: float = 2.0, sr: int = 44100, amp: float = 0.5) -> torch.Tensor:
    t = torch.linspace(0, duration, int(sr * duration))
    wave = amp * torch.sin(2 * math.pi * freq * t)
    return wave.unsqueeze(0).repeat(2, 1)


class TestQualityBasic:

    def test_returns_quality_report(self):
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.4),
            "drums":  _make_sine(80.0, amp=0.3),
            "bass":   _make_sine(60.0, amp=0.2),
            "other":  _make_sine(880.0, amp=0.1),
        }
        report = evaluate_stems(original, stems)
        assert isinstance(report, QualityReport)

    def test_all_stems_evaluated(self):
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums":  _make_sine(80.0, amp=0.3),
            "bass":   _make_sine(60.0, amp=0.2),
            "other":  _make_sine(880.0, amp=0.1),
        }
        report = evaluate_stems(original, stems)
        assert set(report.stems.keys()) == {"vocals", "drums", "bass", "other"}

    def test_snr_is_float(self):
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        assert isinstance(report.stems["vocals"].snr_db, float)

    def test_leakage_ratio_in_range(self):
        """leakage_ratioが0〜1の範囲に収まること。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums":  _make_sine(80.0, amp=0.3),
        }
        report = evaluate_stems(original, stems)
        for name, q in report.stems.items():
            assert 0.0 <= q.leakage_ratio <= 1.0, f"{name} leakage out of range"

    def test_energy_ratio_positive(self):
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums":  _make_sine(80.0, amp=0.3),
        }
        report = evaluate_stems(original, stems)
        for name, q in report.stems.items():
            assert q.energy_ratio >= 0.0


class TestLeakageMetric:

    def test_isolated_signal_has_low_leakage(self):
        """他ステムが完全に別周波数ならleakageが低いこと（相対的に）。"""
        original = _make_sine(440.0)
        # 全く異なる周波数のステム
        stems = {
            "vocals": _make_sine(440.0, amp=0.8),
            "drums":  _make_sine(5000.0, amp=0.01),  # 超小さい他ステム
        }
        report = evaluate_stems(original, stems)
        # vocalsは他(drums)がほぼ無いのでleakageが低いはず
        assert report.stems["vocals"].leakage_ratio < 0.1

    def test_equal_energy_stems_have_moderate_leakage(self):
        """同等エネルギーのステムが複数あればleakageが適度に高い。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums":  _make_sine(80.0, amp=0.5),
            "bass":   _make_sine(60.0, amp=0.5),
        }
        report = evaluate_stems(original, stems)
        # 各ステムは全エネルギーの約1/3 → leakageは約0.67になる
        for name, q in report.stems.items():
            assert q.leakage_ratio > 0.3, f"{name} leakage unexpectedly low"


class TestQualityThresholds:

    def test_very_strict_threshold_fails(self):
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.3)}
        report = evaluate_stems(original, stems, thresholds={"snr_db": 999.0})
        assert report.stems["vocals"].passed is False
        assert report.overall_passed is False
        assert report.retry_needed is True

    def test_all_loose_thresholds_passes(self):
        """全閾値を最大限に緩めれば必ずpassすること。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems, thresholds={
            "snr_db": -999.0,
            "max_leakage_ratio": 1.0,
            "min_energy_ratio": 0.0,
        })
        assert report.stems["vocals"].passed is True

    def test_stem_snr_thresholds_per_stem(self):
        """per-stem SNR閾値が正しく適用されること。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums":  _make_sine(80.0, amp=0.3),
        }
        # vocalsは超厳しく、drumsは超緩く
        report = evaluate_stems(
            original, stems,
            thresholds={"snr_db": 0.0, "max_leakage_ratio": 1.0, "min_energy_ratio": 0.0},
            stem_snr_thresholds={"vocals": 999.0, "drums": -999.0},
        )
        # vocalsは失敗、drumsはSNR条件はpass（leakageも緩いので通る）
        assert report.stems["vocals"].passed is False
        assert report.stems["drums"].passed is True

    def test_failed_stems_list(self):
        """失敗したステムがfailed_stemsに記録されること。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums":  _make_sine(80.0, amp=0.3),
        }
        report = evaluate_stems(original, stems, thresholds={
            "snr_db": 999.0,  # 全部失敗
        })
        assert len(report.failed_stems) == 2
        assert "vocals" in report.failed_stems
        assert "drums" in report.failed_stems


class TestQualityReport:

    def test_summary_contains_stem_names(self):
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums":  _make_sine(80.0, amp=0.3),
        }
        report = evaluate_stems(original, stems)
        summary = report.summary()
        assert "vocals" in summary
        assert "drums" in summary

    def test_summary_shows_leakage(self):
        """summary()にleakage指標が含まれること（v2の変更）。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        summary = report.summary()
        assert "leakage" in summary

    def test_iteration_in_summary(self):
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        report.iteration = 2
        assert "2" in report.summary()


class TestQualityEdgeCases:

    def test_single_stem(self):
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        assert "vocals" in report.stems

    def test_silent_stem_low_energy(self):
        original = _make_sine(440.0, amp=0.8)
        stems = {
            "vocals": torch.zeros(2, original.shape[1]),
            "other":  _make_sine(880.0, amp=0.5),
        }
        report = evaluate_stems(original, stems)
        assert report.stems["vocals"].energy_ratio < 0.01

    def test_notes_for_silent_vocals(self):
        original = _make_sine(440.0, amp=0.8)
        stems = {
            "vocals": torch.zeros(2, original.shape[1]),
            "drums":  _make_sine(80.0, amp=0.5),
        }
        report = evaluate_stems(original, stems)
        joined = " ".join(report.notes)
        # noteがある場合は "vocals" という文字列が含まれる
        if report.notes:
            assert "vocals" in joined
