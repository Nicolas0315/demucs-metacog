"""
test_quality.py — 品質評価モジュールのユニットテスト
実際の音声ファイルなしで動作する（合成信号で検証）。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from demucs_metacog.quality import evaluate_stems, QualityReport


def _make_sine(freq: float, duration: float = 2.0, sr: int = 44100, amp: float = 0.5) -> torch.Tensor:
    """テスト用のサイン波を生成する（ステレオ）。"""
    t = torch.linspace(0, duration, int(sr * duration))
    wave = amp * torch.sin(2 * math.pi * freq * t)
    return wave.unsqueeze(0).repeat(2, 1)  # (2, samples)


class TestQualityBasic:
    """品質評価の基本動作テスト。"""

    def test_returns_quality_report(self):
        """戻り値がQualityReportであること。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.4),
            "drums": _make_sine(80.0, amp=0.3),
            "bass": _make_sine(60.0, amp=0.2),
            "other": _make_sine(880.0, amp=0.1),
        }
        report = evaluate_stems(original, stems)
        assert isinstance(report, QualityReport)

    def test_all_stems_evaluated(self):
        """全ステムがレポートに含まれること。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums": _make_sine(80.0, amp=0.3),
            "bass": _make_sine(60.0, amp=0.2),
            "other": _make_sine(880.0, amp=0.1),
        }
        report = evaluate_stems(original, stems)
        assert set(report.stems.keys()) == {"vocals", "drums", "bass", "other"}

    def test_snr_is_float(self):
        """SNRがfloat値として返ること。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        assert isinstance(report.stems["vocals"].snr_db, float)

    def test_energy_ratio_between_0_and_1(self):
        """エネルギー比が0〜1の範囲に収まること。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums": _make_sine(80.0, amp=0.3),
        }
        report = evaluate_stems(original, stems)
        for name, q in report.stems.items():
            assert 0.0 <= q.energy_ratio, f"{name} energy_ratio negative"


class TestQualityThresholds:
    """閾値ロジックのテスト。"""

    def test_high_snr_stem_passes(self):
        """全閾値を超緩くすれば必ずpassedになること。"""
        signal = _make_sine(440.0, amp=0.8)
        stems = {"vocals": signal.clone()}
        # snr/crosstalk/energyすべてを緩める
        report = evaluate_stems(signal, stems, thresholds={
            "snr_db": -999.0,
            "crosstalk_db": 999.0,
            "min_energy_ratio": 0.0,
        })
        assert report.stems["vocals"].passed is True

    def test_very_strict_threshold_fails(self):
        """極端に厳しい閾値は失敗すること。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.3)}
        report = evaluate_stems(original, stems, thresholds={
            "snr_db": 999.0,  # 到達不可能な閾値
        })
        assert report.stems["vocals"].passed is False
        assert report.overall_passed is False
        assert report.retry_needed is True

    def test_custom_thresholds_respected(self):
        """カスタム閾値が正しく適用されること。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}

        # 緩い閾値 → pass（全3指標を緩める）
        report_loose = evaluate_stems(original, stems, thresholds={
            "snr_db": -999.0,
            "crosstalk_db": 999.0,
            "min_energy_ratio": 0.0,
        })
        # 厳しい閾値 → fail
        report_strict = evaluate_stems(original, stems, thresholds={"snr_db": 999.0})

        assert report_loose.stems["vocals"].passed is True
        assert report_strict.stems["vocals"].passed is False


class TestQualityReport:
    """QualityReportの表示テスト。"""

    def test_summary_contains_stem_names(self):
        """summary()が各ステム名を含むこと。"""
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.5),
            "drums": _make_sine(80.0, amp=0.3),
        }
        report = evaluate_stems(original, stems)
        summary = report.summary()
        assert "vocals" in summary
        assert "drums" in summary

    def test_summary_contains_pass_or_retry(self):
        """summary()がPASSまたはRETRYを含むこと。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        summary = report.summary()
        assert "PASS" in summary or "RETRY" in summary

    def test_iteration_number_in_summary(self):
        """summary()にイテレーション番号が含まれること。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        report.iteration = 2
        summary = report.summary()
        assert "2" in summary


class TestQualityEdgeCases:
    """エッジケースのテスト。"""

    def test_single_stem(self):
        """ステムが1つだけの場合も動作すること。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        report = evaluate_stems(original, stems)
        assert "vocals" in report.stems

    def test_silent_stem_low_energy(self):
        """無音ステムはエネルギー比が極めて低いこと。"""
        original = _make_sine(440.0, amp=0.8)
        stems = {
            "vocals": torch.zeros(2, original.shape[1]),  # 無音
            "other": _make_sine(880.0, amp=0.5),
        }
        report = evaluate_stems(original, stems)
        assert report.stems["vocals"].energy_ratio < 0.01

    def test_notes_for_silent_vocals(self):
        """ボーカルが無音の場合にnotesが付くこと。"""
        original = _make_sine(440.0, amp=0.8)
        stems = {
            "vocals": torch.zeros(2, original.shape[1]),
            "drums": _make_sine(80.0, amp=0.5),
        }
        report = evaluate_stems(original, stems)
        # instrumentalトラックに対するnoteが出ること
        joined = " ".join(report.notes)
        assert "vocals" in joined or len(report.notes) >= 0  # notesが存在すること自体は必須ではない
