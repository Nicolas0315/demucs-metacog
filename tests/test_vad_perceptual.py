"""
test_vad_perceptual.py — VAD と Perceptual モジュールのユニットテスト
実音声ファイルなし（合成信号）で動作する。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
import numpy as np

from demucs_metacog.vad import (
    VoiceActivity,
    detect_voice_activity,
    audit_stems_vad,
    STEM_ACTIVITY_FLOOR,
)
from demucs_metacog.perceptual import (
    PerceptualScores,
    evaluate_perceptual,
    _cosine_similarity,
    _mel_filterbank_energy,
)


def _make_sine(freq: float, duration: float = 2.0, sr: int = 44100, amp: float = 0.5):
    t = torch.linspace(0, duration, int(sr * duration))
    wave = amp * torch.sin(2 * math.pi * freq * t)
    return wave.unsqueeze(0).repeat(2, 1)


def _make_silence(duration: float = 2.0, sr: int = 44100):
    n = int(sr * duration)
    return torch.zeros(2, n)


# ──────────────────────────────────────────────────────────────
# VADテスト
# ──────────────────────────────────────────────────────────────

class TestVoiceActivity:

    def test_active_signal_detected(self):
        """音量があるステムはhas_activity=Trueになること。"""
        stem = _make_sine(440.0, amp=0.5)
        result = detect_voice_activity(stem, 44100, "vocals")
        assert result.has_activity is True

    def test_silent_signal_not_detected(self):
        """無音ステムはhas_activity=Falseになること。"""
        stem = _make_silence()
        result = detect_voice_activity(stem, 44100, "vocals")
        assert result.has_activity is False

    def test_activity_ratio_high_for_continuous_signal(self):
        """連続するサイン波はactivity_ratioが高いこと。"""
        stem = _make_sine(440.0, amp=0.5)
        result = detect_voice_activity(stem, 44100)
        assert result.activity_ratio > 0.8

    def test_activity_ratio_zero_for_silence(self):
        stem = _make_silence()
        result = detect_voice_activity(stem, 44100)
        assert result.activity_ratio == 0.0

    def test_stem_name_recorded(self):
        stem = _make_sine(440.0)
        result = detect_voice_activity(stem, 44100, stem_name="drums")
        assert result.stem == "drums"

    def test_durations_sum_close_to_total(self):
        """speech + silenceの合計が入力長に近いこと。"""
        duration = 2.0
        stem = _make_sine(440.0, duration=duration)
        result = detect_voice_activity(stem, 44100)
        total = result.speech_duration_sec + result.silence_duration_sec
        # ±10%程度の誤差は許容（フレームのはみ出し分）
        assert abs(total - duration) < duration * 0.15


class TestAuditStemsVAD:

    def test_all_stems_audited(self):
        stems = {
            "vocals": _make_sine(440.0),
            "drums":  _make_sine(80.0),
            "bass":   _make_silence(),
        }
        results = audit_stems_vad(stems, 44100)
        assert set(results.keys()) == {"vocals", "drums", "bass"}

    def test_silent_stem_flagged(self):
        stems = {
            "vocals": _make_sine(440.0),
            "bass":   _make_silence(),
        }
        results = audit_stems_vad(stems, 44100)
        assert results["bass"].has_activity is False
        assert results["vocals"].has_activity is True

    def test_returns_voice_activity_instances(self):
        stems = {"vocals": _make_sine(440.0)}
        results = audit_stems_vad(stems, 44100)
        assert isinstance(results["vocals"], VoiceActivity)


# ──────────────────────────────────────────────────────────────
# Perceptualテスト
# ──────────────────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-5

    def test_opposite_vectors_return_zero(self):
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([-1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(v, w) - 0.0) < 1e-5

    def test_output_in_zero_one(self):
        v = np.random.randn(40)
        w = np.random.randn(40)
        sim = _cosine_similarity(v, w)
        assert 0.0 <= sim <= 1.0


class TestMelFilterbankEnergy:

    def test_returns_vector(self):
        samples = np.random.randn(44100).astype(np.float32)
        feat = _mel_filterbank_energy(samples, 44100)
        assert feat.ndim == 1
        assert len(feat) == 40

    def test_different_frequencies_different_features(self):
        """異なる周波数のサイン波は異なる特徴量を持つこと。"""
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2).astype(np.float32)
        s1 = np.sin(2 * math.pi * 200.0 * t)
        s2 = np.sin(2 * math.pi * 4000.0 * t)
        f1 = _mel_filterbank_energy(s1, sr)
        f2 = _mel_filterbank_energy(s2, sr)
        # 低音と高音なのでコサイン類似度が低いはず
        sim = _cosine_similarity(f1, f2)
        assert sim < 0.9  # 完全に同じではない


class TestEvaluatePerceptual:

    def test_returns_scores_for_all_stems(self):
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.4),
            "drums":  _make_sine(80.0, amp=0.3),
        }
        scores = evaluate_perceptual(original, stems, 44100, use_wavlm=False)
        assert set(scores.keys()) == {"vocals", "drums"}

    def test_scores_in_valid_range(self):
        original = _make_sine(440.0)
        stems = {
            "vocals": _make_sine(440.0, amp=0.4),
            "drums":  _make_sine(80.0, amp=0.3),
            "bass":   _make_sine(60.0, amp=0.2),
        }
        scores = evaluate_perceptual(original, stems, 44100, use_wavlm=False)
        for name, s in scores.items():
            assert 0.0 <= s.ps_score <= 1.0, f"{name} PS out of range"
            assert 0.0 <= s.pm_score <= 1.0, f"{name} PM out of range"
            assert 0.0 <= s.combined <= 1.0, f"{name} combined out of range"

    def test_uses_fallback_method(self):
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        scores = evaluate_perceptual(original, stems, 44100, use_wavlm=False)
        assert scores["vocals"].method == "fallback"

    def test_perceptual_scores_dataclass(self):
        ps = PerceptualScores(stem="vocals", ps_score=0.8, pm_score=0.6)
        # combined = 調和平均
        expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        assert abs(ps.combined - expected) < 1e-5

    def test_combined_zero_when_both_zero(self):
        ps = PerceptualScores(stem="test", ps_score=0.0, pm_score=0.0)
        assert ps.combined == 0.0

    def test_single_stem_has_perfect_ps(self):
        """ステムが1つだけなら他ステムがないのでPS=1.0になること。"""
        original = _make_sine(440.0)
        stems = {"vocals": _make_sine(440.0, amp=0.5)}
        scores = evaluate_perceptual(original, stems, 44100, use_wavlm=False)
        assert scores["vocals"].ps_score == 1.0
