"""
test_engine_mock.py — MetaCogEngineのモックテスト (v2)
Demucsモデルをモック化してエンジンのループロジックだけを検証する。
"""
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest

from demucs_metacog.engine import EngineConfig, MetaCogEngine, IterationResult, EngineResult
from demucs_metacog.intent import SeparationIntent
from demucs_metacog.quality import QualityReport, StemQuality


def _make_sine(freq: float = 440.0, duration: float = 1.0, sr: int = 44100, amp: float = 0.5):
    t = torch.linspace(0, duration, int(sr * duration))
    wave = amp * torch.sin(2 * math.pi * freq * t)
    return wave.unsqueeze(0).repeat(2, 1)


def _make_mock_stems():
    return {
        "vocals": _make_sine(440.0, amp=0.4),
        "drums":  _make_sine(80.0,  amp=0.3),
        "bass":   _make_sine(60.0,  amp=0.2),
        "other":  _make_sine(880.0, amp=0.1),
    }


def _make_pass_report(stems):
    sq_map = {k: StemQuality(k, snr_db=15.0, energy_ratio=0.3, leakage_ratio=0.1, passed=True)
              for k in stems}
    return QualityReport(stems=sq_map, overall_passed=True, failed_stems=[])


def _make_fail_report(stems):
    sq_map = {k: StemQuality(k, snr_db=1.0, energy_ratio=0.3, leakage_ratio=0.6, passed=False,
                             reason="SNR low")
              for k in stems}
    return QualityReport(stems=sq_map, overall_passed=False, retry_needed=True,
                         failed_stems=list(stems.keys()))


# v2のシグネチャ: (self, waveform, sr, iteration, model_name, profile)
def _mock_separate(waveform, sr, iteration, model_name, profile):
    return _make_mock_stems()


class TestEngineResult:

    def test_n_iterations(self):
        stems = _make_mock_stems()
        report = _make_pass_report(stems)
        iter_result = IterationResult(iteration=0, stems=stems, report=report,
                                      elapsed_sec=1.0, model_used="htdemucs")
        result = EngineResult(
            final_stems=stems, sample_rate=44100,
            iterations=[iter_result], total_elapsed_sec=1.5, intent="default"
        )
        assert result.n_iterations == 1
        assert result.final_report.overall_passed is True

    def test_summary_contains_metadata(self):
        stems = _make_mock_stems()
        report = _make_pass_report(stems)
        iter_result = IterationResult(iteration=0, stems=stems, report=report,
                                      elapsed_sec=1.0, model_used="htdemucs")
        result = EngineResult(
            final_stems=stems, sample_rate=44100,
            iterations=[iter_result], total_elapsed_sec=1.5, intent="karaoke"
        )
        summary = result.summary()
        assert "MetaCog Engine Result" in summary
        assert "karaoke" in summary

    def test_summary_contains_intent(self):
        stems = _make_mock_stems()
        report = _make_pass_report(stems)
        iter_result = IterationResult(iteration=0, stems=stems, report=report,
                                      elapsed_sec=1.0, model_used="htdemucs")
        result = EngineResult(
            final_stems=stems, sample_rate=44100,
            iterations=[iter_result], total_elapsed_sec=1.5, intent="sample"
        )
        assert "sample" in result.summary()


class TestEngineConfig:

    def test_defaults(self):
        config = EngineConfig()
        assert config.model_name == "htdemucs"
        assert config.max_iterations == 3
        assert config.retry_strategy == "shifts"
        assert config.verbose is True
        assert config.intent == SeparationIntent.DEFAULT

    def test_custom_config_with_intent(self):
        config = EngineConfig(
            model_name="htdemucs_ft",
            max_iterations=2,
            retry_strategy="overlap",
            intent=SeparationIntent.KARAOKE,
            verbose=False,
        )
        assert config.model_name == "htdemucs_ft"
        assert config.intent == SeparationIntent.KARAOKE

    def test_intent_as_string(self):
        """intentを文字列で渡せること。"""
        config = EngineConfig(intent="remix")
        # engine.run()内でget_intent_profile()が文字列を受け取れる
        from demucs_metacog.intent import get_intent_profile
        profile = get_intent_profile(config.intent)
        assert profile.intent == SeparationIntent.REMIX


class TestEngineLoopLogic:

    def _make_engine(self, config, pass_fn):
        """テスト用エンジンを生成するヘルパー。"""
        engine = MetaCogEngine(config)
        engine._separate_with_strategy = _mock_separate
        mock_sep = MagicMock()
        mock_sep.sample_rate = 44100
        engine._separators["htdemucs"] = mock_sep
        engine._separators[config.model_name] = mock_sep
        return engine

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_stops_early_when_quality_passes(self, mock_eval):
        """品質チェックがパスしたら1イテレーションで終わること。"""
        # engine.pyはすべてキーワード引数で呼ぶ → **kwargs で受け取る
        mock_eval.side_effect = lambda **kw: _make_pass_report(kw["stems"])

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = self._make_engine(config, None)

        result = engine.run(_make_sine(), 44100)
        assert result.n_iterations == 1
        assert mock_eval.call_count == 1

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_runs_max_iterations_when_always_fails(self, mock_eval):
        """常に失敗するとmax_iterationsまで実行すること。"""
        mock_eval.side_effect = lambda **kw: _make_fail_report(kw["stems"])

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = self._make_engine(config, None)

        result = engine.run(_make_sine(), 44100)
        assert result.n_iterations == 3
        assert mock_eval.call_count == 3

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_passes_on_second_iteration(self, mock_eval):
        """2回目でパスする場合。"""
        call_count = {"n": 0}

        def side_effect(**kw):
            i = call_count["n"]
            call_count["n"] += 1
            return _make_pass_report(kw["stems"]) if i == 1 else _make_fail_report(kw["stems"])

        mock_eval.side_effect = side_effect

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = self._make_engine(config, None)

        result = engine.run(_make_sine(), 44100)
        assert result.n_iterations == 2
        assert result.final_report.overall_passed is True

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_result_has_all_stems(self, mock_eval):
        mock_eval.side_effect = lambda **kw: _make_pass_report(kw["stems"])

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = self._make_engine(config, None)

        result = engine.run(_make_sine(), 44100)
        assert set(result.final_stems.keys()) == {"vocals", "drums", "bass", "other"}

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_intent_applied_to_iterations(self, mock_eval):
        """intent情報がイテレーション結果に記録されること。"""
        mock_eval.side_effect = lambda **kw: _make_pass_report(kw["stems"])

        config = EngineConfig(max_iterations=3, verbose=False, intent=SeparationIntent.KARAOKE)
        engine = self._make_engine(config, None)

        result = engine.run(_make_sine(), 44100)
        assert result.intent == "karaoke"
        assert result.iterations[0].intent_applied == "karaoke"
