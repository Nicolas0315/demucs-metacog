"""
test_engine_mock.py — MetaCogEngineのモックテスト
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


class TestEngineResult:
    """EngineResultの基本プロパティテスト。"""

    def test_n_iterations(self):
        waveform = _make_sine()
        stems = _make_mock_stems()

        # QualityReportを手動作成
        sq = StemQuality("vocals", snr_db=10.0, energy_ratio=0.5, crosstalk_db=-15.0, passed=True)
        report = QualityReport(stems={"vocals": sq}, overall_passed=True)

        iter_result = IterationResult(iteration=0, stems=stems, report=report, elapsed_sec=1.0)
        result = EngineResult(
            final_stems=stems,
            sample_rate=44100,
            iterations=[iter_result],
            total_elapsed_sec=1.5,
        )

        assert result.n_iterations == 1
        assert result.final_report.overall_passed is True

    def test_summary_contains_metadata(self):
        waveform = _make_sine()
        stems = _make_mock_stems()
        sq = StemQuality("vocals", snr_db=10.0, energy_ratio=0.5, crosstalk_db=-15.0, passed=True)
        report = QualityReport(stems={"vocals": sq}, overall_passed=True)
        iter_result = IterationResult(iteration=0, stems=stems, report=report, elapsed_sec=1.0)
        result = EngineResult(
            final_stems=stems,
            sample_rate=44100,
            iterations=[iter_result],
            total_elapsed_sec=1.5,
        )
        summary = result.summary()
        assert "MetaCog Engine Result" in summary
        assert "1" in summary  # n_iterations


class TestEngineConfig:
    """EngineConfigのデフォルト値テスト。"""

    def test_defaults(self):
        config = EngineConfig()
        assert config.model_name == "htdemucs"
        assert config.max_iterations == 3
        assert config.retry_strategy == "shifts"
        assert config.verbose is True

    def test_custom_config(self):
        config = EngineConfig(
            model_name="htdemucs_ft",
            max_iterations=2,
            retry_strategy="overlap",
            verbose=False,
        )
        assert config.model_name == "htdemucs_ft"
        assert config.max_iterations == 2
        assert config.retry_strategy == "overlap"


class TestEngineLoopLogic:
    """エンジンのループロジックをモックでテスト。"""

    def _make_engine_with_mock(self, pass_on_iteration: int = 0, max_iter: int = 3):
        """
        指定したイテレーションで品質チェックをパスするモックエンジンを作る。
        実際のDemucsモデルはロードしない。
        """
        config = EngineConfig(max_iterations=max_iter, verbose=False)
        engine = MetaCogEngine(config)

        call_count = {"n": 0}

        def mock_separate(waveform, sr, iteration):
            return _make_mock_stems()

        def mock_evaluate(original, stems, thresholds=None):
            i = call_count["n"]
            call_count["n"] += 1
            passed = (i >= pass_on_iteration)
            sq_map = {
                k: StemQuality(k, snr_db=10.0 if passed else 2.0,
                               energy_ratio=0.3, crosstalk_db=-14.0,
                               passed=passed)
                for k in stems
            }
            return QualityReport(
                stems=sq_map,
                overall_passed=passed,
                retry_needed=not passed,
            )

        engine._separate_with_strategy = mock_separate

        import demucs_metacog.engine as eng_mod
        engine._evaluate = mock_evaluate  # 直接差し替えは難しいのでpatchを使う
        return engine, mock_evaluate

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_stops_early_when_quality_passes(self, mock_eval):
        """品質チェックがパスしたら早期終了すること。"""
        # iter=0でパス → 1イテレーションで終わるべき
        def side_effect(original, stems, thresholds=None):
            sq_map = {k: StemQuality(k, 15.0, 0.3, -16.0, True) for k in stems}
            return QualityReport(stems=sq_map, overall_passed=True)

        mock_eval.side_effect = side_effect

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = MetaCogEngine(config)

        # _separate_with_strategy をモック化
        engine._separate_with_strategy = lambda w, sr, iteration: _make_mock_stems()

        waveform = _make_sine()
        result = engine.run(waveform, 44100)

        assert result.n_iterations == 1
        assert mock_eval.call_count == 1

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_runs_max_iterations_when_always_fails(self, mock_eval):
        """常に失敗する場合はmax_iterationsまで実行すること。"""
        def side_effect(original, stems, thresholds=None):
            sq_map = {k: StemQuality(k, 1.0, 0.3, -2.0, False, "SNR low") for k in stems}
            return QualityReport(stems=sq_map, overall_passed=False, retry_needed=True)

        mock_eval.side_effect = side_effect

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = MetaCogEngine(config)
        engine._separate_with_strategy = lambda w, sr, iteration: _make_mock_stems()

        waveform = _make_sine()
        result = engine.run(waveform, 44100)

        assert result.n_iterations == 3
        assert mock_eval.call_count == 3

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_passes_on_second_iteration(self, mock_eval):
        """2回目のイテレーションでパスする場合。"""
        call_count = {"n": 0}

        def side_effect(original, stems, thresholds=None):
            i = call_count["n"]
            call_count["n"] += 1
            passed = (i == 1)  # 2回目(index=1)でパス
            sq_map = {k: StemQuality(k, 15.0 if passed else 2.0, 0.3, -16.0, passed) for k in stems}
            return QualityReport(stems=sq_map, overall_passed=passed, retry_needed=not passed)

        mock_eval.side_effect = side_effect

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = MetaCogEngine(config)
        engine._separate_with_strategy = lambda w, sr, iteration: _make_mock_stems()

        waveform = _make_sine()
        result = engine.run(waveform, 44100)

        assert result.n_iterations == 2
        assert result.final_report.overall_passed is True

    @patch("demucs_metacog.engine.evaluate_stems")
    def test_result_has_all_stems(self, mock_eval):
        """最終結果に全ステムが含まれること。"""
        def side_effect(original, stems, thresholds=None):
            sq_map = {k: StemQuality(k, 15.0, 0.3, -16.0, True) for k in stems}
            return QualityReport(stems=sq_map, overall_passed=True)

        mock_eval.side_effect = side_effect

        config = EngineConfig(max_iterations=3, verbose=False)
        engine = MetaCogEngine(config)
        engine._separate_with_strategy = lambda w, sr, iteration: _make_mock_stems()

        waveform = _make_sine()
        result = engine.run(waveform, 44100)

        assert set(result.final_stems.keys()) == {"vocals", "drums", "bass", "other"}
