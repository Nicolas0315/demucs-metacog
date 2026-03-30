"""
test_intent.py — Intent-Aware プロファイルのテスト
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from demucs_metacog.intent import SeparationIntent, IntentProfile, get_intent_profile


class TestSeparationIntent:

    def test_all_intents_have_profiles(self):
        """全intentにプロファイルが定義されていること。"""
        for intent in SeparationIntent:
            profile = get_intent_profile(intent)
            assert isinstance(profile, IntentProfile)

    def test_string_input_works(self):
        """文字列でもプロファイルが取得できること。"""
        profile = get_intent_profile("karaoke")
        assert profile.intent == SeparationIntent.KARAOKE

    def test_invalid_string_raises(self):
        """無効な文字列はValueErrorになること。"""
        with pytest.raises(ValueError):
            get_intent_profile("invalid_mode")


class TestKaraokeProfile:

    def setup_method(self):
        self.profile = get_intent_profile(SeparationIntent.KARAOKE)

    def test_vocals_snr_higher_than_others(self):
        """karaokeではvocalsのSNR閾値が他より高いこと。"""
        vocals_thr = self.profile.stem_snr_thresholds.get("vocals", self.profile.global_snr_threshold)
        drums_thr  = self.profile.stem_snr_thresholds.get("drums",  self.profile.global_snr_threshold)
        assert vocals_thr > drums_thr

    def test_priority_stems_includes_vocals(self):
        assert "vocals" in (self.profile.priority_stems or [])

    def test_overlap_higher_than_default(self):
        default = get_intent_profile(SeparationIntent.DEFAULT)
        assert self.profile.overlap >= default.overlap

    def test_description_not_empty(self):
        assert len(self.profile.description) > 0


class TestSampleProfile:

    def setup_method(self):
        self.profile = get_intent_profile(SeparationIntent.SAMPLE)

    def test_all_stems_have_equal_or_high_snr(self):
        """sampleでは全ステムのSNR閾値が高め（均等）であること。"""
        thrs = [
            self.profile.stem_snr_thresholds.get(s, self.profile.global_snr_threshold)
            for s in ["vocals", "drums", "bass", "other"]
        ]
        # 全て同じ値のはず
        assert len(set(thrs)) == 1

    def test_priority_stems_is_none(self):
        """sampleでは全ステムを評価するのでpriority_stemsがNone。"""
        assert self.profile.priority_stems is None


class TestMasteringProfile:

    def setup_method(self):
        self.profile = get_intent_profile(SeparationIntent.MASTERING)

    def test_highest_snr_thresholds(self):
        """masteringは最も厳しいSNR閾値を持つこと。"""
        karaoke  = get_intent_profile(SeparationIntent.KARAOKE)
        sample   = get_intent_profile(SeparationIntent.SAMPLE)
        mastering_global = self.profile.global_snr_threshold
        assert mastering_global >= karaoke.global_snr_threshold
        assert mastering_global >= sample.global_snr_threshold

    def test_has_retry_model(self):
        """masteringは再試行モデルが設定されていること。"""
        assert self.profile.retry_model is not None
        assert "ft" in (self.profile.retry_model or "")

    def test_shifts_highest(self):
        """masteringのshiftsが最も多いこと。"""
        default = get_intent_profile(SeparationIntent.DEFAULT)
        assert self.profile.shifts >= default.shifts


class TestIntentThresholds:

    def test_to_quality_thresholds_keys(self):
        """to_quality_thresholds()が必要なキーを持つこと。"""
        profile = get_intent_profile(SeparationIntent.KARAOKE)
        thr = profile.to_quality_thresholds()
        assert "snr_db" in thr
        assert "min_energy_ratio" in thr
        assert "max_leakage_ratio" in thr

    def test_get_stem_snr_fallback(self):
        """未定義ステムはglobal値にフォールバックすること。"""
        profile = get_intent_profile(SeparationIntent.DEFAULT)
        # defaultはstem_snr_thresholdsが空
        assert profile.get_stem_snr("unknown_stem") == profile.global_snr_threshold

    def test_get_stem_snr_specific(self):
        """karaoke: vocals固有のSNR閾値が返ること。"""
        profile = get_intent_profile(SeparationIntent.KARAOKE)
        vocals_thr = profile.get_stem_snr("vocals")
        assert vocals_thr == profile.stem_snr_thresholds["vocals"]
