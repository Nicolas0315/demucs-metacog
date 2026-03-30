"""
vad.py — Voice Activity Detection for stem quality audit
Katala audio_processing.py の VoiceActivityDetector を音声分離用に移植・拡張。

用途:
  - vocalsステムに本当に声が入っているか確認（無声楽曲の早期検出）
  - drumsステムにトランジェントが存在するか確認
  - bassステムに低域エネルギーが存在するか確認
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# 25msフレーム / 10msホップ（Katalaと同じ設定）
VAD_ENERGY_THRESHOLD = 0.01
VAD_FRAME_MS = 25
VAD_HOP_MS = 10
MAX_FRAMES = 3000  # ~30秒相当


@dataclass
class VoiceActivity:
    has_activity: bool = False
    activity_ratio: float = 0.0     # アクティブフレーム / 全フレーム
    speech_duration_sec: float = 0.0
    silence_duration_sec: float = 0.0
    n_segments: int = 0
    avg_segment_duration_sec: float = 0.0
    stem: str = ""


def _to_mono_numpy(tensor: torch.Tensor) -> np.ndarray:
    """(channels, samples) tensor → mono float32 numpy array."""
    mono = tensor.mean(0).detach().cpu().float().numpy()
    return mono


def detect_voice_activity(
    stem: torch.Tensor,
    sample_rate: int,
    stem_name: str = "",
    energy_threshold: float = VAD_ENERGY_THRESHOLD,
) -> VoiceActivity:
    """
    エネルギーベースVADを実行する。

    Args:
        stem: (channels, samples) float tensor
        sample_rate: サンプルレート
        stem_name: ステム名（ログ用）
        energy_threshold: 音声判定の閾値RMS

    Returns:
        VoiceActivity
    """
    samples = _to_mono_numpy(stem)
    frame_size = int(VAD_FRAME_MS * sample_rate / 1000)
    hop_size   = int(VAD_HOP_MS   * sample_rate / 1000)

    if frame_size <= 0 or len(samples) < frame_size:
        return VoiceActivity(stem=stem_name)

    n_frames = min((len(samples) - frame_size) // hop_size + 1, MAX_FRAMES)

    active_frames = 0
    silent_frames = 0
    in_activity = False
    segments = 0

    for i in range(n_frames):
        start = i * hop_size
        frame = samples[start:start + frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))

        if rms > energy_threshold:
            active_frames += 1
            if not in_activity:
                in_activity = True
                segments += 1
        else:
            silent_frames += 1
            in_activity = False

    frame_dur = hop_size / sample_rate
    activity_ratio = active_frames / max(n_frames, 1)

    return VoiceActivity(
        has_activity=active_frames > 0,
        activity_ratio=round(activity_ratio, 3),
        speech_duration_sec=round(active_frames * frame_dur, 2),
        silence_duration_sec=round(silent_frames * frame_dur, 2),
        n_segments=segments,
        avg_segment_duration_sec=round(
            (active_frames * frame_dur) / max(segments, 1), 2
        ),
        stem=stem_name,
    )


# ──────────────────────────────────────────────────────────────
# ステム別の期待アクティビティ閾値
# ──────────────────────────────────────────────────────────────

# activity_ratio がこの値を下回ると「このステムは空に近い」と判断
STEM_ACTIVITY_FLOOR = {
    "vocals": 0.10,   # 曲の10%以上でボーカルが検出されないと警告
    "drums":  0.05,   # ドラムは比較的まばらなのでゆるく
    "bass":   0.20,   # ベースは持続音なのでここは高め
    "other":  0.05,
}


def audit_stems_vad(
    stems: dict[str, torch.Tensor],
    sample_rate: int,
) -> dict[str, VoiceActivity]:
    """全ステムにVADを実行してステム名→VoiceActivityのdictを返す。"""
    return {
        name: detect_voice_activity(stem, sample_rate, stem_name=name)
        for name, stem in stems.items()
    }
