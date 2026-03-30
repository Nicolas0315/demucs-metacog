"""
io.py — 音声ファイルの入出力ユーティリティ。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """
    音声ファイルを読み込む。
    Returns: (waveform: (C, T), sample_rate: int)
    """
    waveform, sr = torchaudio.load(str(path))
    # ステレオに揃える
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform, sr


def save_stems(
    stems: dict[str, torch.Tensor],
    output_dir: str | Path,
    sample_rate: int,
    prefix: str = "",
) -> dict[str, Path]:
    """
    各ステムをWAVファイルとして保存する。
    Returns: stem_name -> saved Path のdict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for name, waveform in stems.items():
        fname = f"{prefix}{name}.wav" if prefix else f"{name}.wav"
        out_path = output_dir / fname
        torchaudio.save(str(out_path), waveform, sample_rate)
        saved[name] = out_path

    return saved
