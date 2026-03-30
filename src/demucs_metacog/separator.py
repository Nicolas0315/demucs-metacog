"""
separator.py — Demucs推論ラッパー
Demucsのモデルロードと分離実行を担当。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)

# デフォルトモデル: Hybrid Transformer Demucs v4
DEFAULT_MODEL = "htdemucs"
STEMS = ("drums", "bass", "other", "vocals")


class DemucsBase:
    """Demucsモデルのシンプルなラッパー。"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from demucs.pretrained import get_model
        logger.info(f"Loading Demucs model: {self.model_name} on {self.device}")
        self._model = get_model(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def separate(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, torch.Tensor]:
        """
        Args:
            waveform: (channels, samples) float tensor
            sample_rate: input sample rate

        Returns:
            dict of stem_name -> (channels, samples) tensor at model's native sr
        """
        self._load()
        from demucs.apply import apply_model

        # Demucsは44100Hz固定
        target_sr = self._model.samplerate
        if sample_rate != target_sr:
            logger.info(f"Resampling {sample_rate}Hz → {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        # バッチ次元追加: (1, channels, samples)
        wav_batch = waveform.unsqueeze(0).to(self.device)

        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav_batch,
                device=self.device,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=False,
            )

        # sources shape: (1, n_stems, channels, samples)
        stems_out: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self._model.sources):
            stems_out[name] = sources[0, i].cpu()

        return stems_out

    @property
    def sample_rate(self) -> int:
        self._load()
        return self._model.samplerate
