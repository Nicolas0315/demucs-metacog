"""
perceptual.py — Perceptual quality estimation (MAPSS近似)

MAPSSの完全実装はSSLモデル（Wav2Vec2/MERT）+ 拡散マップ + マハラノビス距離を必要とするが、
ここでは「SSLモデルなし」で動作するフォールバック実装と、
「WavLM利用可能時」の本格実装を両方提供する。

MAPSS論文 (Eskimez et al. 2025) の2指標を近似:
  - PS (Perceptual Separation) : 他ステムからの漏れ込みを知覚的に測定
  - PM (Perceptual Match)      : 分離出力と元ソースの一致度を測定

フォールバック（SSLなし）:
  - PS ≈ スペクトル重心の差分ベースのクロストーク推定
  - PM ≈ MFCC-like係数（mel-filterbank energy）のコサイン類似度

SSL利用可能時（transformers インストール済み）:
  - WavLM-base の hidden state 平均をステム埋め込みとして使用
  - PS/PM をコサイン距離で計算

Huang Limitへの対処:
  このモジュールは「外部検証器」として機能する。
  Demucsモデル自身とは独立した評価シグナルを提供することで、
  同一モデルの内部バイアスから切り離した品質判断を実現する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────────────────────────

@dataclass
class PerceptualScores:
    """MAPSS近似スコア。"""
    stem: str
    ps_score: float    # Perceptual Separation [0,1] — 高いほど他から分離できている
    pm_score: float    # Perceptual Match [0,1]      — 高いほど元信号に一致
    method: str = "fallback"  # "fallback" | "wavlm"

    @property
    def combined(self) -> float:
        """PS と PM の調和平均。両方が高い場合にのみ高くなる。"""
        if self.ps_score + self.pm_score < 1e-6:
            return 0.0
        return 2 * self.ps_score * self.pm_score / (self.ps_score + self.pm_score)


# ──────────────────────────────────────────────────────────────
# フォールバック実装（SSLモデルなし）
# ──────────────────────────────────────────────────────────────

def _mel_filterbank_energy(
    samples: np.ndarray,
    sample_rate: int,
    n_mels: int = 40,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """
    簡易Mel Filterbank Energyを計算する（librosaなし版）。
    MFCC-likeな特徴量として使用。
    """
    # Short-time Fourier Transform
    n_frames = max((len(samples) - n_fft) // hop_length + 1, 1)
    n_frames = min(n_frames, 500)  # 計算コスト制限

    power_spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = samples[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame = frame * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        power_spec[:, i] = spectrum

    # 簡易Melフィルタバンク（線形近似）
    freq_bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    mel_low = 0.0
    mel_high = 2595.0 * np.log10(1 + sample_rate / 2 / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    mel_energy = np.zeros((n_mels, n_frames), dtype=np.float32)
    for m in range(n_mels):
        lower = hz_points[m]
        center = hz_points[m + 1]
        upper = hz_points[m + 2]

        for k, freq in enumerate(freq_bins):
            if lower <= freq <= center:
                w = (freq - lower) / (center - lower + 1e-10)
            elif center < freq <= upper:
                w = (upper - freq) / (upper - center + 1e-10)
            else:
                w = 0.0
            mel_energy[m] += w * power_spec[k]

    # 平均を特徴ベクトルとして返す
    return mel_energy.mean(axis=1)  # shape: (n_mels,)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度 [-1, 1] を [0, 1] に正規化して返す。"""
    norm_a = np.linalg.norm(a) + 1e-10
    norm_b = np.linalg.norm(b) + 1e-10
    cos = float(np.dot(a, b) / (norm_a * norm_b))
    return (cos + 1.0) / 2.0  # [-1,1] → [0,1]


def _compute_fallback_scores(
    stem_arr: np.ndarray,
    original_arr: np.ndarray,
    other_stems: list[np.ndarray],
    sample_rate: int,
) -> tuple[float, float]:
    """
    SSLなしのフォールバックPS/PM計算。

    PS: このステムの特徴量が他ステムの混合とどれだけ異なるか
        → 高いほど分離できている
    PM: このステムと元信号の特徴量がどれだけ一致しているか
        → 高いほど原音を保持している
    """
    # ── PM: ステム vs 元信号のMelスペクトル一致度 ────────
    feat_stem = _mel_filterbank_energy(stem_arr, sample_rate)
    feat_orig = _mel_filterbank_energy(original_arr, sample_rate)
    pm = _cosine_similarity(feat_stem, feat_orig)

    # ── PS: ステムが他ステムの混合からどれだけ独立しているか ──
    if not other_stems:
        ps = 1.0
    else:
        # 他ステムの平均特徴量
        other_feats = [_mel_filterbank_energy(o, sample_rate) for o in other_stems]
        other_mean = np.mean(other_feats, axis=0)
        # 類似度が低いほど分離できている → 1 - similarity
        ps = 1.0 - _cosine_similarity(feat_stem, other_mean)

    return float(np.clip(ps, 0.0, 1.0)), float(np.clip(pm, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────
# WavLM実装（オプション）
# ──────────────────────────────────────────────────────────────

_WAVLM_MODEL = None
_WAVLM_PROCESSOR = None
_WAVLM_AVAILABLE: Optional[bool] = None

def _check_wavlm() -> bool:
    global _WAVLM_AVAILABLE
    if _WAVLM_AVAILABLE is not None:
        return _WAVLM_AVAILABLE
    try:
        from transformers import WavLMModel, AutoProcessor  # noqa: F401
        _WAVLM_AVAILABLE = True
    except ImportError:
        _WAVLM_AVAILABLE = False
    return _WAVLM_AVAILABLE


def _get_wavlm_embedding(samples: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """WavLMのhidden states平均を埋め込みとして返す。"""
    global _WAVLM_MODEL, _WAVLM_PROCESSOR
    if not _check_wavlm():
        return None

    try:
        from transformers import WavLMModel, AutoProcessor

        if _WAVLM_MODEL is None:
            _WAVLM_PROCESSOR = AutoProcessor.from_pretrained("microsoft/wavlm-base")
            _WAVLM_MODEL = WavLMModel.from_pretrained("microsoft/wavlm-base")
            _WAVLM_MODEL.eval()

        # 最大10秒に切り詰めてメモリを節約
        max_samples = sample_rate * 10
        samples_clipped = samples[:max_samples]

        # 16kHzにリサンプル（WavLMの要求）
        if sample_rate != 16000:
            from scipy.signal import resample
            target_len = int(len(samples_clipped) * 16000 / sample_rate)
            samples_clipped = resample(samples_clipped, target_len).astype(np.float32)

        inputs = _WAVLM_PROCESSOR(
            samples_clipped, sampling_rate=16000, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = _WAVLM_MODEL(**inputs)

        # last_hidden_state の平均を特徴量として使用
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding

    except Exception:
        return None


def _compute_wavlm_scores(
    stem_arr: np.ndarray,
    original_arr: np.ndarray,
    other_stems: list[np.ndarray],
    sample_rate: int,
) -> Optional[tuple[float, float]]:
    """WavLM埋め込みを使ったPS/PM計算。"""
    emb_stem = _get_wavlm_embedding(stem_arr, sample_rate)
    if emb_stem is None:
        return None

    emb_orig = _get_wavlm_embedding(original_arr, sample_rate)
    if emb_orig is None:
        return None

    pm = _cosine_similarity(emb_stem, emb_orig)

    if not other_stems:
        ps = 1.0
    else:
        other_embs = [_get_wavlm_embedding(o, sample_rate) for o in other_stems]
        other_embs = [e for e in other_embs if e is not None]
        if not other_embs:
            ps = 1.0
        else:
            other_mean = np.mean(other_embs, axis=0)
            ps = 1.0 - _cosine_similarity(emb_stem, other_mean)

    return float(np.clip(ps, 0.0, 1.0)), float(np.clip(pm, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────
# パブリックAPI
# ──────────────────────────────────────────────────────────────

def evaluate_perceptual(
    original: torch.Tensor,
    stems: dict[str, torch.Tensor],
    sample_rate: int,
    use_wavlm: bool = True,
) -> dict[str, PerceptualScores]:
    """
    全ステムのPS/PMスコアを計算する。

    Args:
        original: (channels, samples) — ミックス信号
        stems:    ステム名 → (channels, samples) tensor
        sample_rate: サンプルレート
        use_wavlm: WavLMが使えるなら使う（Falseでフォールバック強制）

    Returns:
        ステム名 → PerceptualScores
    """
    orig_np = original.mean(0).detach().cpu().float().numpy()
    stems_np = {k: v.mean(0).detach().cpu().float().numpy() for k, v in stems.items()}

    results: dict[str, PerceptualScores] = {}
    method_used = "fallback"

    for name, arr in stems_np.items():
        others = [v for k, v in stems_np.items() if k != name]

        scores = None
        if use_wavlm and _check_wavlm():
            scores = _compute_wavlm_scores(arr, orig_np, others, sample_rate)
            if scores is not None:
                method_used = "wavlm"

        if scores is None:
            scores = _compute_fallback_scores(arr, orig_np, others, sample_rate)
            method_used = "fallback"

        ps, pm = scores
        results[name] = PerceptualScores(
            stem=name, ps_score=round(ps, 3), pm_score=round(pm, 3),
            method=method_used,
        )

    return results
