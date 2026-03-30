"""
intent.py — Intent-Aware 分離モード定義

「なぜ分離するか」= 用途によって最適な品質基準とエンジン挙動が変わる。
Demucsは用途を知らないが、このレイヤーが知っている。

モード設計の哲学:
- karaoke    : ボーカル除去が目的。vocals の クリアさ最優先、phase整合は二の次
- sample     : 特定ステムをDAWに持ち込む。全ステムのクロストークを厳しく評価
- remix      : ステム間のフェーザー/グルーを残したい。energy_ratioをゆるくする
- analysis   : 音楽構造解析目的。バランス重視、全ステム均等評価
- mastering  : マスタリング用途。SNRを最も厳しく評価
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SeparationIntent(str, Enum):
    KARAOKE   = "karaoke"    # ボーカル除去（カラオケ、歌練習）
    SAMPLE    = "sample"     # サンプル抽出（DAW素材）
    REMIX     = "remix"      # リミックス（ステムを再構成）
    ANALYSIS  = "analysis"   # 楽曲分析（BPM/コード/構成解析）
    MASTERING = "mastering"  # マスタリング参照
    DEFAULT   = "default"    # 汎用（デフォルト）


@dataclass
class IntentProfile:
    """
    用途に応じた分離戦略プロファイル。
    EngineConfigに注入されてエンジンの挙動を変える。
    """
    intent: SeparationIntent

    # ── 品質閾値 ──────────────────────────────────────────────
    # 各ステムごとのSNR閾値（Noneなら全ステム共通値を使う）
    stem_snr_thresholds: dict[str, float] = field(default_factory=dict)
    global_snr_threshold: float = 8.0

    # エネルギー閾値（低いほど寛容）
    min_energy_ratio: float = 0.01

    # ── 分離戦略 ──────────────────────────────────────────────
    # 評価対象ステム（Noneなら全ステム評価）
    priority_stems: Optional[list[str]] = None

    # 再試行時に使うDemucsモデル（Noneなら同じモデルを使う）
    retry_model: Optional[str] = None

    # Demucsのoverlap設定
    overlap: float = 0.25

    # shifts設定
    shifts: int = 1

    # ── メタデータ ────────────────────────────────────────────
    description: str = ""

    def to_quality_thresholds(self) -> dict:
        """evaluate_stems()が受け取る形式に変換。"""
        return {
            "snr_db": self.global_snr_threshold,
            "min_energy_ratio": self.min_energy_ratio,
            # leakageはIntentによって変えない（per-stem SNRで制御する）
            "max_leakage_ratio": 0.45,
        }

    def get_stem_snr(self, stem: str) -> float:
        """ステム固有のSNR閾値を返す（未設定ならglobal値）。"""
        return self.stem_snr_thresholds.get(stem, self.global_snr_threshold)


# ──────────────────────────────────────────────────────────────────────────────
# プリセットプロファイル
# ──────────────────────────────────────────────────────────────────────────────

def get_intent_profile(intent: SeparationIntent | str) -> IntentProfile:
    """用途名からIntentProfileを返す。"""
    if isinstance(intent, str):
        intent = SeparationIntent(intent)

    if intent == SeparationIntent.KARAOKE:
        return IntentProfile(
            intent=intent,
            description="ボーカル除去。vocalsのSNRを最優先。drums/bass/otherはゆるく評価。",
            # vocalsだけ厳しく評価、他は緩い
            stem_snr_thresholds={
                "vocals": 12.0,   # ボーカルは高SNR必須
                "drums":   4.0,
                "bass":    4.0,
                "other":   4.0,
            },
            global_snr_threshold=4.0,
            min_energy_ratio=0.005,  # ボーカルなし曲も許容
            priority_stems=["vocals"],
            overlap=0.35,   # セグメント境界のボーカル漏れを減らす
            shifts=2,
        )

    elif intent == SeparationIntent.SAMPLE:
        return IntentProfile(
            intent=intent,
            description="DAWサンプル抽出。全ステムを均等に厳しく評価。",
            stem_snr_thresholds={
                "vocals": 10.0,
                "drums":  10.0,
                "bass":   10.0,
                "other":  10.0,
            },
            global_snr_threshold=10.0,
            min_energy_ratio=0.02,
            priority_stems=None,  # 全ステム評価
            overlap=0.4,
            shifts=2,
        )

    elif intent == SeparationIntent.REMIX:
        return IntentProfile(
            intent=intent,
            description="リミックス。ステム間の自然なグルーを保つ。エネルギー基準を緩める。",
            stem_snr_thresholds={
                "vocals": 8.0,
                "drums":  6.0,
                "bass":   6.0,
                "other":  5.0,
            },
            global_snr_threshold=6.0,
            min_energy_ratio=0.005,  # 薄いパッドやアンビエンスも許容
            priority_stems=["vocals", "drums"],
            overlap=0.25,   # 過度な平滑化を避けてグルーを保つ
            shifts=1,
        )

    elif intent == SeparationIntent.ANALYSIS:
        return IntentProfile(
            intent=intent,
            description="楽曲分析。全ステムをバランスよく評価。",
            global_snr_threshold=7.0,
            min_energy_ratio=0.01,
            priority_stems=None,
            overlap=0.3,
            shifts=1,
        )

    elif intent == SeparationIntent.MASTERING:
        return IntentProfile(
            intent=intent,
            description="マスタリング参照。最高精度。SNR基準が最も厳しい。",
            stem_snr_thresholds={
                "vocals": 14.0,
                "drums":  12.0,
                "bass":   12.0,
                "other":  10.0,
            },
            global_snr_threshold=12.0,
            min_energy_ratio=0.02,
            priority_stems=None,
            retry_model="htdemucs_ft",  # 再試行時は高精度モデルに切り替え
            overlap=0.45,
            shifts=3,
        )

    else:  # DEFAULT
        return IntentProfile(
            intent=SeparationIntent.DEFAULT,
            description="汎用デフォルト。",
            global_snr_threshold=8.0,
            min_energy_ratio=0.01,
            overlap=0.25,
            shifts=1,
        )
