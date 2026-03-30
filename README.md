# demucs-metacog

**Demucs + Temporal Metacognition Engine**

Demucs (HT Demucs v4) にメタ認知ループを追加した音声ステム分離エンジン。
「分離 → 品質監査 → 再分離」のループで、1発の推論で品質が足りない場合に自動的に再試行する。

## アーキテクチャ

```
入力音声
  │
  ▼
┌────────────────────────────────┐
│  MetaCogEngine.run()           │
│                                │
│  for i in range(max_iter):     │
│    ┌─────────────────────────┐ │
│    │  DemucsBase.separate()  │ │  ← 直感パス（時間領域+周波数領域のデュアルUNet）
│    └──────────┬──────────────┘ │
│               │                │
│    ┌──────────▼──────────────┐ │
│    │  evaluate_stems()       │ │  ← 監査層（SNR/クロストーク/エネルギー比）
│    └──────────┬──────────────┘ │
│               │                │
│    ┌──────────▼──────────────┐ │
│    │  passed?                │ │
│    │  YES → break            │ │  ← 早期終了
│    │  NO  → adjust params    │ │  ← shifts/overlap調整して再試行
│    └─────────────────────────┘ │
└────────────────────────────────┘
  │
  ▼
出力ステム (vocals / drums / bass / other)
```

### 品質評価指標

| 指標 | 意味 | デフォルト閾値 |
|------|------|-------------|
| SNR (dB) | 信号対雑音比 | ≥ 8.0 dB |
| Crosstalk (dB) | 他ステムとの干渉 | ≤ -12.0 dB |
| Energy ratio | ステムのエネルギー占有率 | ≥ 1% |

### 再分離ストラテジー

- `shifts` (デフォルト): イテレーション毎にランダムシフト数を増加 → 位相揺らぎの平均化でアーティファクト低減
- `overlap`: セグメント間のオーバーラップ率を増加 → セグメント境界ノイズを低減

## 使い方

```bash
# インストール（uv）
uv sync

# 分離実行
uv run separate.py input.mp3
uv run separate.py input.wav --output-dir out/ --max-iter 3 --model htdemucs_ft
uv run separate.py input.wav --strategy overlap --verbose

# テスト
uv run pytest tests/ -v
```

## ファイル構成

```
demucs-metacog/
├── src/demucs_metacog/
│   ├── __init__.py
│   ├── separator.py   # Demucs推論ラッパー
│   ├── quality.py     # 品質評価・監査層
│   ├── engine.py      # メタ認知ループオーケストレーター
│   └── io.py          # 音声入出力ユーティリティ
├── tests/
│   ├── test_quality.py      # 品質評価ユニットテスト（合成信号）
│   └── test_engine_mock.py  # エンジンループテスト（モック）
├── separate.py   # CLI エントリーポイント
└── pyproject.toml
```

## 設計哲学

このプロジェクトは [Katala Engine](../katala/) の**メタ認知ループ設計**を音声分離に移植したもの。

- **直感パス**: Demucsによる高速推論（Transformer + UNet）
- **監査層**: SNR/クロストーク指標による客観的品質判定（Same-Model Biasを避けるために独立した評価関数）
- **修正ループ**: 失敗なら推論パラメータを変えて再実行（ARC-AGIメタ認知エンジンの「自己修正」と同じ構造）

## 今後の拡張ベクトル

1. **Intent-Aware分離**: 用途（カラオケ/サンプル抽出/リミックス）に応じて分離戦略を切り替える
2. **Semantic Token Integration**: コード進行・楽器役割の事前推定を補助信号に使う
3. **Streaming-First**: リアルタイム分離（Discord Voice対応）
4. **Distillation Loss**: WavLM/Music2Vecの知識を訓練損失に組み込む
