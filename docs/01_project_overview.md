# プロジェクト概要

## SoulX-Singer とは

SoulX-Singerは、Soul AI Labが開発した**ゼロショット歌声合成（Singing Voice Synthesis, SVS）モデル**。
短いプロンプト音声を与えるだけで、未知の歌手の音色を模倣した歌声をファインチューニングなしで生成できる。

- **リリース日**: 2026年2月6日
- **ライセンス**: Apache 2.0
- **推奨Python**: 3.10
- **学習データ規模**: 42,000時間以上（中国語・英語・広東語）

## 主な機能

| 機能 | 説明 |
|------|------|
| ゼロショット歌声合成 | 未知の歌手でもプロンプト音声だけで音色をクローン |
| メロディ制御 | F0（基本周波数）コンターで音高を指定 |
| スコア制御 | MIDIノート（音高＋デュレーション）で音高を指定 |
| 歌声編集 | 歌詞を変えても自然なプロソディを維持 |
| クロスリンガル合成 | 音色と言語を分離し、異なる言語での歌唱が可能 |

## 対応言語

| 言語 | 前処理フラグ | ASRモデル | G2Pモデル | 音素プレフィックス |
|------|-------------|-----------|-----------|-------------------|
| 中国語（普通話） | `Mandarin` | Paraformer (FunASR) | g2pM | `zh_` |
| 英語 | `English` | Parakeet (NVIDIA) | g2p_en | `en_` |
| 広東語 | `Cantonese` | — | ToJyutping | `yue_` |

### 音素辞書の内訳（`phone_set.json`）

- 特殊トークン: 10個（`<PAD>`, `<SP>`, `<AP>`, `<UNK>`, `<BOW>`, `<EOW>`, `<BOS>`, `<EOS>`, `<MASK>`, `<SEP>`）
- `en_` 音素: 70個
- `zh_` 音素: 1,201個
- `yue_` 音素: 1,529個
- **合計 vocab_size**: 3,000

## 現在の状態

- 推論コードとモデルは公開済み
- **学習コードは未公開**
- Web UI、HuggingFace Spacesデモ、評価ベンチマークは未リリース（ロードマップに記載）

## 依存関係

依存関係は `pyproject.toml` で管理し、`uv` で環境構築を行う。

### 推論用（`[project.dependencies]`）

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| torch | 2.10.0 | PyTorch |
| torchaudio | 2.10.0 | 音声処理 |
| torchcodec | 0.10.0 | コーデック |
| transformers | 4.41.2 | Hugging Face |
| accelerate | 1.11.0 | GPU高速化 |
| librosa | 0.11.0 | 音声解析 |
| soundfile | 0.13.1 | 音声I/O |
| omegaconf | 2.3.0 | 設定管理 |
| gradio | 6.3.0 | UI |
| numpy | 2.2.6 | 数値計算 |
| scipy | 1.15.3 | 科学計算 |

### 前処理用（`[project.optional-dependencies] preprocess`）

主要な追加パッケージ: funasr, nemo_toolkit, g2p_en, g2pM, ToJyutping, mido, pretty_midi, pyworld, webrtcvad, sageattention 等

`uv sync --extra preprocess` で追加インストール可能。
