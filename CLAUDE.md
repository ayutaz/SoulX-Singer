# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

SoulX-Singerは、ゼロショット歌声合成（SVS）モデル。未知の歌手の高品質な歌声をファインチューニングなしで生成可能。メロディ制御（F0 contour）とスコア制御（MIDIノート）の2つの制御モードに対応。対応言語は中国語（普通話）、英語、広東語。

## 環境構築

```bash
# 推論用のみ
uv sync

# 前処理も含む場合
uv sync --extra preprocess
```

## モデルダウンロード

```bash
uv run huggingface-cli download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer
uv run huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess
```

## コマンド

### 推論の実行

```bash
# PYTHONPATHをプロジェクトルートに設定する必要あり
export PYTHONPATH=$(pwd):$PYTHONPATH

# サンプル実行
bash example/infer.sh

# 直接実行
uv run python -m cli.inference \
    --device cuda \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --prompt_wav_path <プロンプト音声パス> \
    --prompt_metadata_path <プロンプトメタデータJSON> \
    --target_metadata_path <ターゲットメタデータJSON> \
    --phoneset_path soulxsinger/utils/phoneme/phone_set.json \
    --save_dir <出力先> \
    --control score   # melody または score
```

`--auto_shift` フラグでプロンプトとターゲットのF0中央値に基づく自動ピッチシフトが有効になる。

### 前処理パイプライン

```bash
uv run python -m preprocess.pipeline \
    --audio_path <入力音声> \
    --save_dir <出力先> \
    --language Mandarin   # Mandarin / English / Cantonese
    --device cuda \
    --vocal_sep True      # ボーカル分離の有無
    --max_merge_duration 30000
```

## アーキテクチャ

### ディレクトリ構成

- `cli/` - CLI推論エントリーポイント (`inference.py`)
- `soulxsinger/` - コアモデルパッケージ
  - `config/` - モデル設定YAML
  - `models/` - メインモデルとサブモジュール
  - `utils/` - データ処理、音声ユーティリティ、音素辞書
- `preprocess/` - 前処理パイプライン（ボーカル分離、F0抽出、ASR、ノート転写）
  - `tools/` - 各前処理ツール（f0_extraction, g2p, vocal_separation, note_transcription, midi_parser等）
  - `tools/midi_editor/` - TypeScript/React製MIDIエディタ
- `example/` - サンプル音声、メタデータ、実行スクリプト

### 推論パイプライン

```
プロンプト音声 + ターゲットメタデータ(JSON)
  → DataProcessor: 音素・ピッチ・タイミング情報をテンソルに変換
  → SoulXSinger.infer():
      → Embedding (音素/ピッチ/タイプ) + ConvNeXt前処理
      → expand_states(): トークンレベル → メルフレームレベルに展開
      → CFMDecoder (FlowMatchingTransformer, 22層): メルスペクトログラム生成
      → Vocos (ConvNeXt + ISTFT): 波形合成 (24kHz)
  → セグメント単位で生成し、オーバーラップ付きで結合
```

### 前処理パイプライン

```
入力音声
  → Mel-Band Roformer: ボーカル分離 + デリバーブ
  → RMVPE: F0（基本周波数）抽出
  → VAD: 音声区間検出
  → ASR: 歌詞文字起こし (Paraformer=中国語 / Parakeet=英語)
  → ROSVOT: ノートレベルのピッチ・デュレーション推定
  → セグメントマージ → メタデータJSON出力
```

### コアモデル構成

- **エンコーダ**: 音素(vocab=3000), ピッチ(256), タイプ(256), F0(361ビン, C1-B6) の各Embedding(dim=512) + ConvNeXtV2Block×4
- **FlowMatchingTransformer**: hidden=1024, layers=22, heads=16, CFG dropout=0.2, cosineスケジューラ。デフォルト32ステップ、CFG=3で拡散生成
- **Vocos Vocoder**: ConvNeXtバックボーン(30層) + ISTFTHead。メルスペクトログラム→波形変換

### メタデータJSON形式

各セグメントに以下のフィールドを持つ:
- `time`: [開始ms, 終了ms]
- `phoneme`: スペース区切り音素列
- `duration`: 各ノートの長さ（秒）
- `f0`: フレームレベルF0値（Hz）
- `note_pitch` / `note_type`: MIDIピッチとノートタイプ
- `words` / `word_durs`: 歌詞と各単語の長さ

### 重要な技術的ポイント

- `PYTHONPATH`をプロジェクトルートに設定しないとモジュールインポートが失敗する
- 推論はセグメント単位で行い、オーバーラップ付きで結合する（長尺対応）
- F0は361ビン（C1-B6、約7オクターブ）に離散化される
- DataProcessorは`<BOW>`/`<EOW>`トークンを挿入し、英語は`-`区切り＋`<SEP>`マーカーを使用
- メルスペクトログラムのパラメータ: FFT=1920, hop=480, mels=128, SR=24kHz, mean=-4.92, var=8.14
- 設定はOmegaConfで管理（`soulxsinger/config/soulxsinger.yaml`）
