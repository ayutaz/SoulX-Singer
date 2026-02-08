# アーキテクチャ詳細

## ディレクトリ構成

```
SoulX-Singer/
├── cli/
│   └── inference.py              # 推論CLIエントリーポイント
├── soulxsinger/                  # コアモデルパッケージ
│   ├── config/
│   │   └── soulxsinger.yaml      # モデル設定
│   ├── models/
│   │   ├── soulxsinger.py        # メインモデルクラス
│   │   └── modules/
│   │       ├── convnext.py       # ConvNeXtV2前処理ブロック
│   │       ├── decoder.py        # CFMDecoderラッパー
│   │       ├── flow_matching.py  # Flow Matching Transformer（中核）
│   │       ├── llama.py          # DiffLlamaアーキテクチャ
│   │       ├── mel_transform.py  # メルスペクトログラム変換
│   │       └── vocoder.py        # Vocos ニューラルボコーダ
│   └── utils/
│       ├── audio_utils.py        # 音声処理ユーティリティ
│       ├── data_processor.py     # データ前処理（推論用）
│       ├── file_utils.py         # ファイルI/O
│       ├── pitch_utils.py        # F0/ピッチ変換
│       └── phoneme/
│           └── phone_set.json    # 音素辞書（3,000語彙）
├── preprocess/                   # 前処理パイプライン
│   ├── pipeline.py               # メインオーケストレータ
│   ├── utils.py                  # ユーティリティ
│   ├── requirements.txt          # 前処理用依存関係
│   └── tools/
│       ├── f0_extraction.py      # F0抽出（RMVPE）
│       ├── g2p.py                # 書記素→音素変換
│       ├── lyric_transcription.py # ASR歌詞文字起こし
│       ├── midi_parser.py        # MIDI ↔ メタデータ変換
│       ├── note_transcription/   # ノート転写（ROSVOT）
│       ├── vocal_separation/     # ボーカル分離（Roformer）
│       └── midi_editor/          # Web MIDIエディタ（TypeScript/React）
├── example/                      # サンプルデータ・スクリプト
│   ├── audio/                    # サンプル音声・メタデータ
│   ├── infer.sh                  # 推論実行スクリプト
│   └── preprocess.sh             # 前処理実行スクリプト
└── pretrained_models/            # ダウンロードしたモデル（.gitignore対象）
```

## 全体パイプライン

```
[入力音声] ──→ [前処理パイプライン] ──→ [メタデータJSON]
                                            │
[プロンプト音声] ──────────────────┐         │
[プロンプトメタデータJSON] ────────┤         │
[ターゲットメタデータJSON] ────────┴─→ [SoulXSinger推論] ──→ [合成歌声WAV 24kHz]
```

## 前処理パイプライン詳細

`preprocess/pipeline.py` の `PreprocessPipeline` クラスが5段階の処理を統括。

```
入力音声
  │
  ├─ 1. Vocal Separation（ボーカル分離）
  │     Mel-Band Roformer + Dereverberation（残響除去）
  │     → 伴奏を除去し、クリーンなボーカルトラックを抽出
  │
  ├─ 2. F0 Extraction（基本周波数抽出）
  │     RMVPE（深層U-Net）
  │     → フレームレベルのF0値（Hz）を推定
  │
  ├─ 3. VAD（音声区間検出）
  │     WebRTC VAD
  │     → 音声のある区間を検出・セグメント分割
  │
  ├─ 4. Lyric Transcription（歌詞文字起こし）
  │     Paraformer（中国語）/ Parakeet（英語）
  │     → 歌詞テキストを認識
  │
  ├─ 5. Note Transcription（ノート転写）
  │     ROSVOT（U-Net系）
  │     → ノートレベルのピッチ・デュレーションを推定
  │
  └─ 6. Segment Merge & Export
        短いセグメントをマージ → メタデータJSON出力
```

## 推論パイプライン詳細

### SoulXSingerモデル構造

```
入力: メタデータ (phoneme_id, note_pitch, note_type, f0)
  │
  ├─ note_text_encoder:   Embedding(3000, 512)  ← 音素ID
  ├─ note_pitch_encoder:  Embedding(256, 512)   ← ノートピッチ
  ├─ note_type_encoder:   Embedding(256, 512)   ← ノートタイプ
  │
  ├─ 加算 → preflow: ConvNeXtV2Block × 4       ← 特徴量精錬
  │
  ├─ expand_states()                            ← トークン→メルフレーム展開
  │
  ├─ f0_encoder: Embedding(361, 512)            ← F0（361ビン, C1-B6）
  │   加算
  │
  └─ CFMDecoder (FlowMatchingTransformer)       ← 拡散生成
      │  hidden_size: 1024
      │  num_layers: 22
      │  num_heads: 16
      │  n_steps: 32（デフォルト）
      │  cfg: 3（classifier-free guidance）
      │
      └─ Vocos Vocoder                          ← 波形合成
          ConvNeXt 30層 + ISTFTHead
          → 出力: 24kHz WAVファイル
```

### FlowMatchingTransformer（拡散モデル中核）

- **ベースアーキテクチャ**: DiffLlama（Llama系Transformer + 拡散ステップ条件付き）
- **ODE定義**: `xt = ((1 - (1 - sigma) * t) * z + t * x) * mask`
- **推論**: `reverse_diffusion()` でn_steps回の逆拡散
- **CFG**: classifier-free guidanceで無条件/条件付き生成を混合
- **マスキング**: プロンプト領域は固定、生成領域のみ拡散

### Vocos Vocoder

- Amphionプロジェクト由来の実装
- **ConvNeXtバックボーン**: 30層
- **ISTFTHead**: STFT振幅 + 位相を予測 → 逆STFTで波形復元
- **凍結状態**: 推論時は `requires_grad=False`, `eval()` モード
- メルスペクトログラム → 波形変換（24kHz）

### DataProcessor

`soulxsinger/utils/data_processor.py` がメタデータJSONからテンソルへの変換を担当。

主な処理:
1. 連続する同一音素のマージ
2. `<PAD>` プレフィックストークンの挿入
3. 各ノートに `<BOW>` (begin-of-word) / `<EOW>` (end-of-word) トークンを挿入
4. 英語はハイフン(`-`)区切りで `<SEP>` マーカーを使用
5. メルフレームレベルへのアライメントインデックス生成
6. F0長のメルフレーム数への整合

### セグメント単位推論

長い楽曲は一度に生成せず、セグメント単位で推論:
1. ターゲットメタデータの各セグメントを順に処理
2. 各セグメントで独立にメルスペクトログラムを生成
3. オーバーラップ付きで結合し、継ぎ目を滑らかに

## 設定パラメータ

`soulxsinger/config/soulxsinger.yaml` の全パラメータ:

```yaml
infer:
  n_steps: 32              # 拡散サンプリングステップ数
  cfg: 3                   # Classifier-free guidance スケール

audio:
  hop_size: 480            # STFTホップサイズ
  sample_rate: 24000       # サンプルレート
  max_length: 36000        # 最大フレーム数
  n_fft: 1920              # FFTサイズ
  num_mels: 128            # メルバンド数
  win_size: 1920           # 窓サイズ
  fmin: 0                  # 最低周波数
  fmax: 12000              # 最高周波数
  mel_var: 8.14            # メル正規化分散
  mel_mean: -4.92          # メル正規化平均

model:
  encoder:
    vocab_size: 3000       # 音素語彙サイズ
    text_dim: 512          # 音素Embedding次元
    pitch_dim: 512         # ピッチEmbedding次元
    type_dim: 512          # タイプEmbedding次元
    f0_bin: 361            # F0量子化ビン数 (C1-B6)
    f0_dim: 512            # F0 Embedding次元
    num_layers: 4          # ConvNeXtブロック数

  flow_matching:
    mel_dim: 128           # メル次元
    hidden_size: 1024      # Transformer隠れ層
    num_layers: 22         # Transformerレイヤー数
    num_heads: 16          # アテンションヘッド数
    cfg_drop_prob: 0.2     # CFGドロップアウト率
    use_embedding: False   # 条件付けにEmbeddingを使用
    cond_codebook_size: 512
    cond_scale_factor: 1
    sigma: 1e-5            # ノイズスケジュールパラメータ
    time_scheduler: cos    # 時間スケジューラ（コサイン）
```

## メタデータJSON形式

各セグメントが以下のフィールドを持つ:

```json
{
  "time": [0, 5000],
  "phoneme": "zh_shi4 zh_jie4 zh_zhong1 ...",
  "duration": "0.15 0.20 0.30 ...",
  "f0": "261.63 293.66 329.63 ...",
  "note_pitch": "60 62 64 ...",
  "note_type": "1 1 1 ...",
  "words": "世界 中 ...",
  "word_durs": "0.35 0.20 ..."
}
```

| フィールド | 説明 |
|-----------|------|
| `time` | セグメントの開始・終了時刻（ミリ秒） |
| `phoneme` | スペース区切りの音素列（言語プレフィックス付き） |
| `duration` | 各ノートの長さ（秒） |
| `f0` | フレームレベルのF0値（Hz） |
| `note_pitch` | MIDIノート番号 |
| `note_type` | ノートタイプ（歌唱/休符など） |
| `words` | 歌詞の単語列 |
| `word_durs` | 各単語の長さ（秒） |
