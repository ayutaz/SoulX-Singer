# 学習コード分析

## 概要

SoulX-Singerは推論コードのみ公開されているが、コードベース内に学習に必要な核心的なコンポーネントが既に実装されている。本ドキュメントでは、学習コード構築の実現可能性と必要な作業を分析する。

## 既存の学習関連コード

### 1. Flow Matching 損失関数（`flow_matching.py`）

#### `compute_loss()` メソッド

学習時のメイン損失計算。ランダムなタイムステップをサンプリングし、フロー予測の誤差を計算する。

```python
# flow_matching.py より
def compute_loss(self, x, cond, mask, is_prompt, ...):
    # タイムステップ t をランダムサンプリング [sigma, 1.0]
    t = torch.rand([b, 1, 1], ...)
    t = 1 - torch.cos(t * 0.5 * math.pi)  # コサインスケジューラ
    # loss_t() を呼び出して損失計算
    return self.loss_t(x, cond, mask, t, is_prompt, ...)
```

#### `loss_t()` メソッド

特定のタイムステップでの損失計算。Conditional Flow Matching (CFM) の損失を実装。

```python
def loss_t(self, x, cond, mask, t, is_prompt, ...):
    # ノイズ追加: xt = ((1-(1-sigma)*t) * z + t * x) * mask
    # フロー予測: model(xt, ...)
    # 返り値: (noise, x, flow_pred, final_mask, prompt_len) + 補助損失
```

#### `forward_diffusion()` メソッド

学習時のノイズ追加プロセス。CFGドロップアウトを含む。

```python
def forward_diffusion(self, x, t, is_prompt=None):
    # ODE: xt = ((1 - (1 - sigma) * t) * z + t * x) * mask
    # CFGドロップアウト: prob=0.2 で条件をランダムに除去
    # マスキング: プロンプト領域は固定、生成領域のみノイズ追加
```

#### 損失計算の例（コード内コメントより）

```python
noise, x, flow_pred, final_mask, prompt_len = outputs["output"]
flow_gt = x - (1 - 1e-5) * noise  # グランドトゥルースのフロー
diff_loss = F.l1_loss(flow_pred, flow_gt, reduction="none") * final_mask
diff_loss = torch.mean(diff_loss, dim=2).sum() / final_mask.sum()
```

### 2. CFMDecoder（`decoder.py`）

FlowMatchingTransformerのラッパー。`forward()` メソッドが `compute_loss()` を呼び出す構造。

```python
class CFMDecoder(nn.Module):
    def forward(self, x, cond, mask, is_prompt, ...):
        return self.model.compute_loss(x, cond, mask, is_prompt, ...)
```

### 3. 補助損失

#### REPA損失（Realignment）

中間層の隠れ状態を使ったアライメント損失。指定レイヤーから特徴量を抽出し、MLP層で射影。

```python
# flow_matching.py 内
self.repa_mlp_layer = nn.Linear(hidden_size, repa_dim)
# 学習時: 隠れ状態をアライメントターゲットと比較
```

#### CTC損失

音素認識のための補助損失。中間層の出力を音素次元に射影。

```python
self.ctc_mlp_layer = nn.Linear(hidden_size, ctc_dim)
# 学習時: CTC損失で音素アライメントを補強
```

### 4. DiffLlama（`llama.py`）

学習に対応した構造:
- **Adaptive Layer Normalization**: 拡散ステップに条件付けされた正規化
- **勾配チェックポイント**: メモリ節約のためのサポート（一部未実装）
- **非因果的アテンション**: メルスペクトログラム全体を参照可能

### 5. Vocoder（`vocoder.py`）

明示的に凍結:
```python
self.vocoder.eval()
for param in self.vocoder.parameters():
    param.requires_grad = False
```
→ 学習時もVocoderは更新しない（事前学習済みを使用）

## 不足しているコンポーネント

### 1. SoulXSinger.forward()（学習用フォワードパス）

現状は `infer()` メソッドのみ。学習用の `forward()` または `compute_loss()` メソッドが必要。

**実装に必要な処理:**
1. プロンプト音声からメルスペクトログラムを抽出
2. ターゲットのメルスペクトログラム（グランドトゥルース）を取得
3. 条件エンコーディング（音素・ピッチ・タイプ・F0）
4. プロンプトとターゲットの条件を結合
5. CFMDecoder.forward() で損失計算
6. 補助損失（REPA, CTC）を加算

### 2. Dataset / DataLoader

メタデータJSON + 音声ファイルからバッチを構成するPyTorch Dataset。

**必要な機能:**
- 音声ファイルの読み込みとメルスペクトログラム変換
- メタデータJSONのパースとテンソル化
- バッチ内のパディング・マスキング
- プロンプト/ターゲットのペアリング

### 3. 学習ループ（train.py）

- Optimizer（AdamW推奨）
- Learning Rate Scheduler（Warmup + Cosine Decay等）
- 勾配クリッピング
- チェックポイント保存・復元
- ロギング（wandb/tensorboard）
- 分散学習（DDP）対応

### 4. 検証ループ

- 検証データでの損失計算
- サンプル音声生成（定期的に）
- メトリクス計算

## ファインチューニング戦略

### 推奨アプローチ

ゼロからの学習ではなく、既存モデルのファインチューニングが現実的。

```
既存チェックポイント (model.pt)
  │
  ├─ Embedding層の拡張
  │   note_text_encoder: Embedding(3000, 512) → Embedding(3000+N, 512)
  │   既存の重みは保持、新規音素分はランダム初期化
  │
  ├─ ConvNeXt前処理: 凍結 or 低学習率
  ├─ FlowMatchingTransformer: 低学習率で微調整
  ├─ Vocoder: 完全凍結
  │
  └─ 日本語歌声データで追加学習
```

### 凍結/非凍結の推奨設定

| コンポーネント | パラメータ数（概算） | 推奨 |
|---------------|---------------------|------|
| Embedding層（拡張分） | 小 | 学習（高学習率） |
| Embedding層（既存分） | 小 | 凍結 or 低学習率 |
| ConvNeXt前処理 | 中 | 低学習率 |
| FlowMatchingTransformer | 大（22層×1024） | 低学習率 |
| Vocoder | 大（30層） | 完全凍結 |

### 学習パラメータの推定

コードから読み取れるヒント:
- **CFGドロップアウト率**: 0.2（学習時に20%の確率で条件を除去）
- **損失関数**: L1 Loss（フロー予測 vs グランドトゥルース）
- **タイムステップスケジューラ**: コサイン（低ノイズ領域を重視）
- **シグマ**: 1e-5（ほぼノイズなしから開始）
- **マスキング**: プロンプト領域は損失計算から除外

## 実装の優先順位

| 優先度 | 作業 | 難易度 | 依存関係 |
|--------|------|--------|----------|
| 1 | 日本語音素辞書の定義 | 低 | なし |
| 2 | 日本語G2Pの実装 | 低 | 1 |
| 3 | 日本語ASRの統合 | 中 | なし |
| 4 | Dataset/DataLoaderの実装 | 中 | 1, 2 |
| 5 | SoulXSinger.forward()の実装 | 中 | なし |
| 6 | Embedding拡張ロジック | 低 | 1 |
| 7 | 学習ループ（train.py） | 中 | 4, 5, 6 |
| 8 | 検証・評価パイプライン | 中 | 7 |

## リスクと課題

### 技術的リスク

1. **データ量の不足**: 既存モデルは42,000時間で学習。少量の日本語データでは品質が低下する可能性
2. **カタストロフィック忘却**: ファインチューニングで既存言語の性能が低下するリスク
3. **音素体系の違い**: 日本語のモーラ構造と中国語の声調体系は根本的に異なる
4. **Vocoder**: 24kHz固定のVocoderが日本語特有の音韻に対応できるか未検証

### 緩和策

1. **段階的ファインチューニング**: まずEmbedding層のみ、次に全体を微調整
2. **混合学習**: 日本語データと既存言語データを混ぜて学習
3. **LoRA/PEFT**: パラメータ効率的なファインチューニング手法の検討
4. **データ増強**: ピッチシフト、テンポ変更等で学習データを拡充
