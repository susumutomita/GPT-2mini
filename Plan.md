# Development Plan

## 実行計画 (Exec Plans)

### train_gpt2_mini.py 実装 - 2026-01-07

**目的 (Objective)**:
- Decoder-only Transformer（GPT-2 系）を単一ファイルで実装
- `nn.Transformer` を使わず、Attention / MLP / LayerNorm / Residual を自前で組む
- 「壊して観測して理解する」ための実験スイッチを提供

**制約 (Guardrails)**:
- 単一ファイル（train_gpt2_mini.py）で完結
- Mac（MPS）優先、無ければ CPU で動作
- PyTorch のみ依存（pip install torch で動く）
- 学習が成立する条件を観測できる設計

**タスク (TODOs)**:

#### Phase 1: コアモジュール実装
- [x] Token Embedding + Positional Embedding
- [x] Causal Self-Attention（Multi-Head）
- [x] MLP（Feed-Forward Network）
- [x] Transformer Block（Pre-LN / Post-LN 切替可能）
- [x] GPT モデル全体

#### Phase 2: 学習インフラ
- [x] Character-level Tokenizer（シンプル実装）
- [x] データローダー（data.txt 読み込み）
- [x] 学習ループ（AdamW, grad clipping）
- [x] ロギング（loss, grad_norm）

#### Phase 3: 実験スイッチ
- [x] `--disable_residual`: Residual 接続を無効化
- [x] `--disable_attention`: Attention を無効化（MLP-only）
- [x] `--ln_style pre|post`: LayerNorm の位置切替

#### Phase 4: 生成と検証
- [x] テキスト生成（sampling, temperature）
- [x] 学習中のサンプル生成
- [x] MPS/CPU デバイス自動選択

#### Phase 5: 追加機能
- [x] チェックポイント保存・読み込み（--save/--load）
- [x] 推論専用モード（--generate）
- [x] 対話モード（--interactive）
- [x] ブログ記事執筆（Transformer 解説 + GPT→Claude 進化）

**検証手順 (Validation)**:
```bash
# 基本動作確認
python train_gpt2_mini.py --steps 100

# data.txt での学習
python train_gpt2_mini.py --data data.txt --steps 2000

# 実験スイッチ確認
python train_gpt2_mini.py --disable_residual 1 --steps 100
python train_gpt2_mini.py --ln_style post --steps 100
python train_gpt2_mini.py --disable_attention 1 --steps 100
```

**アーキテクチャ設計**:
```
GPT2Mini
├── tok_emb: nn.Embedding(vocab_size, n_embd)
├── pos_emb: nn.Embedding(block_size, n_embd)
├── drop: nn.Dropout(dropout)
├── blocks: nn.ModuleList([Block x n_layer])
│   └── Block
│       ├── ln1: nn.LayerNorm(n_embd)
│       ├── attn: CausalSelfAttention
│       │   ├── c_attn: nn.Linear(n_embd, 3 * n_embd)  # Q, K, V
│       │   ├── c_proj: nn.Linear(n_embd, n_embd)
│       │   └── causal_mask: tril buffer
│       ├── ln2: nn.LayerNorm(n_embd)
│       └── mlp: MLP
│           ├── c_fc: nn.Linear(n_embd, 4 * n_embd)
│           ├── gelu: nn.GELU()
│           └── c_proj: nn.Linear(4 * n_embd, n_embd)
├── ln_f: nn.LayerNorm(n_embd)
└── lm_head: nn.Linear(n_embd, vocab_size, bias=False)
```

**デフォルトハイパーパラメータ**:
- n_layer: 4
- n_head: 4
- n_embd: 128
- block_size: 256
- dropout: 0.1
- learning_rate: 3e-4
- weight_decay: 0.1
- grad_clip: 1.0

**未解決の質問 (Open Questions)**:
- [x] Weight tying（tok_emb と lm_head の重み共有）を実装するか？ → 実装済み

**進捗ログ (Progress Log)**:
- [2026-01-07 開始] Plan.md 作成、実装開始
- [2026-01-07] train_gpt2_mini.py 実装完了（約400行）
  - コアモジュール: CausalSelfAttention, MLP, Block, GPT2Mini
  - 学習インフラ: CharDataset, train(), AdamW + cosine LR
  - 実験スイッチ: --disable_residual, --disable_attention, --ln_style
  - 生成機能: generate() with temperature, top_k
- [2026-01-07] 動作検証完了
  - MPS デバイスで正常動作
  - 50 steps で loss 4.6 → 2.3 に収束
  - 全実験スイッチの動作確認済み
- [2026-01-07] チェックポイント・推論モード追加（PR #11）
  - --save/--load: モデルの保存・読み込み
  - --generate: 推論のみモード
  - --interactive: 対話的生成モード
  - .gitignore にモデルファイルを追加
- [2026-01-07] ブログ記事 2 本公開（PR #198）
  - gpt2-mini-transformer-learning.md: Transformer の仕組み解説
  - gpt2-mini-to-claude-evolution.md: GPT-2 から Claude への進化

**検証結果**:
| 設定 | 30step後 Loss | 特徴 |
|------|--------------|------|
| Baseline (Pre-LN) | 2.3 | 安定して学習 |
| Residual OFF | 4.15 | 学習がほぼ停滞（情報が流れない） |
| Post-LN | 3.64 | 学習は進むが Pre-LN より遅い |
| Attention OFF | 2.78 | 高速だが文脈依存が弱い |

---

## 振り返り (Retrospective)

### 問題1: データ長 < block_size でクラッシュ
- **問題**: サンプルデータが短く、block_size=256 でエラー発生
- **原因**: データ長チェックなしで randint を呼んでいた
- **対策**: block_size を自動調整するコードを追加
