# GPT-2 mini (PyTorch, Mac/MPS) — 学習して仕組みを理解するための最小実装

このリポジトリは「LLMの仕組み理解」を目的に、Decoder-only Transformer（GPT-2系）を **小さく実装して学習** します。

- 目的：実装の完成度や最速化ではなく、**学習が成立する条件 / 失敗する条件を観測して腹落ち** させる
- 実装方針：`nn.Transformer` は使わず、Attention / MLP / LayerNorm / Residual を自前で組む
- 実行環境：Mac（MPS）優先。無ければCPUで動作

---

## Quick Start

### 1) 依存関係
```bash
pip install torch
```

### 2) データ（任意）

学習用テキストを `data.txt` として置く（UTF-8のプレーンテキスト）。
なければスクリプト内の小さなサンプルで動きます（学習は弱いので推奨は `data.txt`）。

### 3) 学習

```bash
python train_gpt2_mini.py --data data.txt --steps 2000
```

---

## 学びを最大化する実験（おすすめ）

### A) Residual を消す（だいたい壊れる）

```bash
python train_gpt2_mini.py --data data.txt --disable_residual 1 --steps 800
```

### B) LayerNorm を Post-LN にする（不安定さが見える）

```bash
python train_gpt2_mini.py --data data.txt --ln_style post --steps 1500
```

### C) Attention を消して MLP だけにする（何ができなくなるか）

```bash
python train_gpt2_mini.py --data data.txt --disable_attention 1 --steps 1500
```

---

## 何が「成功」か
- loss が継続的に下がる
- 生成サンプルが、データの癖（単語や句読点、改行パターン）を学び始める
- 実験スイッチで挙動が変わり、「なぜそうなるか」を説明できる

---

## リポジトリ構成（想定）

```
.
├── train_gpt2_mini.py      # 学習・生成スクリプト（単一ファイル）
├── data.txt                # 学習用テキスト（任意）
├── docs/
│   ├── experiments.md      # 実験メモ（観測テンプレ）
│   └── notes.md            # 理解メモ（式・実装対応）
└── LICENSE
```

---

## Roadmap（短期）
- loss/grad_norm に加えて activation統計（LN前後）をログ化
- attention entropy をログ化（「どれだけ鋭く見ているか」）
- 学習を安定させるハイパラ（LR, warmup, weight decay）比較
- （余力があれば）BPE/tokenizer と小規模データセットでの比較

---

## License

MIT（予定）。必要なら変更してください。
