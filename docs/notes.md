# Notes（理解メモ）

実装と概念の対応を残すメモ。あとで読み返して「腹落ち」を再現できる形にする。

---

## 1. 目的

- Transformer を「読める・説明できる」状態にする
- 学習が成立する要因（Residual/LN/Attention/最適化）を、実験で裏付ける

---

## 2. Decoder-only Transformer（GPT系）の流れ

1) Token Embedding + Positional Embedding
2) N 回 Block を通す
3) 最後に LayerNorm
4) Linear（語彙次元）→ logits
5) Cross Entropy（次トークン）

---

## 3. Blockの中身

### Pre-LN（安定）
- x -> LN -> Attention -> +residual
- x -> LN -> MLP -> +residual

Pre-LN は勾配が流れやすく、深くしても学習が成立しやすい。

### Post-LN（不安定）
- Attention -> +residual -> LN
- MLP -> +residual -> LN

Post-LN は初期条件や LR にシビアになりがち（挙動観測に向く）。

---

## 4. Causal Self-Attention

- Q,K,V を線形変換で作る
- スケールド内積：softmax(QK^T / sqrt(d))
- causal mask（未来を見ない）
- 重み付き和で出力

### 観測テーマ

- Attention entropy が下がる（尖る）と何が起きる？
- context length を伸ばした時、どこまで依存が伸びる？

---

## 5. "壊して学ぶ"観点

- Residual OFF：深さ方向に情報が運べなくなる
- Attention OFF：位置をまたぐ情報が混ざらない（MLP は局所変換）
- LN 位置変更：勾配・分布の安定性が変わる

---

## 6. 次に追加したいログ

- activation mean/std（LN 前後）
- grad norm（層別）
- attention entropy（ヘッド別の平均）

理解の軸：
「分布が崩れた結果、何が破綻したか」を追えるようにする。
