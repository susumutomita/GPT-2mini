# Experiments

このドキュメントは「壊して観測して理解する」ための実験ログ。

---

## 観測テンプレ

- 日付：
- コマンド：
- データ：
- モデル：
  - layers / heads / embd / block_size：
  - LN style（pre/post）：
  - residual（on/off）：
  - attention（on/off）：
- ハイパラ：
  - lr / wd / dropout / grad_clip：
- 結果：
  - train loss：
  - val loss：
  - grad_norm：
  - 生成サンプル（抜粋）：
- 所感（重要）：
  - 何が起きた？
  - 何が原因っぽい？
  - 次の一手は？

---

## 実験1：Baseline（Pre-LN, Residual ON, Attention ON）

- コマンド：
  ```bash
  python train_gpt2_mini.py --data data.txt --steps 2000
  ```
- 期待：
  - loss が下がる
  - 生成がデータの癖を持ち始める

---

## 実験2：Residual OFF（学習が壊れる）

- コマンド：
  ```bash
  python train_gpt2_mini.py --data data.txt --disable_residual 1 --steps 800
  ```
- 観測ポイント：
  - loss が下がりにくい/発散しやすい
  - grad_norm が不安定になるか
- 理解メモ：
  - residual が「情報の高速道路」になっている
  - deep になるほど必要性が増す

---

## 実験3：Post-LN（不安定化の体感）

- コマンド：
  ```bash
  python train_gpt2_mini.py --data data.txt --ln_style post --steps 1500
  ```
- 観測ポイント：
  - 学習の立ち上がりが遅い/発散しやすい
  - LR 依存が強くなる傾向

---

## 実験4：Attention OFF（MLP-only）

- コマンド：
  ```bash
  python train_gpt2_mini.py --data data.txt --disable_attention 1 --steps 1500
  ```
- 観測ポイント：
  - 直近のパターンは覚えるが、依存が伸びない
  - "並び"の一般化が弱い
