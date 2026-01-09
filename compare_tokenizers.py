#!/usr/bin/env python3
"""
Character-level と BPE トークナイザーの比較

Usage:
    python compare_tokenizers.py
"""

import torch
from train_gpt2_mini import CharDataset, GPT2Mini, GPTConfig
from train_gpt2_bpe import BPEDataset, GPT2Mini as GPT2MiniBPE


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    print("=" * 70)
    print("Character-level vs BPE トークナイザー比較")
    print("=" * 70)
    print()

    # テストテキスト
    test_texts = [
        "吾輩は猫である",
        "私は学生です",
        "今日は良い天気ですね",
        "Hello, World!",
        "龍",  # Character-level では未知文字
        "😀",  # 絵文字
    ]

    # データ読み込み
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("=== トークナイザー比較 ===")
    print()

    # Character-level トークナイザー
    print("【Character-level】")
    char_dataset = CharDataset(text[:10000], block_size=256)  # 小さいデータで初期化
    print(f"  語彙サイズ: {char_dataset.vocab_size}")
    print()

    # BPE トークナイザー（学習済みモデルからロード）
    print("【BPE】")
    checkpoint = torch.load("model_bpe.pt", map_location=device, weights_only=False)
    bpe_tokenizer = checkpoint["tokenizer"]
    print(f"  語彙サイズ: {bpe_tokenizer.vocab_size}")
    print()

    print("=== トークン化の例 ===")
    print()

    for text_sample in test_texts:
        print(f'入力: "{text_sample}"')
        print("-" * 50)

        # Character-level
        try:
            char_tokens = [char_dataset.stoi[c] for c in text_sample]
            char_decoded = "".join(char_dataset.itos[t] for t in char_tokens)
            print(f"  Char: {len(char_tokens)} トークン → {char_tokens[:10]}...")
            print(f"  復元: {char_decoded}")
        except KeyError as e:
            print(f"  Char: エラー - 未知文字 {e}")

        # BPE
        bpe_tokens = bpe_tokenizer.encode(text_sample)
        bpe_decoded = bpe_tokenizer.decode(bpe_tokens)
        print(f"  BPE:  {len(bpe_tokens)} トークン → {bpe_tokens[:10]}...")
        print(f"  復元: {bpe_decoded}")

        print()

    print("=== モデル比較 ===")
    print()

    # Character-level モデル
    print("【Character-level モデル (model_v2.pt)】")
    char_checkpoint = torch.load("model_v2.pt", map_location=device, weights_only=False)
    char_config = char_checkpoint["config"]
    print(f"  語彙サイズ: {char_config.vocab_size}")
    print(f"  レイヤー数: {char_config.n_layer}")
    print(f"  ヘッド数: {char_config.n_head}")
    print(f"  埋め込み次元: {char_config.n_embd}")
    print(f"  コンテキスト長: {char_config.block_size}")

    # BPE モデル
    print()
    print("【BPE モデル (model_bpe.pt)】")
    bpe_config = checkpoint["config"]
    print(f"  語彙サイズ: {bpe_config.vocab_size}")
    print(f"  レイヤー数: {bpe_config.n_layer}")
    print(f"  ヘッド数: {bpe_config.n_head}")
    print(f"  埋め込み次元: {bpe_config.n_embd}")
    print(f"  コンテキスト長: {bpe_config.block_size}")

    print()
    print("=== 生成比較 ===")
    print()

    prompt = "吾輩は"

    # Character-level 生成
    print(f"【Character-level】プロンプト: '{prompt}'")
    full_char_dataset = CharDataset(text, block_size=char_config.block_size)
    char_model = GPT2Mini(char_config).to(device)
    char_model.load_state_dict(char_checkpoint["model"])
    char_model.train(False)

    idx = torch.tensor([[full_char_dataset.stoi[c] for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = char_model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
    char_output = "".join(full_char_dataset.itos[t] for t in generated[0].tolist())
    print(char_output)
    print()

    # BPE 生成
    print(f"【BPE】プロンプト: '{prompt}'")
    bpe_model = GPT2MiniBPE(bpe_config).to(device)
    bpe_model.load_state_dict(checkpoint["model"])
    bpe_model.train(False)

    idx = torch.tensor([bpe_tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = bpe_model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
    bpe_output = bpe_tokenizer.decode(generated[0].tolist())
    print(bpe_output)
    print()

    print("=== まとめ ===")
    print()
    print("| 項目 | Character-level | BPE |")
    print("|------|-----------------|-----|")
    print(f"| 語彙サイズ | {char_config.vocab_size} | {bpe_config.vocab_size} |")
    print(f"| 未知文字対応 | × | ○ |")
    print(f"| 日本語の文字化け | なし | あり（バイトレベル） |")
    print(f"| トークン効率 | 1文字=1トークン | 圧縮あり |")


if __name__ == "__main__":
    main()
