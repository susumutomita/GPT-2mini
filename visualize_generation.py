#!/usr/bin/env python3
"""
GPT-2 mini の生成過程を可視化するスクリプト

モデルが「どう考えて」次の文字を選んでいるかを表示:
1. 各ステップの確率分布（Top-k）
2. 選ばれた文字とその確率
3. 累積テキスト
"""

import argparse
import torch
import torch.nn.functional as F
from train_gpt2_mini import GPT2Mini, GPTConfig, CharDataset


def visualize_generation(
    model: GPT2Mini,
    dataset: CharDataset,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 10,
    device: str = "cpu",
):
    """生成過程を可視化"""
    model.eval()

    # プロンプトをトークン化
    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    generated_text = prompt

    print("=" * 70)
    print(f"プロンプト: 「{prompt}」")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("=" * 70)
    print()

    with torch.no_grad():
        for step in range(max_tokens):
            # block_size を超えないようにクロップ
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature  # 最後のトークンの logits

            # 確率分布を計算
            probs = F.softmax(logits, dim=-1)

            # Top-k の候補を取得
            top_probs, top_indices = torch.topk(probs[0], min(top_k, probs.size(-1)))

            # 表示
            print(f"--- Step {step + 1} ---")
            print(f"現在のテキスト: 「{generated_text}」")
            print()
            print("次の文字の候補（確率順）:")
            print("-" * 40)

            for i, (prob, token_idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
                char = dataset.itos.get(token_idx, "?")
                # 特殊文字の表示を工夫
                if char == "\n":
                    char_display = "\\n（改行）"
                elif char == " ":
                    char_display = "␣（空白）"
                elif char == "\t":
                    char_display = "\\t（タブ）"
                else:
                    char_display = f"「{char}」"

                bar = "█" * int(prob * 50)  # 確率バー
                marker = " ← 選択" if i == 0 else ""
                print(f"  {i+1:2d}. {char_display:10s} {prob*100:5.1f}% {bar}{marker}")

            print()

            # Top-k filtering して実際にサンプリング
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs_filtered = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs_filtered, num_samples=1)

            # 選ばれた文字
            selected_char = dataset.itos.get(idx_next.item(), "?")
            selected_prob = probs[0, idx_next.item()].item()

            print(f">>> 選択された文字: 「{selected_char}」（確率: {selected_prob*100:.1f}%）")
            print()

            # 連結
            idx = torch.cat((idx, idx_next), dim=1)
            generated_text += selected_char

            # 改行があったら少し区切り
            if selected_char == "\n":
                print("=" * 70)
                print()

    print("=" * 70)
    print("最終生成テキスト:")
    print("-" * 70)
    print(generated_text)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="GPT-2 mini 生成過程の可視化")
    parser.add_argument("--data", type=str, required=True, help="学習データファイル")
    parser.add_argument("--load", type=str, required=True, help="チェックポイントファイル")
    parser.add_argument("--prompt", type=str, default="吾輩は", help="開始プロンプト")
    parser.add_argument("--max_tokens", type=int, default=30, help="生成トークン数")
    parser.add_argument("--temperature", type=float, default=0.8, help="サンプリング温度")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k サンプリング")

    # Model config (checkpoint と合わせる)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=512)

    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load checkpoint first to get vocab
    checkpoint = torch.load(args.load, map_location=device, weights_only=False)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = len(stoi)
    print(f"Loaded checkpoint from {args.load}")
    print(f"Vocabulary size: {vocab_size}")

    # Create dummy dataset with checkpoint's vocab
    class DummyDataset:
        def __init__(self, stoi, itos):
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)

        def encode(self, s):
            return [self.stoi.get(c, 0) for c in s]

        def decode(self, tokens):
            return "".join([self.itos.get(t, "?") for t in tokens])

    dataset = DummyDataset(stoi, itos)

    # Config
    config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        vocab_size=vocab_size,
    )

    # Model
    model = GPT2Mini(config).to(device)
    model.load_state_dict(checkpoint["model"])
    print()

    # Visualize
    visualize_generation(
        model=model,
        dataset=dataset,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )


if __name__ == "__main__":
    main()
