#!/usr/bin/env python3
"""
GPT-2 mini — 学習して仕組みを理解するための最小実装

目的: LLM の仕組み理解
実装方針: nn.Transformer は使わず、Attention / MLP / LayerNorm / Residual を自前で組む
実行環境: Mac（MPS）優先、無ければ CPU で動作

Usage:
    python train_gpt2_mini.py --data data.txt --steps 2000
    python train_gpt2_mini.py --disable_residual 1 --steps 800
    python train_gpt2_mini.py --ln_style post --steps 1500
    python train_gpt2_mini.py --disable_attention 1 --steps 1500
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class GPTConfig:
    """GPT-2 mini のハイパーパラメータ"""

    # Model architecture
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    vocab_size: int = 256  # Character-level (ASCII + 拡張)
    dropout: float = 0.1

    # Experiment switches
    disable_residual: bool = False
    disable_attention: bool = False
    ln_style: str = "pre"  # "pre" or "post"

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0


# =============================================================================
# Model Components
# =============================================================================
class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention

    Q, K, V を線形変換で作り、スケールド内積で重み付け。
    causal mask で未来を見ないようにする。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V を一括で計算
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 出力プロジェクション
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: 未来を見ないための下三角行列
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Q, K, V を計算
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # att = softmax(QK^T / sqrt(d_k)) * V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Causal mask: 未来のトークンをマスク
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 重み付き和
        y = att @ v  # (B, n_head, T, head_dim)

        # Multi-head を結合: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 出力プロジェクション
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Feed-Forward Network (Position-wise)

    2層の線形変換 + GELU 活性化
    hidden_dim = 4 * n_embd（GPT-2 の標準）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Block

    Pre-LN（安定）:
        x -> LN -> Attention -> +residual -> LN -> MLP -> +residual

    Post-LN（不安定、観測用）:
        x -> Attention -> +residual -> LN -> MLP -> +residual -> LN
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

        self.ln_style = config.ln_style
        self.disable_residual = config.disable_residual
        self.disable_attention = config.disable_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ln_style == "pre":
            # Pre-LN: LN -> Sublayer -> Residual
            if not self.disable_attention:
                attn_out = self.attn(self.ln1(x))
                if self.disable_residual:
                    x = attn_out
                else:
                    x = x + attn_out

            mlp_out = self.mlp(self.ln2(x))
            if self.disable_residual:
                x = mlp_out
            else:
                x = x + mlp_out
        else:
            # Post-LN: Sublayer -> Residual -> LN
            if not self.disable_attention:
                attn_out = self.attn(x)
                if self.disable_residual:
                    x = attn_out
                else:
                    x = x + attn_out
                x = self.ln1(x)

            mlp_out = self.mlp(x)
            if self.disable_residual:
                x = mlp_out
            else:
                x = x + mlp_out
            x = self.ln2(x)

        return x


class GPT2Mini(nn.Module):
    """
    GPT-2 mini モデル

    Decoder-only Transformer:
    1) Token Embedding + Positional Embedding
    2) N回 Block を通す
    3) 最後に LayerNorm
    4) Linear（語彙次元）→ logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language Model Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tok_emb と lm_head の重み共有
        self.tok_emb.weight = self.lm_head.weight

        # 重み初期化
        self.apply(self._init_weights)

        # パラメータ数の表示
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        """重み初期化（GPT-2 スタイル）"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target token indices (for training)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar (if targets provided)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        # Token + Positional Embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.tok_emb(idx)  # (B, T, n_embd)
        pos_emb = self.pos_emb(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm + LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss calculation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        テキスト生成（自己回帰）

        Args:
            idx: (B, T) 初期トークン列
            max_new_tokens: 生成するトークン数
            temperature: サンプリング温度（高いほどランダム）
            top_k: 上位 k 個からサンプリング
        """
        for _ in range(max_new_tokens):
            # block_size を超えないようにクロップ
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 最後のトークンの logits

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Softmax -> サンプリング
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 連結
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# Data Loading
# =============================================================================
class CharDataset:
    """
    Character-level Dataset

    テキストを文字単位でトークン化し、学習データを生成。
    """

    def __init__(self, data: str, block_size: int):
        # 文字 -> インデックスのマッピング
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)

        print(f"Dataset: {len(data)} characters, {self.vocab_size} unique")

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """ランダムなバッチを取得"""
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def encode(self, s: str) -> list[int]:
        """文字列をトークン列に変換"""
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, tokens: list[int]) -> str:
        """トークン列を文字列に変換"""
        return "".join([self.itos.get(t, "?") for t in tokens])


# =============================================================================
# Training
# =============================================================================
def get_device() -> str:
    """利用可能なデバイスを取得（MPS > CUDA > CPU）"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def train(
    model: GPT2Mini,
    dataset: CharDataset,
    config: GPTConfig,
    steps: int,
    batch_size: int,
    device: str,
    log_interval: int = 100,
    sample_interval: int = 500,
):
    """学習ループ"""
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler (cosine with warmup)
    warmup_steps = min(100, steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return config.learning_rate * step / warmup_steps
        decay_ratio = (step - warmup_steps) / (steps - warmup_steps)
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    print(f"\nTraining on {device} for {steps} steps")
    print(f"Batch size: {batch_size}, Block size: {config.block_size}")
    print(f"LR: {config.learning_rate}, WD: {config.weight_decay}, Grad clip: {config.grad_clip}")
    print("-" * 60)

    start_time = time.time()

    for step in range(steps):
        # Learning rate scheduling
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        x, y = dataset.get_batch(batch_size, device)

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update
        optimizer.step()

        # Logging
        if step % log_interval == 0 or step == steps - 1:
            elapsed = time.time() - start_time
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"grad_norm {grad_norm:.4f} | lr {lr:.2e} | "
                f"time {elapsed:.1f}s"
            )

        # Sample generation
        if step > 0 and step % sample_interval == 0:
            generate_sample(model, dataset, device)

    print("-" * 60)
    print(f"Training completed in {time.time() - start_time:.1f}s")


@torch.no_grad()
def generate_sample(model: GPT2Mini, dataset: CharDataset, device: str, prompt: str = None):
    """サンプル生成"""
    model.eval()

    if prompt is None:
        # ランダムな開始点
        start_idx = torch.randint(len(dataset.data) - 10, (1,)).item()
        prompt = dataset.decode(dataset.data[start_idx : start_idx + 5].tolist())

    print(f"\n--- Sample (prompt: {repr(prompt)}) ---")

    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
    text = dataset.decode(generated[0].tolist())

    print(text)
    print("---\n")

    model.train()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="GPT-2 mini training")

    # Data
    parser.add_argument("--data", type=str, default=None, help="Path to training data (text file)")

    # Model
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=256, help="Context window size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")

    # Experiment switches
    parser.add_argument(
        "--disable_residual", type=int, default=0, help="Disable residual connections (0 or 1)"
    )
    parser.add_argument("--disable_attention", type=int, default=0, help="Disable attention (MLP-only)")
    parser.add_argument("--ln_style", type=str, default="pre", choices=["pre", "post"], help="LayerNorm style")

    # Logging
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--sample_interval", type=int, default=500, help="Generate sample every N steps")

    # Checkpoint
    parser.add_argument("--save", type=str, default="model.pt", help="Path to save checkpoint")
    parser.add_argument("--load", type=str, default=None, help="Path to load checkpoint")

    # Inference mode
    parser.add_argument("--generate", action="store_true", help="Generate only (no training)")
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    if args.data and os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded data from {args.data}")
    else:
        # サンプルデータ（data.txt がない場合）
        text = """
吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。
何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。
しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕えて煮て食うという話である。
しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
""".strip()
        print("Using sample data (provide --data for custom training)")

    # block_size をデータ長に合わせて調整
    actual_block_size = min(args.block_size, len(text) - 1)
    if actual_block_size < args.block_size:
        print(f"Note: block_size adjusted from {args.block_size} to {actual_block_size} (data length: {len(text)})")

    # Create dataset
    dataset = CharDataset(text, actual_block_size)

    # Config
    config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=actual_block_size,
        vocab_size=dataset.vocab_size,
        dropout=args.dropout,
        disable_residual=bool(args.disable_residual),
        disable_attention=bool(args.disable_attention),
        ln_style=args.ln_style,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    print(f"\nModel config:")
    print(f"  n_layer: {config.n_layer}, n_head: {config.n_head}, n_embd: {config.n_embd}")
    print(f"  block_size: {config.block_size}, vocab_size: {config.vocab_size}")
    print(f"  disable_residual: {config.disable_residual}")
    print(f"  disable_attention: {config.disable_attention}")
    print(f"  ln_style: {config.ln_style}")

    # Create model
    model = GPT2Mini(config).to(device)

    # Load checkpoint if specified
    if args.load and os.path.exists(args.load):
        checkpoint = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        # Restore vocab mapping
        dataset.stoi = checkpoint["stoi"]
        dataset.itos = checkpoint["itos"]
        print(f"Loaded checkpoint from {args.load}")

    # Mode selection
    if args.interactive:
        # Interactive mode
        interactive_generate(model, dataset, device, args.temperature, args.max_tokens)
    elif args.generate:
        # Generate only
        prompt = args.prompt if args.prompt else "吾輩は"
        print(f"\n=== Generation (prompt: {repr(prompt)}) ===")
        idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
        model.eval()
        generated = model.generate(idx, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=40)
        print(dataset.decode(generated[0].tolist()))
    else:
        # Train mode
        train(
            model=model,
            dataset=dataset,
            config=config,
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
            log_interval=args.log_interval,
            sample_interval=args.sample_interval,
        )

        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        }
        torch.save(checkpoint, args.save)
        print(f"\nCheckpoint saved to {args.save}")

        # Final generation
        print("\n=== Final Generation ===")
        generate_sample(model, dataset, device, prompt="吾輩は")

        print(f"\nTo generate more text:")
        print(f"  python train_gpt2_mini.py --load {args.save} --generate --prompt '好きなテキスト'")
        print(f"  python train_gpt2_mini.py --load {args.save} --interactive")


def interactive_generate(model: GPT2Mini, dataset: CharDataset, device: str, temperature: float, max_tokens: int):
    """インタラクティブ生成モード"""
    model.eval()
    print("\n=== Interactive Mode ===")
    print("Enter a prompt and press Enter to generate. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Bye!")
                break
            if not prompt:
                continue

            idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
            generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=40)
            print("\n" + dataset.decode(generated[0].tolist()) + "\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
