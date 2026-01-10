#!/usr/bin/env python3
"""
スケーリング則観測実験

複数のモデルサイズで学習を実行し、パラメータ数 vs Loss の関係を記録する。
結果は JSON に保存され、plot_scaling.py で可視化できる。

Usage:
    python run_scaling_experiment.py --data data.txt
    python run_scaling_experiment.py --data data.txt --quick  # 短縮版
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ModelConfig:
    """モデル設定"""

    name: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 256
    dropout: float = 0.0  # スケーリング実験では dropout=0 が標準


# スケーリング実験用のモデルサイズ定義
SCALING_CONFIGS = [
    ModelConfig(name="tiny", n_layer=2, n_head=2, n_embd=64),
    ModelConfig(name="small", n_layer=4, n_head=4, n_embd=128),
    ModelConfig(name="medium", n_layer=6, n_head=6, n_embd=192),
    ModelConfig(name="base", n_layer=6, n_head=8, n_embd=256),
    ModelConfig(name="large", n_layer=8, n_head=8, n_embd=384),
]


# =============================================================================
# Model (train_gpt2_mini.py から流用)
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [Block(config.n_embd, config.n_head, config.block_size, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Dataset
# =============================================================================
class CharDataset:
    def __init__(self, data: str, block_size: int):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)


# =============================================================================
# Training
# =============================================================================
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_model(
    model: GPT,
    dataset: CharDataset,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    log_interval: int = 100,
) -> dict:
    """モデルを学習し、結果を返す"""
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # Cosine LR scheduler with warmup
    warmup_steps = min(100, steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, steps - warmup_steps)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    losses = []
    start_time = time.time()

    for step in range(steps):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = dataset.get_batch(batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0 or step == steps - 1:
            losses.append({"step": step, "loss": loss.item()})

    elapsed = time.time() - start_time
    final_loss = losses[-1]["loss"]

    return {
        "final_loss": final_loss,
        "losses": losses,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (steps * batch_size * dataset.block_size) / elapsed,
    }


def compute_validation_loss(model: GPT, dataset: CharDataset, device: str, num_batches: int = 50) -> float:
    """検証用 loss を計算（平均）"""
    model.train(False)
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(32, device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train(True)
    return total_loss / num_batches


# =============================================================================
# Main Experiment
# =============================================================================
def run_experiment(
    data_path: str,
    output_path: str = "scaling_results.json",
    steps: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    configs: list[ModelConfig] = None,
):
    """スケーリング実験を実行"""
    device = get_device()
    print(f"Device: {device}")

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Data: {len(text):,} characters")

    # Use a fixed block_size for fair comparison
    block_size = 256
    dataset = CharDataset(text, block_size)
    print(f"Vocab size: {dataset.vocab_size}")

    if configs is None:
        configs = SCALING_CONFIGS

    results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training: {config.name}")
        print(f"  n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")

        # Create model
        model = GPT(config, dataset.vocab_size).to(device)
        n_params = model.count_parameters()
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        # Train
        train_result = train_model(
            model=model,
            dataset=dataset,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        # Validation (more stable than final training loss)
        val_loss = compute_validation_loss(model, dataset, device)

        result = {
            "name": config.name,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "n_params": n_params,
            "train_loss": train_result["final_loss"],
            "val_loss": val_loss,
            "elapsed_seconds": train_result["elapsed_seconds"],
            "tokens_per_second": train_result["tokens_per_second"],
            "steps": steps,
            "batch_size": batch_size,
            "block_size": block_size,
        }
        results.append(result)

        print(f"  Train Loss: {train_result['final_loss']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Time: {train_result['elapsed_seconds']:.1f}s")

        # Clean up GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "scaling_law",
                "data_path": data_path,
                "data_size": len(text),
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print("\nSummary:")
    print(f"{'Name':<10} {'Params':>12} {'Val Loss':>12}")
    print("-" * 36)
    for r in results:
        print(f"{r['name']:<10} {r['n_params']:>12,} {r['val_loss']:>12.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Scaling Law Experiment")
    parser.add_argument("--data", type=str, default="data.txt", help="Path to training data")
    parser.add_argument("--output", type=str, default="scaling_results.json", help="Output JSON path")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer steps, smaller models)")

    args = parser.parse_args()

    if args.quick:
        # Quick モード: 少ないステップ、小さいモデルのみ
        configs = [
            ModelConfig(name="tiny", n_layer=2, n_head=2, n_embd=64),
            ModelConfig(name="small", n_layer=4, n_head=4, n_embd=128),
            ModelConfig(name="medium", n_layer=6, n_head=6, n_embd=192),
        ]
        steps = 500
    else:
        configs = SCALING_CONFIGS
        steps = args.steps

    run_experiment(
        data_path=args.data,
        output_path=args.output,
        steps=steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        configs=configs,
    )


if __name__ == "__main__":
    main()
