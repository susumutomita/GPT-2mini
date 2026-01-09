#!/usr/bin/env python3
"""
GPT-2 mini with BPE — Character-level との比較用

Usage:
    # BPE で学習
    python train_gpt2_bpe.py --data data.txt --vocab_size 2000 --steps 3000

    # 生成
    python train_gpt2_bpe.py --load model_bpe.pt --generate --prompt '吾輩は'
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer


@dataclass
class GPTConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    vocab_size: int = 256
    dropout: float = 0.1
    disable_residual: bool = False
    disable_attention: bool = False
    ln_style: str = "pre"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
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
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_style = config.ln_style
        self.disable_residual = config.disable_residual
        self.disable_attention = config.disable_attention

    def forward(self, x):
        if self.ln_style == "pre":
            if not self.disable_attention:
                attn_out = self.attn(self.ln1(x))
                x = x + attn_out if not self.disable_residual else attn_out
            mlp_out = self.mlp(self.ln2(x))
            x = x + mlp_out if not self.disable_residual else mlp_out
        else:
            if not self.disable_attention:
                attn_out = self.attn(x)
                x = (x + attn_out if not self.disable_residual else attn_out)
                x = self.ln1(x)
            mlp_out = self.mlp(x)
            x = (x + mlp_out if not self.disable_residual else mlp_out)
            x = self.ln2(x)
        return x


class GPT2Mini(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")

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

    def forward(self, idx, targets=None):
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class BPEDataset:
    def __init__(self, text: str, block_size: int, vocab_size: int = 2000):
        self.block_size = block_size
        print("BPE トークナイザーを学習中...")
        self.tokenizer = BPETokenizer()
        self.tokenizer.train(text, vocab_size=vocab_size, verbose=True)
        self.vocab_size = self.tokenizer.vocab_size
        print("テキストをトークン化中...")
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        print(f"Dataset: {len(text)} 文字 → {len(self.data)} トークン")

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def encode(self, s: str):
        return self.tokenizer.encode(s)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(model, dataset, config, steps, batch_size, device, log_interval=100, sample_interval=500):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_steps = min(100, steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return config.learning_rate * step / warmup_steps
        decay_ratio = (step - warmup_steps) / (steps - warmup_steps)
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    print(f"\nTraining on {device} for {steps} steps")
    print(f"Batch size: {batch_size}, Block size: {config.block_size}")
    print("-" * 60)

    start_time = time.time()
    for step in range(steps):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        x, y = dataset.get_batch(batch_size, device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % log_interval == 0 or step == steps - 1:
            elapsed = time.time() - start_time
            print(f"step {step:5d} | loss {loss.item():.4f} | grad_norm {grad_norm:.4f} | lr {lr:.2e} | time {elapsed:.1f}s")

        if step > 0 and step % sample_interval == 0:
            generate_sample(model, dataset, device)

    print("-" * 60)
    print(f"Training completed in {time.time() - start_time:.1f}s")


@torch.no_grad()
def generate_sample(model, dataset, device, prompt=None):
    model.eval()
    if prompt is None:
        prompt = "吾輩は"
    print(f"\n--- Sample (prompt: '{prompt}') ---")
    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
    text = dataset.decode(generated[0].tolist())
    print(text)
    print("---\n")
    model.train()


def main():
    parser = argparse.ArgumentParser(description="GPT-2 mini with BPE")
    parser.add_argument("--data", type=str, default="data.txt")
    parser.add_argument("--vocab_size", type=int, default=2000, help="BPE vocabulary size")
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--save", type=str, default="model_bpe.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", type=str, default="吾輩は")
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text)} characters from {args.data}")

    dataset = BPEDataset(text, args.block_size, vocab_size=args.vocab_size)

    config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        vocab_size=dataset.vocab_size,
        learning_rate=args.lr,
    )
    print(f"\nModel config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
    print(f"BPE vocab_size={config.vocab_size}")

    model = GPT2Mini(config).to(device)

    if args.load and os.path.exists(args.load):
        checkpoint = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        dataset.tokenizer = checkpoint["tokenizer"]
        print(f"Loaded checkpoint from {args.load}")

    if args.generate:
        print(f"\n=== Generation (prompt: '{args.prompt}') ===")
        idx = torch.tensor([dataset.encode(args.prompt)], dtype=torch.long, device=device)
        model.eval()
        generated = model.generate(idx, max_new_tokens=args.max_tokens, temperature=0.8, top_k=40)
        print(dataset.decode(generated[0].tolist()))
    else:
        train(model, dataset, config, args.steps, args.batch_size, device, args.log_interval, args.sample_interval)
        checkpoint = {"model": model.state_dict(), "config": config, "tokenizer": dataset.tokenizer}
        torch.save(checkpoint, args.save)
        print(f"\nCheckpoint saved to {args.save}")
        generate_sample(model, dataset, device, args.prompt)


if __name__ == "__main__":
    main()
