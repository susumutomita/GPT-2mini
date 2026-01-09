#!/usr/bin/env python3
"""
BPE (Byte Pair Encoding) トークナイザーの実装

Character-level との違いを理解するための教育用実装。

BPE のアルゴリズム:
1. 初期語彙 = 全てのユニークな文字（またはバイト）
2. テキスト中で最も頻出する連続ペアを見つける
3. そのペアを新しいトークンとしてマージ
4. 目標の語彙サイズになるまで 2-3 を繰り返す
"""

import re
import sys
from collections import Counter
from typing import Dict, List, Tuple


class BPETokenizer:
    """
    Byte Pair Encoding トークナイザー

    Usage:
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=1000)
        tokens = tokenizer.encode("吾輩は猫である")
        text = tokenizer.decode(tokens)
    """

    def __init__(self):
        self.vocab: Dict[int, bytes] = {}  # id -> bytes
        self.merges: Dict[Tuple[int, int], int] = {}  # (id1, id2) -> new_id
        self.vocab_size = 256  # 初期は全バイト

    def _get_stats(self, ids: List[int]) -> Counter:
        """連続するペアの出現頻度をカウント"""
        pairs = Counter()
        for i in range(len(ids) - 1):
            pairs[(ids[i], ids[i + 1])] += 1
        return pairs

    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """指定したペアを新しい ID にマージ"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """
        BPE モデルを学習

        Args:
            text: 学習テキスト
            vocab_size: 目標語彙サイズ（256 以上）
            verbose: 学習過程を表示
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 (base bytes)"

        # 初期化: 全バイトを語彙に追加
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        # テキストをバイト列に変換し、ID リストに
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        if verbose:
            print(f"学習開始: {len(text)} 文字 → {len(ids)} バイト")
            print(f"目標語彙サイズ: {vocab_size}")
            print("-" * 50)
            sys.stdout.flush()

        num_merges = vocab_size - 256
        for i in range(num_merges):
            # 最頻出ペアを見つける
            stats = self._get_stats(ids)
            if not stats:
                break

            top_pair = stats.most_common(1)[0][0]
            freq = stats[top_pair]

            # 頻度が 1 以下なら終了
            if freq <= 1:
                if verbose:
                    print(f"これ以上マージするペアがありません（{i} 回でストップ）")
                break

            # 新しいトークンを作成
            new_id = 256 + i
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            # マージを適用
            ids = self._merge(ids, top_pair, new_id)

            if verbose and (i + 1) % 50 == 0:
                token_str = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"マージ {i+1}/{num_merges}: '{token_str}' (頻度: {freq}, 残り: {len(ids)} トークン)")
                sys.stdout.flush()

        self.vocab_size = len(self.vocab)
        if verbose:
            print("-" * 50)
            print(f"学習完了: 語彙サイズ = {self.vocab_size}")
            # 圧縮率を計算
            original_len = len(text.encode("utf-8"))
            compressed_len = len(ids)
            print(f"圧縮率: {original_len} → {compressed_len} ({compressed_len/original_len*100:.1f}%)")
            sys.stdout.flush()

    def encode(self, text: str) -> List[int]:
        """テキストをトークン ID リストに変換"""
        ids = list(text.encode("utf-8"))

        # 学習したマージを順番に適用
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # マージ可能なペアの中で、最も早く学習したものを適用
            pair = min(
                stats.keys(),
                key=lambda p: self.merges.get(p, float("inf"))
            )
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = self._merge(ids, pair, new_id)

        return ids

    def decode(self, ids: List[int]) -> str:
        """トークン ID リストをテキストに変換"""
        text_bytes = b"".join(self.vocab[id] for id in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def get_vocab_examples(self, n: int = 20) -> List[Tuple[int, str]]:
        """語彙の例を表示（マージで作られたトークン）"""
        examples = []
        for id in range(256, min(256 + n, self.vocab_size)):
            token_bytes = self.vocab[id]
            token_str = token_bytes.decode("utf-8", errors="replace")
            examples.append((id, token_str))
        return examples


def main():
    """BPE の動作デモ"""
    # テストテキスト
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 60)
    print("BPE トークナイザーのデモ")
    print("=" * 60)
    print()

    # BPE を学習
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=2000, verbose=True)

    print()
    print("=== マージで作られたトークンの例 ===")
    for id, token in tokenizer.get_vocab_examples(30):
        print(f"  {id}: '{token}'")

    print()
    print("=== エンコード・デコードのテスト ===")
    test_texts = [
        "吾輩は猫である",
        "私の名前は田中です",
        "Hello World",
        "龍",  # Character-level では扱えなかった
        "😀",  # 絵文字
    ]

    for t in test_texts:
        ids = tokenizer.encode(t)
        decoded = tokenizer.decode(ids)
        print(f"  '{t}' → {len(ids)} トークン → '{decoded}'")


if __name__ == "__main__":
    main()
