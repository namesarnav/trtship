"""A small BERT-style encoder classifier written in plain PyTorch (no ``transformers`` needed).

It has the interface features that make transformer deployment interesting: three integer inputs
(token ids, attention mask, segment ids), a dynamic batch axis *and* a dynamic sequence axis, and
attention written as explicit matmuls so the exported graph is ordinary ONNX. Weights are
deterministic random values unless a checkpoint is supplied.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, intermediate), nn.GELU(), nn.Linear(intermediate, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each: batch, heads, seq, head_dim
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) + additive_mask
        context = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, seq, hidden)
        x = self.norm1(x + self.out(context))
        return self.norm2(x + self.ffn(x))  # type: ignore[no-any-return]


class BertStyleClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
        intermediate: int = 512,
        max_positions: int = 512,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, hidden)
        self.positions = nn.Embedding(max_positions, hidden)
        self.segments = nn.Embedding(2, hidden)
        self.embed_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(_Layer(hidden, heads, intermediate) for _ in range(layers))
        self.pooler = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        seq = input_ids.shape[1]
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        x = self.tokens(input_ids) + self.positions(positions) + self.segments(token_type_ids)
        x = self.embed_norm(x)
        # 0 where attended, a large negative number where masked; broadcast over heads and queries.
        additive = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * -1e4
        for layer in self.layers:
            x = layer(x, additive)
        logits: torch.Tensor = self.classifier(torch.tanh(self.pooler(x[:, 0])))
        return logits


def build(
    seed: int = 0,
    vocab_size: int = 1000,
    hidden: int = 128,
    layers: int = 2,
    heads: int = 4,
    intermediate: int = 512,
    max_positions: int = 512,
    num_labels: int = 2,
) -> nn.Module:
    torch.manual_seed(seed)
    return BertStyleClassifier(
        vocab_size, hidden, layers, heads, intermediate, max_positions, num_labels
    )
