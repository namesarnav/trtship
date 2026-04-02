"""Factories referenced from test configs as ``trtship_fixtures.models:<name>``."""

from __future__ import annotations

import torch
from torch import nn


def tiny_mlp(seed: int = 0, in_features: int = 16, out_features: int = 4) -> nn.Module:
    """[batch, 16] float32 -> [batch, 4] float32."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, out_features))


class _TokenClassifier(nn.Module):
    """BERT-shaped I/O: two integer inputs with a dynamic sequence length, two outputs."""

    def __init__(self, vocab: int = 100, hidden: int = 8, classes: int = 3) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed(input_ids) * attention_mask.unsqueeze(-1).to(torch.float32)
        return self.head(hidden.mean(dim=1)), hidden


def token_classifier(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return _TokenClassifier()


class _DictOut(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return {"doubled": x * 2, "unused": None, "total": x.sum(dim=1, keepdim=True)}


def dict_output() -> nn.Module:
    return _DictOut()


class _FlattenBatchSeq(nn.Module):
    """Output dim 0 is batch*seq, which no single symbol explains."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-1])


def flatten_batch_seq() -> nn.Module:
    return _FlattenBatchSeq()


class _Scalar(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


def scalar_output() -> nn.Module:
    return _Scalar()


class _NonTensor(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        return x, 3


def non_tensor_output() -> nn.Module:
    return _NonTensor()


class _Rank(nn.Module):
    """Output rank depends on input size: unsupported."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.shape[0] % 2 == 0 else x.unsqueeze(0)


def rank_shifting() -> nn.Module:
    return _Rank()


class _Fp64(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float64)


def float64_output() -> nn.Module:
    return _Fp64()


class _Explodes(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("shape mismatch in layer 3")


def exploding_forward() -> nn.Module:
    return _Explodes()


class _Fft(nn.Module):
    """Uses an operator with no ONNX mapping (aten::fft_rfft)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude: torch.Tensor = torch.fft.rfft(x).abs()
        return magnitude


def unsupported_op() -> nn.Module:
    return _Fft()


class _BakedBatch(nn.Module):
    """Tracing freezes ``int(x.shape[0])`` into the graph, so the exported model is only correct
    at the traced batch size even though its declared shapes say the batch is dynamic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = int(x.shape[0])
        return x.reshape(n, -1).sum(dim=1, keepdim=True) * torch.ones(n, 1)


def baked_batch() -> nn.Module:
    return _BakedBatch()


class _DataDependent(nn.Module):
    """Data-dependent Python control flow: tracing emits a TracerWarning."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2
        return x


def data_dependent_branch() -> nn.Module:
    return _DataDependent()


def returns_not_a_module() -> object:
    return "definitely not a module"


def raises_on_build() -> nn.Module:
    raise ValueError("bad hyperparameters")


not_callable = 42
