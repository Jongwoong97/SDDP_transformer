from typing import Sequence, Iterator
import torch

from torch.utils.data import Sampler


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially from a given list of indices.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)