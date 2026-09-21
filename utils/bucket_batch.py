"""Length-aware batches for the fixed-width DOS cache."""
import math

import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import default_collate


class LengthBucketBatchSampler(Sampler):
    """Stable-sort bounded sampler windows by valid atom count."""

    def __init__(self, sampler, lengths, batch_size, window_batches=20):
        if batch_size <= 0 or window_batches <= 0:
            raise ValueError("batch_size and window_batches must be positive")
        self.sampler = sampler
        self.lengths = torch.as_tensor(lengths, dtype=torch.long).cpu()
        self.batch_size = int(batch_size)
        self.window_size = self.batch_size * int(window_batches)

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        indices = list(iter(self.sampler))
        for start in range(0, len(indices), self.window_size):
            window = indices[start:start + self.window_size]
            window.sort(key=lambda index: int(self.lengths[index]))
            for batch_start in range(0, len(window), self.batch_size):
                yield window[batch_start:batch_start + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.sampler) / self.batch_size)


def trim_atom_padding_collate(samples):
    """Trim only terminal atom padding from elements and positions."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_tokens = max(2 + int((sample[0][2:] != 0).sum().item()) for sample in samples)
    trimmed = []
    for sample in samples:
        fields = list(sample)
        fields[0] = fields[0][:max_tokens]
        fields[1] = fields[1][:max_tokens]
        trimmed.append(fields)
    return default_collate(trimmed)
