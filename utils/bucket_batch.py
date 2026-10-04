"""Length-aware batches for the fixed-width DOS cache."""
import math

import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
from utils.crystal import CrystalGeometry


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
    if isinstance(samples[0][1], CrystalGeometry):
        return crystal_collate(samples)
    max_tokens = max(2 + int((sample[0][2:] != 0).sum().item()) for sample in samples)
    trimmed = []
    for sample in samples:
        fields = list(sample)
        fields[0] = fields[0][:max_tokens]
        fields[1] = fields[1][:max_tokens]
        trimmed.append(fields)
    return default_collate(trimmed)


def crystal_collate(samples):
    """Pad complete atom lists to this batch's maximum; cell has no atom slots."""
    if not samples or not all(isinstance(s[1], CrystalGeometry) for s in samples):
        raise ValueError("explicit crystal samples required")
    batch = [pad_sequence([s[0] for s in samples], batch_first=True),
             CrystalGeometry(torch.stack([s[1].cell for s in samples]),
                             pad_sequence([s[1].frac_coords for s in samples], batch_first=True))]
    for i in range(2, len(samples[0])):
        values = [s[i] for s in samples]
        batch.append(None if all(v is None for v in values) else default_collate(values))
    return batch


def split_crystal_batch(batch, pair_budget):
    """Split computation, preserving sample order and every atom.

    The budget counts padded atom pairs B*L², not model bytes. A single crystal
    exceeding it is rejected explicitly, never cropped. Callers accumulate
    sample-weighted losses and perform one optimizer update for the input batch.
    """
    if not isinstance(batch[1], CrystalGeometry) or pair_budget <= 0:
        raise ValueError("explicit geometry and positive pair budget required")
    counts = batch[0].ne(0).sum(-1)
    chunks, start, maximum = [], 0, 0
    for i, count in enumerate(counts.tolist()):
        if count <= 0 or count * count > pair_budget:
            raise ValueError(f"complete crystal with {count} atoms exceeds pair budget {pair_budget}")
        maximum = max(maximum, count)
        if (i - start + 1) * maximum * maximum > pair_budget:
            chunks.append((start, i))
            start, maximum = i, count
    chunks.append((start, len(counts)))
    result = []
    for start, end in chunks:
        length = int(counts[start:end].max())
        fields = [batch[0][start:end, :length],
                  CrystalGeometry(batch[1].cell[start:end], batch[1].frac_coords[start:end, :length])]
        fields.extend(value[start:end] if torch.is_tensor(value) else value for value in batch[2:])
        result.append(fields)
    return result


def iter_computation_batches(loader, pair_budget=None):
    """Bound evaluation tensor sizes, retaining loader sample order."""
    for batch in loader:
        if isinstance(batch[1], CrystalGeometry):
            budget = pair_budget if pair_budget is not None else automatic_pair_budget(batch)
            yield from split_crystal_batch(batch, budget)
        else:
            yield batch


def automatic_pair_budget(batch):
    """Keep the historical pair-volume target and always fit one full crystal."""
    maximum = int(batch[0].ne(0).sum(-1).max())
    return max(len(batch[0]) * 80 ** 2, maximum ** 2)
