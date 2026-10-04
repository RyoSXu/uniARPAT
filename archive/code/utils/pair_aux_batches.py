"""Deterministic, label-free pair plans for the eDOS pair auxiliary loss."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from hashlib import sha256
from itertools import combinations
from math import gcd
import json
from typing import Iterable, Sequence

import numpy as np


CompositionKey = tuple[tuple[int, int], ...]


@dataclass(frozen=True, order=True)
class PairRecord:
    """One legal pair whose endpoints have identical absolute composition."""

    reduced_key: CompositionKey
    absolute_key: CompositionKey
    index_a: int
    index_b: int
    mpid_a: str
    mpid_b: str


def composition_keys(elements_row: Sequence[int], sentinel_count: int = 2):
    """Return absolute and reduced composition keys for one padded element row."""
    row = np.asarray(elements_row)
    if row.ndim != 1:
        raise ValueError("elements_row must be one-dimensional")
    if sentinel_count < 0 or sentinel_count >= row.size:
        raise ValueError("sentinel_count must leave at least one element slot")
    atoms = row[sentinel_count:]
    atoms = atoms[atoms != 0]
    if atoms.size == 0:
        raise ValueError("composition contains no atoms after sentinels")
    atomic_numbers, counts = np.unique(atoms, return_counts=True)
    absolute = tuple(
        (int(atomic_number), int(count))
        for atomic_number, count in zip(atomic_numbers, counts)
    )
    divisor = reduce(gcd, (count for _, count in absolute))
    reduced = tuple((atomic_number, count // divisor) for atomic_number, count in absolute)
    return absolute, reduced


def build_pair_universe(elements, sample_ids: Sequence[str]) -> tuple[PairRecord, ...]:
    """Enumerate all within-absolute-composition pairs without reading labels."""
    rows = np.asarray(elements)
    ids = [str(sample_id) for sample_id in sample_ids]
    if rows.ndim != 2:
        raise ValueError("elements must be a two-dimensional array")
    if len(rows) != len(ids):
        raise ValueError("elements and sample_ids must have the same length")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")

    absolute_groups: dict[CompositionKey, list[tuple[int, str, CompositionKey]]] = defaultdict(list)
    for index, (row, sample_id) in enumerate(zip(rows, ids)):
        absolute, reduced = composition_keys(row)
        absolute_groups[absolute].append((index, sample_id, reduced))

    pairs = []
    for absolute, members in absolute_groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda member: (member[1], member[0]))
        for left, right in combinations(members, 2):
            index_a, mpid_a, reduced_a = left
            index_b, mpid_b, reduced_b = right
            if reduced_a != reduced_b:
                raise AssertionError("one absolute composition mapped to multiple reduced keys")
            pairs.append(PairRecord(
                reduced_key=reduced_a,
                absolute_key=absolute,
                index_a=index_a,
                index_b=index_b,
                mpid_a=mpid_a,
                mpid_b=mpid_b,
            ))
    return tuple(sorted(pairs))


def _pair_order_digest(seed: int, reduced_key: CompositionKey, pair: PairRecord) -> str:
    payload = {
        "seed": int(seed),
        "reduced_key": reduced_key,
        "mpid_a": pair.mpid_a,
        "mpid_b": pair.mpid_b,
    }
    return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def select_epoch_pairs(
    universe: Iterable[PairRecord], seed: int, epoch: int
) -> tuple[PairRecord, ...]:
    """Select exactly one deterministic, cyclically rotated pair per reduced group."""
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    groups: dict[CompositionKey, list[PairRecord]] = defaultdict(list)
    for pair in universe:
        groups[pair.reduced_key].append(pair)

    selected = []
    for reduced_key in sorted(groups):
        ordered = sorted(
            groups[reduced_key],
            key=lambda pair: (_pair_order_digest(seed, reduced_key, pair), pair),
        )
        selected.append(ordered[epoch % len(ordered)])
    return tuple(selected)


def pair_plan_hash(pairs: Iterable[PairRecord], seed: int, epoch: int) -> str:
    """Hash the complete ordered plan and its selection coordinates."""
    records = [
        {
            "reduced_key": pair.reduced_key,
            "absolute_key": pair.absolute_key,
            "index_a": pair.index_a,
            "index_b": pair.index_b,
            "mpid_a": pair.mpid_a,
            "mpid_b": pair.mpid_b,
        }
        for pair in pairs
    ]
    payload = {"schema": 1, "seed": int(seed), "epoch": int(epoch), "pairs": records}
    return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def pair_universe_hash(pairs: Iterable[PairRecord], seed: int) -> str:
    """Hash every legal candidate plus the frozen epoch-selection policy."""
    records = [
        {
            "reduced_key": pair.reduced_key,
            "absolute_key": pair.absolute_key,
            "index_a": pair.index_a,
            "index_b": pair.index_b,
            "mpid_a": pair.mpid_a,
            "mpid_b": pair.mpid_b,
        }
        for pair in sorted(pairs)
    ]
    payload = {
        "schema": 1,
        "policy": "sha256-pair-order-cyclic-epoch-v1",
        "seed": int(seed),
        "pairs": records,
    }
    return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def batch_pair_plan(
    pairs: Sequence[PairRecord], pairs_per_batch: int = 16
) -> tuple[tuple[PairRecord, ...], ...]:
    """Split a plan into fixed-size pair batches, preserving plan order."""
    if pairs_per_batch <= 0:
        raise ValueError("pairs_per_batch must be positive")
    return tuple(
        tuple(pairs[start:start + pairs_per_batch])
        for start in range(0, len(pairs), pairs_per_batch)
    )


def schedule_auxiliary_batches(main_steps: int, auxiliary_batches: int):
    """Place auxiliary batches evenly with at most one at any main step."""
    if main_steps < 0 or auxiliary_batches < 0:
        raise ValueError("step and batch counts must be nonnegative")
    if auxiliary_batches > main_steps:
        raise ValueError("auxiliary_batches cannot exceed main_steps")
    if auxiliary_batches == 0:
        return tuple()
    schedule = tuple(
        ((batch_index + 1) * main_steps // auxiliary_batches - 1, batch_index)
        for batch_index in range(auxiliary_batches)
    )
    if len({step for step, _ in schedule}) != auxiliary_batches:
        raise AssertionError("auxiliary schedule assigned more than one batch to a main step")
    return schedule


class PairPlanBatchSampler:
    """Epoch-aware sampler that emits A endpoints followed by B endpoints."""

    def __init__(self, universe: Sequence[PairRecord], seed: int, pairs_per_batch: int = 16):
        if pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        self.universe = tuple(universe)
        self.seed = int(seed)
        self.pairs_per_batch = int(pairs_per_batch)
        self.epoch = 0
        self._group_count = len({pair.reduced_key for pair in self.universe})

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def epoch_pairs(self, epoch: int | None = None) -> tuple[PairRecord, ...]:
        selected_epoch = self.epoch if epoch is None else int(epoch)
        return select_epoch_pairs(self.universe, self.seed, selected_epoch)

    def plan_hash(self, epoch: int | None = None) -> str:
        selected_epoch = self.epoch if epoch is None else int(epoch)
        return pair_plan_hash(self.epoch_pairs(selected_epoch), self.seed, selected_epoch)

    def frozen_plan_hash(self) -> str:
        return pair_universe_hash(self.universe, self.seed)

    def __iter__(self):
        for batch in batch_pair_plan(self.epoch_pairs(), self.pairs_per_batch):
            yield [pair.index_a for pair in batch] + [pair.index_b for pair in batch]

    def __len__(self) -> int:
        return (self._group_count + self.pairs_per_batch - 1) // self.pairs_per_batch
