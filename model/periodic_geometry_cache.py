"""Bounded structure-only cache for periodic local messages.

Keys use exact lattice and valid fractional-coordinate contents, so no dataset
IDs or changes to the crystal input contract are needed. Parameter-dependent
features are never stored. Entries are process-local and absent from state_dict.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from model.periodic_messages import (
    _MessageGeometry, _record_context, prepare_shared_message_geometry,
)
from utils.crystal import CrystalGeometry
from utils.periodic_geometry import (
    PeriodicNeighborRecords, build_periodic_records, validate_periodic_inputs,
)

GIB = 1024 ** 3
MAX_CACHE_BYTES = 16 * GIB


def geometry_cache_budget(device):
    device = torch.device(device)
    if device.type != "cuda":
        return 0
    return min(MAX_CACHE_BYTES, torch.cuda.get_device_properties(device).total_memory // 2)


def _tensor_bytes(records, geometry):
    seen, total = set(), 0
    for value in (*vars(records).values(), *vars(geometry).values()):
        values = value if isinstance(value, tuple) and value and torch.is_tensor(value[0]) else (value,)
        for tensor in values:
            if not torch.is_tensor(tensor):
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


@dataclass(frozen=True)
class _StructureEntry:
    records: PeriodicNeighborRecords
    geometry: _MessageGeometry
    n_atoms: int
    nbytes: int


class PeriodicGeometryCache:
    """Reuse immutable crystal geometry across batches, with LRU eviction.

    An explicit budget permits small CPU tests. The production default is CUDA
    only, capped at 16 GiB and half of device capacity. A conservative workspace
    estimate can evict additional entries before the learned forward/backward.
    """

    def __init__(self, route, device, *, max_bytes=None, local_blocks=4):
        if route not in ("A", "B+"):
            raise ValueError("geometry cache requires A or B+")
        self.route = route
        self.max_bytes = geometry_cache_budget(device) if max_bytes is None else max_bytes
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes < 0:
            raise ValueError("cache budget must be a nonnegative integer")
        if isinstance(local_blocks, bool) or not isinstance(local_blocks, int) or local_blocks < 1:
            raise ValueError("positive local block count required")
        self.local_blocks = local_blocks
        self._entries = OrderedDict()
        self._context = None
        self.nbytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = self.oom_fallbacks = 0
        self._budget_policy = "explicit" if max_bytes is not None else "min(16GiB,half_device)"

    def clear(self):
        self._entries.clear()
        self._context = None
        self.nbytes = 0

    def configuration(self):
        return {"enabled": True, "key": "cell-frac-content-v1", "population": "lazy",
                "max_bytes": self.max_bytes, "budget_policy": self._budget_policy,
                "workspace_guard": "local-pairs-v1"}

    def statistics(self):
        return {"entries": len(self._entries), "tensor_bytes": self.nbytes,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "bypasses": self.bypasses, "oom_fallbacks": self.oom_fallbacks}

    def _evict(self):
        _, entry = self._entries.popitem(last=False)
        self.nbytes -= entry.nbytes
        self.evictions += 1

    def _reserve_workspace(self, device, required):
        if device.type != "cuda":
            return
        while self._entries:
            free, _ = torch.cuda.mem_get_info(device)
            reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            if free + reusable >= required:
                break
            self._evict()

    def _make_entry(self, cell, coordinates, radius):
        with torch.inference_mode(False), torch.no_grad():
            # Entries may be populated during inference, then used in training.
            # Saved linear inputs must be normal tensors in both cases.
            cell = cell.clone() if cell.is_inference() else cell
            coordinates = coordinates.clone() if coordinates.is_inference() else coordinates
            geometry = CrystalGeometry(cell[None], coordinates[None])
            mask = torch.zeros((1, len(coordinates)), dtype=torch.bool, device=cell.device)
            records = build_periodic_records(geometry, mask, radius)
            # Pair receivers depend on the assembled batch's padded width; do
            # not retain an extra int64 tensor for every pair in the dataset.
            prepared = prepare_shared_message_geometry(records, self.route, pair_receivers=False)
        return _StructureEntry(records, prepared, len(coordinates), _tensor_bytes(records, prepared))

    def get(self, geometry, padding_mask, *, radius):
        """Return aligned physical records and shared features, or normal-path None."""
        if (not isinstance(geometry, CrystalGeometry) or self.max_bytes == 0 or
                geometry.cell.requires_grad or geometry.frac_coords.requires_grad):
            self.bypasses += 1
            return None
        validate_periodic_inputs(geometry, padding_mask)
        context = (geometry.device, geometry.dtype, float(radius))
        if context != self._context:
            self.clear()
            self._context = context
        mask_cpu = padding_mask.detach().cpu()
        if len(mask_cpu) == 0 or mask_cpu.all(-1).any():
            self.bypasses += 1
            return None
        # One small transfer per batch, rather than synchronizing per crystal.
        cell_cpu = geometry.cell.detach().cpu()
        coordinates_cpu = geometry.frac_coords.detach().cpu()
        entries, slots = [], []
        try:
            for b in range(len(cell_cpu)):
                valid = (~mask_cpu[b]).nonzero(as_tuple=True)[0]
                prefix = torch.equal(valid, torch.arange(len(valid)))
                slot = None if prefix else valid.to(device=geometry.device)
                coordinates = coordinates_cpu[b, valid]
                key = (cell_cpu[b].numpy().tobytes(), coordinates.numpy().tobytes())
                entry = self._entries.get(key)
                if entry is None:
                    self.misses += 1
                    self._reserve_workspace(geometry.device, 2 * GIB)
                    coordinates_gpu = (geometry.frac_coords[b, :len(valid)] if prefix
                                       else geometry.frac_coords[b, slot])
                    entry = self._make_entry(geometry.cell[b], coordinates_gpu, radius)
                    if entry.nbytes <= self.max_bytes:
                        while self.nbytes + entry.nbytes > self.max_bytes:
                            self._evict()
                        self._entries[key] = entry
                        self.nbytes += entry.nbytes
                else:
                    self.hits += 1
                    self._entries.move_to_end(key)
                entries.append(entry)
                slots.append(slot)
            result = self._assemble(entries, slots, padding_mask, radius)
        except torch.cuda.OutOfMemoryError:
            # Cache allocation failure has no learned computation or RNG effect.
            # Free the optional cache, then let the ordinary path handle the batch.
            entries.clear()
            self.clear()
            self.oom_fallbacks += 1
            torch.cuda.empty_cache()
            return None
        del entries, entry
        records, prepared = result
        size = records.vectors.element_size()
        # Resource guard, not a claim that all batches fit. An intrinsically
        # oversized uncached batch still raises instead of changing its math.
        work = 4 * GIB * size // 4
        work += (len(prepared.pairs[0]) * 1024 * size * self.local_blocks
                 if prepared.pairs is not None else len(records.distances) * 2048 * size)
        self._reserve_workspace(geometry.device, work)
        return result

    def _assemble(self, entries, slots, padding_mask, radius):
        _, width = padding_mask.shape
        values = {name: torch.cat([getattr(e.records, name) for e in entries])
                  for name in ("shifts", "vectors", "distances")}
        values.update({name: torch.cat([getattr(e.records, name) if slot is None
                                       else slot[getattr(e.records, name)] for e, slot in zip(entries, slots)])
                       for name in ("dst", "src")})
        values["batch"] = torch.cat([e.records.batch + b for b, e in enumerate(entries)])
        records = PeriodicNeighborRecords(**values, radius=radius, padding_mask=padding_mask)
        gs = [e.geometry for e in entries]
        offsets, offset = [], 0
        for entry in entries:
            offsets.append(offset)
            offset += len(entry.records.distances)

        def expand(field):
            rows = []
            for g, entry, slot in zip(gs, entries, slots):
                values = getattr(g, field)
                if slot is None:
                    rows.append(F.pad(values, (0, width-entry.n_atoms)))
                else:
                    rows.append(values.new_zeros((1, width)).index_copy(1, slot, values))
            return torch.cat(rows)

        degree = expand("degree")
        effective = expand("effective_count")
        receiver = records.batch * width + records.dst
        sender = records.batch * width + records.src
        pairs = (tuple(torch.cat([g.pairs[i] + off for g, off in zip(gs, offsets)]) for i in range(2))
                 if self.route == "A" else None)
        prepared = _MessageGeometry(
            torch.cat([g.unit for g in gs]), torch.cat([g.radial for g in gs]),
            torch.cat([g.cutoff for g in gs]), degree, effective, degree*(degree-1)//2,
            receiver, sender, torch.cat([g.order + off for g, off in zip(gs, offsets)]), pairs,
            record_context=_record_context(records), prepared_route=self.route)
        if pairs is not None:
            return records, replace(prepared, angular_basis=torch.cat([g.angular_basis for g in gs]),
                                    pair_cutoff=torch.cat([g.pair_cutoff for g in gs]),
                                    pair_receiver=receiver[pairs[0]])
        return records, replace(prepared, harmonics=torch.cat([g.harmonics for g in gs]))
