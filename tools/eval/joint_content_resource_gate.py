#!/usr/bin/env python3
"""Candidate-1 joint edge content V100 resource gate (one cold step, three arms).

This tool reuses the production recipe, the isolated-subprocess pattern, the
CUDA-event timing and the VRAM accounting of ``tools/eval/g2_resource_gate.py``
and adds the candidate-1 initialization path and content modes.

Three arms -- ``control`` / ``radial`` / ``joint`` -- each start from the frozen
B7 epoch-33 checkpoint and run exactly one complete ``train_one_step`` on the
same fixed Q1 train batch 0 (``shuffle=False``, batch 32). The valid and test
splits are never constructed. Each arm runs in its own clean process, and its
RNG is reset after the B7 weights are loaded so the arms enter the first
iteration from the same stream.

The reported numbers are single cold-step measurements. They are cost evidence
for deciding whether a pilot can be requested; they are not a steady-state
claim and no automatic cost threshold is applied. A clear OOM, non-finite loss
or missing G2 gradient is a hard failure and no result files are written.

The normal entry point refuses to run unless CUDA is present on a V100.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import torch
import yaml

G2_PREFIX = "encoder.g2_msgs."
RADIAL_SUFFIXES = (
    "W_v.weight", "W_v.bias", "W_g.weight", "W_g.bias",
    "W_o.weight", "W_o.bias", "alpha", "rbf_centers", "rbf_width",
)
JOINT_SUFFIXES = (
    "W_i.weight", "W_i.bias", "phi1.weight", "phi1.bias",
    "phi2.weight", "phi2.bias",
)
ARMS = ("control", "radial", "joint")

B7_CKPT_DEFAULT = "./output/ablation_m1_e9ctl/checkpoint_best.pth"
B7_CONFIG_DEFAULT = "./output/ablation_m1_e9ctl/config_used.yaml"
B7_CKPT_SHA256 = "cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40"
B7_EXPECTED_EPOCH = 33
B7_EXPECTED_MODEL = "M1"
B7_EXPECTED_SEED = 42
B7_EXPECTED_KEYS = 205
DEFAULT_RESULTS_JSON = "./results/joint_content_resource_gate_v100.json"
LOSS_KEYS = ("loss", "loss_edos", "loss_phdos", "loss_eta")


# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable, no GPU, no data)
# ---------------------------------------------------------------------------

def file_sha256(path, chunk_size=1 << 20):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_g2_keys(num_encoder_layers, mode):
    """G2 parameter/buffer keys the given arm must register, as a set."""
    if mode not in ARMS:
        raise ValueError(f"mode must be one of {ARMS}, got {mode!r}")
    if mode == "control":
        return set()
    radial = {
        f"{G2_PREFIX}{layer}.{suffix}"
        for layer in range(num_encoder_layers)
        for suffix in RADIAL_SUFFIXES
    }
    if mode == "radial":
        return radial
    joint = {
        f"{G2_PREFIX}{layer}.{suffix}"
        for layer in range(num_encoder_layers)
        for suffix in JOINT_SUFFIXES
    }
    return radial | joint


def extract_checkpoint_state(payload):
    """Return the transformer state dict from a runner checkpoint payload."""
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("init checkpoint payload must be a dict with a 'model' state dict")
    state = payload["model"]
    if not isinstance(state, dict):
        raise ValueError("init checkpoint 'model' entry must be a state dict")
    return state


def verify_b7_checkpoint(payload, sha256, *, expected_sha256=B7_CKPT_SHA256,
                         expected_epoch=B7_EXPECTED_EPOCH,
                         expected_model=B7_EXPECTED_MODEL,
                         expected_seed=B7_EXPECTED_SEED,
                         expected_keys=B7_EXPECTED_KEYS):
    """Validate the B7 identity and return its G2-free state dict.

    Fails on a wrong file hash, epoch, model, seed, key count or on any G2 key.
    """
    if sha256 != expected_sha256:
        raise ValueError(
            f"init checkpoint SHA256 mismatch: {sha256} != {expected_sha256}")
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    model_name = payload.get("model_name") if isinstance(payload, dict) else None
    seed = payload.get("seed") if isinstance(payload, dict) else None
    if epoch != expected_epoch:
        raise ValueError(f"init checkpoint epoch mismatch: {epoch!r} != {expected_epoch}")
    if model_name != expected_model:
        raise ValueError(
            f"init checkpoint model mismatch: {model_name!r} != {expected_model!r}")
    if seed != expected_seed:
        raise ValueError(f"init checkpoint seed mismatch: {seed!r} != {expected_seed}")
    state = extract_checkpoint_state(payload)
    g2_keys = {key for key in state if key.startswith(G2_PREFIX)}
    if g2_keys:
        raise ValueError(
            f"B7 init checkpoint must not carry G2 keys, found {len(g2_keys)}")
    if len(state) != expected_keys:
        raise ValueError(
            f"B7 init state dict has {len(state)} keys, expected {expected_keys}")
    return state


def verify_b7_run_config(config):
    """Validate the frozen B7 run recipe paired with the initialization file."""
    if not isinstance(config, dict) or not isinstance(config.get("cli"), dict):
        raise ValueError("B7 config must contain a cli mapping")
    cli = config["cli"]
    expected = {
        "model": "M1",
        "epochs": 35,
        "batch_size": 32,
        "seed": 42,
        "norm": "sumnorm",
        "scale_mode": "eta",
    }
    mismatches = {
        key: (cli.get(key), value)
        for key, value in expected.items()
        if cli.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B7 config mismatch: {mismatches}")
    if "use_g2" in cli:
        raise ValueError("frozen B7 config must predate and omit use_g2")
    return cli


def validate_arm_state_dict(mode, model_keys, init_keys, num_encoder_layers):
    """Check key alignment and return the G2 keys the arm may miss from B7."""
    expected = expected_g2_keys(num_encoder_layers, mode)
    actual = {key for key in model_keys if key.startswith(G2_PREFIX)}
    if actual != expected:
        raise ValueError(
            f"{mode} arm G2 key set mismatch: missing={sorted(expected - actual)[:5]} "
            f"unexpected={sorted(actual - expected)[:5]}")
    init_g2 = {key for key in init_keys if key.startswith(G2_PREFIX)}
    if init_g2:
        raise ValueError(
            f"init checkpoint must be a G2-free B7 backbone, found {sorted(init_g2)[:5]}")
    shared = set(model_keys) - expected
    missing = shared - set(init_keys)
    unexpected = set(init_keys) - shared
    if missing or unexpected:
        raise ValueError(
            f"init checkpoint key mismatch for {mode}: "
            f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    return expected


def load_initial_state(transformer, mode, init_state, num_encoder_layers):
    """Load the B7 backbone under the per-arm contract.

    ``control`` loads strictly; ``radial`` may miss exactly the complete radial
    G2 key set; ``joint`` goes through the production joint loader.
    """
    from run_ablation_experiments import load_joint_initial_state

    expected = validate_arm_state_dict(
        mode, set(transformer.state_dict()), set(init_state), num_encoder_layers)
    if mode == "joint":
        return load_joint_initial_state(transformer, init_state)
    result = transformer.load_state_dict(init_state, strict=(mode == "control"))
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError(
            f"{mode} load did not miss exactly the G2 keys: "
            f"missing={sorted(result.missing_keys)[:5]} "
            f"unexpected={sorted(result.unexpected_keys)[:5]}")
    return result


def verify_initialized(transformer, mode, init_state, optimizer, num_encoder_layers):
    """Pre-optimization invariants for one arm.

    Shared (non-G2) weights must be bit-equal to the B7 init, the G2 key set
    must match the mode exactly, every radial/joint ``alpha`` must be zero and
    the optimizer must not have restored any state.
    """
    state = transformer.state_dict()
    expected = expected_g2_keys(num_encoder_layers, mode)
    actual = {key for key in state if key.startswith(G2_PREFIX)}
    if actual != expected:
        raise ValueError(f"{mode} G2 key set mismatch after init")
    shared = set(state) - expected
    for key in sorted(shared):
        if key not in init_state:
            raise ValueError(f"shared key missing from init state: {key}")
        if not torch.equal(state[key].detach().cpu(), init_state[key].detach().cpu()):
            raise ValueError(f"shared weight differs from B7 init: {key}")
    if mode in ("radial", "joint"):
        for layer in range(num_encoder_layers):
            alpha = state[f"{G2_PREFIX}{layer}.alpha"]
            if torch.count_nonzero(alpha).item() != 0:
                raise ValueError(f"G2 alpha must be zero after init (layer {layer})")
    if len(optimizer.state) != 0:
        raise ValueError("optimizer state must be empty before the first step")
    return expected


def _ratio(numerator, denominator):
    return (numerator / denominator) if denominator else None


def build_comparison(arms):
    """Cold-step time/memory ratios between the three arms (never steady state)."""
    def pair(left, right):
        left_cold = arms[left]["cold"]
        right_cold = arms[right]["cold"]
        return {
            "time_ratio": _ratio(left_cold["gpu_time_ms"], right_cold["gpu_time_ms"]),
            "wall_time_ratio": _ratio(
                left_cold["wall_time_ms"], right_cold["wall_time_ms"]),
            "peak_alloc_ratio": _ratio(
                left_cold["peak_alloc_mb"], right_cold["peak_alloc_mb"]),
            "peak_reserved_ratio": _ratio(
                left_cold["peak_reserved_mb"], right_cold["peak_reserved_mb"]),
        }

    return {
        "joint_vs_control": pair("joint", "control"),
        "joint_vs_radial": pair("joint", "radial"),
        "radial_vs_control": pair("radial", "control"),
    }


def require_cuda_device(device):
    """Reject a non-CUDA device before any model is moved."""
    if device.type != "cuda":
        raise RuntimeError(
            f"the resource gate requires a CUDA device, got {device.type!r}")
    return device


def require_v100(device_name):
    """Reject any accelerator whose name does not contain ``V100``."""
    if "V100" not in device_name:
        raise RuntimeError(f"the resource gate requires a V100, got {device_name!r}")
    return device_name


def batch_digest(batch):
    """Deterministic digest of a batch, used to prove all arms share batch 0."""
    digest = hashlib.sha256()
    for item in batch:
        if torch.is_tensor(item):
            digest.update(str(tuple(item.shape)).encode())
            digest.update(
                item.detach().to(torch.float64).contiguous().numpy().tobytes())
        else:
            digest.update(repr(item).encode())
    return digest.hexdigest()


def verify_same_batch(arms):
    """Require all isolated workers to have consumed one identical train batch."""
    digests = {mode: arms[mode]["batch"]["batch_sha256"] for mode in ARMS}
    sizes = {mode: arms[mode]["batch"]["batch_size"] for mode in ARMS}
    if set(sizes.values()) != {32}:
        raise ValueError(f"resource gate requires batch size 32 in every arm: {sizes}")
    if len(set(digests.values())) != 1:
        raise ValueError(f"resource-gate batch mismatch across arms: {digests}")
    return next(iter(digests.values()))


# ---------------------------------------------------------------------------
# Worker: build one arm, verify it, run exactly one training step
# ---------------------------------------------------------------------------

def _ensure_imports():
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(eval_dir, "..", ".."))
    for path in (repo_root, eval_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def build_arm_yaml_cfg(mode, data_dir, r_cut=5.5):
    """Production ``g2_resource_gate`` recipe plus the candidate-1 content mode."""
    _ensure_imports()
    from g2_resource_gate import build_production_yaml_cfg

    yaml_cfg = build_production_yaml_cfg(
        use_g2=(mode != "control"), r_cut=r_cut, data_dir=data_dir)
    content_mode = "joint" if mode == "joint" else "radial"
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["g2_content_mode"] = content_mode
    return yaml_cfg


def measure_arm(mode, init_ckpt, init_config, expected_sha256, data_dir, device_str,
                r_cut=5.5, seed=42):
    """Measure one cold training step for one arm inside a clean process."""
    _ensure_imports()
    from datasets.dataset import Dos_Dataset
    from run_ablation_experiments import setup_ablation_seed
    from torch.utils.data import DataLoader
    from utils.builder import ConfigBuilder

    if mode not in ARMS:
        raise ValueError(f"mode must be one of {ARMS}, got {mode!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("the resource gate requires an available CUDA device")
    device = require_cuda_device(torch.device(device_str))
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device_name = require_v100(torch.cuda.get_device_name(device_index))

    # 1. Frozen B7 checkpoint identity: hash, epoch, model, seed, 205 keys, no G2.
    sha256 = file_sha256(init_ckpt)
    payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    init_state = verify_b7_checkpoint(payload, sha256, expected_sha256=expected_sha256)
    with open(init_config) as handle:
        verify_b7_run_config(yaml.safe_load(handle))

    # 2. The single fixed batch: Q1 train, shuffle=False, batch 0 only.
    dataset = Dos_Dataset(
        data_dir=data_dir, split="train", dos_minmax=True, dos_sumnorm=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    for batch0 in loader:
        break
    else:
        raise RuntimeError("Q1 train split returned no batch 0")
    n_atoms = int((batch0[0][:, 2:82] != 0).sum().item())
    batch_summary = {
        "batch_size": int(batch0[0].shape[0]),
        "atom_slots": int(batch0[0][:, 2:82].numel()),
        "nonzero_atom_flags": n_atoms,
        "batch_sha256": batch_digest(batch0),
    }

    # 3. Build the arm from the production recipe.
    yaml_cfg = build_arm_yaml_cfg(mode, data_dir, r_cut=r_cut)
    num_encoder_layers = int(
        yaml_cfg["model"]["params"]["sub_model"]["transformer"]["num_encoder_layers"])
    setup_ablation_seed(seed)
    builder = ConfigBuilder(**yaml_cfg)
    model = builder.get_model()
    model.to(device)
    model.model["transformer"].train()
    total_params = sum(
        p.numel() for p in model.model["transformer"].parameters() if p.requires_grad)

    # 4. Load the shared B7 weights under the per-mode contract.
    load_initial_state(model.model["transformer"], mode, init_state, num_encoder_layers)

    # 5. Pre-optimization invariants: shared weights bit-equal to B7, G2 alpha
    #    exactly zero, exact G2 key set, no restored optimizer state.
    optimizer = model.optimizer["transformer"]
    verify_initialized(
        model.model["transformer"], mode, init_state, optimizer, num_encoder_layers)

    # 6. Fair RNG start: re-seed after loading and before the first forward.
    setup_ablation_seed(seed)

    # 7. Exactly one complete training step on the same batch.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    loss_dict = model.train_one_step(batch0, step=0)
    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    for key in LOSS_KEYS:
        if key not in loss_dict:
            raise RuntimeError(f"train_one_step did not return {key}")
        value = float(loss_dict[key])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite {key}={value} for arm {mode}")

    alpha_grad_norms = []
    if mode in ("radial", "joint"):
        for layer, message in enumerate(model.model["transformer"].encoder.g2_msgs):
            alpha = message.alpha
            if alpha.grad is None or not torch.isfinite(alpha.grad).all():
                raise RuntimeError(
                    f"arm {mode} layer {layer} has no finite G2 alpha gradient")
            alpha_grad_norms.append(float(alpha.grad.detach().norm().cpu()))

    return {
        "arm": mode,
        "content_mode": "joint" if mode == "joint" else "radial",
        "use_g2": mode != "control",
        "device": device_name,
        "trainable_params": int(total_params),
        "alpha_grad_norms": alpha_grad_norms,
        "batch": batch_summary,
        "cold": {
            "gpu_time_ms": float(start_event.elapsed_time(end_event)),
            "wall_time_ms": (wall_end - wall_start) * 1000.0,
            "peak_alloc_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
            "loss": float(loss_dict["loss"]),
            "loss_edos": float(loss_dict["loss_edos"]),
            "loss_phdos": float(loss_dict["loss_phdos"]),
            "loss_eta": float(loss_dict["loss_eta"]),
        },
    }


# ---------------------------------------------------------------------------
# Master: isolated subprocesses and atomic result writing
# ---------------------------------------------------------------------------

def run_isolated_process(mode, init_ckpt, init_config, expected_sha256, data_dir, r_cut,
                         device, out_json):
    """Run one arm in a clean interpreter so VRAM cannot cross-contaminate."""
    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--mode", mode,
        "--init_ckpt", init_ckpt,
        "--init_config", init_config,
        "--expected_sha256", expected_sha256,
        "--data_dir", data_dir,
        "--r_cut", str(r_cut),
        "--device", device,
        "--out_json", out_json,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Worker {mode} failed with code {result.returncode}:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(
            f"arm {mode} crashed (possible OOM, non-finite loss, or CUDA error)")
    with open(out_json) as handle:
        return json.load(handle)


def atomic_write_json(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def atomic_write_csv(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    os.replace(tmp, path)


def summary_csv_rows(summary):
    rows = [[
        "arm", "params", "gpu_time_ms", "wall_time_ms", "peak_alloc_mb",
        "peak_reserved_mb", "loss", "loss_edos", "loss_phdos", "loss_eta",
    ]]
    for mode in ARMS:
        arm = summary["arms"][mode]
        cold = arm["cold"]
        rows.append([
            mode, arm["trainable_params"],
            f"{cold['gpu_time_ms']:.3f}", f"{cold['wall_time_ms']:.3f}",
            f"{cold['peak_alloc_mb']:.2f}", f"{cold['peak_reserved_mb']:.2f}",
            f"{cold['loss']:.6f}", f"{cold['loss_edos']:.6f}",
            f"{cold['loss_phdos']:.6f}", f"{cold['loss_eta']:.6f}",
        ])
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Internal worker mode")
    parser.add_argument("--mode", choices=list(ARMS), default="control")
    parser.add_argument("--init_ckpt", default=B7_CKPT_DEFAULT)
    parser.add_argument("--init_config", default=B7_CONFIG_DEFAULT)
    parser.add_argument("--expected_sha256", default=B7_CKPT_SHA256)
    parser.add_argument("--data_dir", default="./data/train4ARPAT")
    parser.add_argument("--r_cut", type=float, default=5.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_json", default=DEFAULT_RESULTS_JSON)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.worker:
        metrics = measure_arm(
            args.mode, args.init_ckpt, args.init_config, args.expected_sha256, args.data_dir,
            args.device, r_cut=args.r_cut)
        atomic_write_json(metrics, args.out_json)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("the resource gate requires CUDA on a V100")
    device = require_cuda_device(torch.device(args.device))
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    gpu_name = require_v100(torch.cuda.get_device_name(device_index))
    results_dir = os.path.dirname(os.path.abspath(args.out_json)) or "."
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 66)
    print("  JOINT CONTENT (candidate 1) V100 RESOURCE GATE - single cold step")
    print(f"  Device: {gpu_name}")
    print(f"  Batch size: 32 (Q1 train, shuffle=False, batch 0)")
    print(f"  Init: {args.init_ckpt}")
    print("=" * 66)

    arms = {}
    temp_paths = []
    try:
        for mode in ARMS:
            temp_path = os.path.join(results_dir, f"_tmp_joint_content_{mode}.json")
            temp_paths.append(temp_path)
            print(f"\n[{ARMS.index(mode) + 1}/{len(ARMS)}] Measuring {mode} in a clean process...")
            arms[mode] = run_isolated_process(
                mode, args.init_ckpt, args.init_config, args.expected_sha256, args.data_dir,
                args.r_cut, args.device, temp_path)
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

    batch_sha256 = verify_same_batch(arms)
    summary = {
        "tool": "joint_content_resource_gate",
        "measurement": "single_cold_train_step",
        "steady_state": False,
        "note": "one complete train_one_step per arm on the same fixed batch; "
                "cold-step timings are cost evidence, not a steady-state claim, "
                "and carry no automatic stop threshold",
        "device": gpu_name,
        "batch_size": 32,
        "batch_sha256": batch_sha256,
        "r_cut": args.r_cut,
        "init_ckpt": args.init_ckpt,
        "init_config": args.init_config,
        "expected_sha256": args.expected_sha256,
        "arms": arms,
        "comparison": build_comparison(arms),
        "verdict": "MEASURED_NO_OOM_SINGLE_STEP",
    }
    atomic_write_json(summary, args.out_json)
    csv_path = (args.out_json[:-5] if args.out_json.endswith(".json")
                else args.out_json) + ".csv"
    atomic_write_csv(summary_csv_rows(summary), csv_path)

    print("\n" + "=" * 66)
    print("  RESOURCE GATE SUMMARY (cold step, not steady state)")
    print("=" * 66)
    for mode in ARMS:
        cold = arms[mode]["cold"]
        print(f"  {mode:8s} params={arms[mode]['trainable_params']:,} "
              f"time={cold['gpu_time_ms']:.1f}ms wall={cold['wall_time_ms']:.1f}ms "
              f"peak_alloc={cold['peak_alloc_mb']:.1f}MB "
              f"peak_reserved={cold['peak_reserved_mb']:.1f}MB loss={cold['loss']:.4f}")
    for name, values in summary["comparison"].items():
        print(f"  {name:18s} time_ratio={values['time_ratio']} "
              f"peak_alloc_ratio={values['peak_alloc_ratio']}")
    print(f"VERDICT: {summary['verdict']}")
    print(f"Saved: {args.out_json} and {csv_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
