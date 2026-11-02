"""One fixed LR2 E+S scheduling intervention; local and global formulas stay put."""
from types import SimpleNamespace

from model.local_global_encoder import MIDPOINT_ORDER
from utils.local_residual_experiment import validate_residual_model

MIDPOINT_SCHEMA = "joint-spectral-balance-layout-model-v1"


def layout_ablation_contract():
    return {"reference": "LR2 E+S staged L2/G6", "local_route": "LR2",
            "reference_layout": "staged", "candidate_layout": "midpoint",
            "reference_order": [["local", 0], ["local", 1]] + [["global", i] for i in range(6)],
            "execution_order": [list(step) for step in MIDPOINT_ORDER],
            "local_calls": 2, "internal_local_blocks": 4, "global_attention": 6, "global_ffn": 6,
            "local_formula": "LN64(F(s,R))", "widths": [64, 128],
            "internal_normalization": "post-norm", "outer_normalization": "pre-norm512",
            "local_readout": "W_out*s2+b_out", "outer_residual": "h+P_out(s2)",
            "all_initial_tensors_identical": True, "joint_training": True,
            "scope": "move only the second complete local call after global layer 3; "
                     "same independent modules, parameter keys, initialization, loss and selection"}


def validate_midpoint_model(transformer, params, *, require_identity=False):
    """Validate the actual order, then reuse every staged LR2 content check."""
    encoder = transformer.encoder
    if (params.get("local_route") != "LR2" or params.get("local_layout") != "midpoint"
            or params.get("scale_mode") != "per_atom" or encoder.layout != "midpoint"
            or encoder.depth != 2 or encoder.global_depth != 6
            or encoder.execution_order != MIDPOINT_ORDER):
        raise ValueError("midpoint must execute L1 G1 G2 G3 L2 G4 G5 G6 on LR2 per-atom spectra")
    # No live attribute is mutated. Historical validators continue to require
    # staged recipes; this proxy checks the unchanged modules and tensors only.
    reference_params = {**params, "local_layout": "staged"}
    proxy = SimpleNamespace(layout="staged", **{name: getattr(encoder, name) for name in
        ("route", "geometry_route", "depth", "global_depth", "local_layers", "global_layers")})
    if hasattr(encoder, "local_readout_gate"):
        raise ValueError("midpoint cannot introduce a local readout gate")
    validate_residual_model(SimpleNamespace(local_route=transformer.local_route, encoder=proxy,
                                           named_parameters=transformer.named_parameters),
                            reference_params, require_identity=require_identity)
