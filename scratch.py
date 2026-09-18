import tokenize
from io import BytesIO

params = {
    'model_name', 'epochs', 'batch_size', 'lr', 'skip_existing', 'seed', 'tag', 'data_dir', 
    'edos_num', 'phdos_num', 'atom_feat', 'energy_code', 'edos_grid', 'tv_w', 'grad_w', 'peak_w', 
    'tail_w', 'tail_start', 'augment', 'disp_sigma', 'norm', 'use_mask', 'dropout', 'weight_decay', 
    'warmup_epochs', 'lambda_ph', 'grad_clip', 'scale_mode', 'freeze_backbone', 'init_ckpt', 'scale_sup_w', 
    'eta_sup_w', 'delta_edos', 'delta_phdos', 'scalar_mode', 'scalar_sup_w', 'use_g1', 'g1_r_cut', 
    'g1_max_neighbors', 'q1_coord', 'q1_hidden', 'q2_fourier', 'w_w1', 'w_huber'
}

with open('run_ablation_experiments.py', 'r') as f:
    code = f.read()

# First replace the old yaml cfg with yaml_cfg
# But wait, it's easier to just do it via string replace since cfg is a unique enough token
code = code.replace("cfg['", "yaml_cfg['")
code = code.replace("cfg = yaml.load(f, Loader=yaml.FullLoader)", "yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)")
code = code.replace("builder = ConfigBuilder(**cfg)", "builder = ConfigBuilder(**yaml_cfg)")
code = code.replace("'config': cfg}", "'config': yaml_cfg}")
code = code.replace("with open('configs/config.yaml') as f:", "with open('configs/default.yaml') as f:")

if 'from utils.experiment_config import ExperimentConfig' not in code:
    code = code.replace('from model.model import basemodel', 'from model.model import basemodel\nfrom utils.experiment_config import ExperimentConfig')

# We need to find the bounds of train_and_eval
sig_start = code.find('def train_and_eval(')
sig_end = code.find('):', sig_start) + 2
main_idx = code.find("if __name__ == '__main__':")

train_eval_body = code[sig_end:main_idx]

# Tokenize and replace
tokens = list(tokenize.tokenize(BytesIO(train_eval_body.encode('utf-8')).readline))
out = []
prev_tok = None
for i, tok in enumerate(tokens):
    if tok.type == tokenize.NAME and tok.string in params:
        # Check if previous token was a dot
        if i > 0 and tokens[i-1].string == '.':
            out.append((tok.type, tok.string))
        elif i > 0 and tokens[i-1].string == '=' and tokens[i-2].string in params:
            # kwarg assignment like w_w1=w_w1 -> w_w1=cfg.w_w1
            out.append((tokenize.NAME, 'cfg.' + tok.string))
        elif i < len(tokens) - 1 and tokens[i+1].string == '=':
            # this is a kwarg key like edos_num=edos_num, we only replace the value, not the key
            out.append((tok.type, tok.string))
        else:
            out.append((tokenize.NAME, 'cfg.' + tok.string))
    else:
        out.append((tok.type, tok.string))

new_body = tokenize.untokenize(out).decode('utf-8')

# Assemble
new_sig = "def train_and_eval(cfg: ExperimentConfig):"
code = code[:sig_start] + new_sig + new_body + code[main_idx:]

# Update __main__
new_main = """if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='uniARPAT Ablation Experiments Runner')
    parser.add_argument('--model', type=str, default='M1', choices=['M1', 'M2', 'M3', 'M4', 'M5', 'all'], help='Model variant to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--skip_existing', action='store_true', help='Skip variant if test summary already exists')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (H1 hygiene, recorded in history CSV)')
    parser.add_argument('--tag', type=str, default='', help='Run tag, e.g. _b96: isolates save_dir/results from h1 outputs')
    parser.add_argument('--data_dir', type=str, default='./data/train4ARPAT', help='Dataset root (C2b: per-arm dir)')
    parser.add_argument('--edos_num', type=int, default=128, help='eDOS output bins (C2b grid arms)')
    parser.add_argument('--phdos_num', type=int, default=64, help='phDOS output bins (C2b grid arms)')
    parser.add_argument('--atom_feat', type=str, default='legacy3', choices=['legacy3', 'mendeleev24'], help='Atom feature table (C1.1)')
    parser.add_argument('--energy_code', type=str, default='none', choices=['none', 'edos'], help='eDOS bin-energy code (C1.2)')
    parser.add_argument('--edos_grid', type=str, default='', help='C1.2 grid key (E0/E2..) in grids.json or path to centers npy')
    parser.add_argument('--tv_w', type=float, default=0.0, help='C1.3 TV weight')
    parser.add_argument('--grad_w', type=float, default=0.0, help='C1.3 gradient-match weight')
    parser.add_argument('--peak_w', type=float, default=1.0, help='C1.3 peak-region weight')
    parser.add_argument('--tail_w', type=float, default=1.0, help='C1.3 phDOS tail weight')
    parser.add_argument('--tail_start', type=int, default=-1, help='C1.3 tail start bin (-1=off)')
    parser.add_argument('--augment', action='store_true', help='C1.4 phonon displacement aug (train only)')
    parser.add_argument('--disp_sigma', type=float, default=0.01, help='C1.4 displacement sigma (frac)')
    parser.add_argument('--norm', type=str, default='sumnorm', choices=['minmax', 'sumnorm'], help='Target norm (C2.1 merged default; minmax recovers legacy)')
    parser.add_argument('--use_mask', action='store_true', help='C2.3 coverage-mask the loss (eval protocol unchanged)')
    parser.add_argument('--dropout', type=float, default=None, help='B4 transformer dropout (default config 0.1)')
    parser.add_argument('--weight_decay', type=float, default=None, help='B4 AdamW weight decay (default 0.01)')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='B4 warmup epochs (default 5)')
    parser.add_argument('--lambda_ph', type=float, default=None, help='B4 phonon loss weight (default 1.0)')
    parser.add_argument('--grad_clip', type=float, default=None, help='B4 grad clip max-norm (default off)')
    parser.add_argument('--w_w1', type=float, default=None, help='L3 W1/CDF term weight (default 1.0)')
    parser.add_argument('--w_huber', type=float, default=None, help='L3 Huber term weight (default 1.0)')
    parser.add_argument('--scale_mode', type=str, default='eta', choices=['none', 'decoupled', 'eta'], help='C2.4/H1 supervised scale/coverage head')
    parser.add_argument('--eta_sup_w', type=float, default=1.0, help='H1 eta/gamma supervision weight')
    parser.add_argument('--delta_edos', type=float, default=0.09375, help='H1 eDOS bin width (E0)')
    parser.add_argument('--delta_phdos', type=float, default=19.6875, help='H1 phDOS bin width (P0)')
    parser.add_argument('--scalar_mode', type=str, default='none', choices=['none', 's1'], help='S1 boundary scalar heads')
    parser.add_argument('--scalar_sup_w', type=float, default=1.0, help='S1 scalar supervision weight')
    parser.add_argument('--use_g1', action='store_true', help='E9-P0 G1 exact sparse graph + hub token')
    parser.add_argument('--g1_r_cut', type=float, default=5.5, help='G1 cutoff Angstrom (Design-E: 5.5)')
    parser.add_argument('--g1_max_neighbors', type=int, default=48, help='G1 per-atom neighbor cap (Design-E: 48)')
    parser.add_argument('--q1_coord', action='store_true', help='E9-P0 Q1 coordinate trunk MLPs')
    parser.add_argument('--q1_hidden', type=int, default=128, help='Q1 trunk hidden dim')
    parser.add_argument('--q2_fourier', action='store_true', help='E9-P0 Q2 RFF trunk (implies trunk pathway)')
    parser.add_argument('--freeze_backbone', action='store_true', help='C2.4 Phase A: train scale head only')
    parser.add_argument('--init_ckpt', type=str, default='', help='C2.4 init weights (strict=False)')
    parser.add_argument('--scale_sup_w', type=float, default=1.0, help='C2.4 scale supervision weight')
    args = parser.parse_args()

    if args.model == 'all':
        for m in ['M1', 'M2', 'M3', 'M4', 'M5']:
            args.model = m
            cfg = ExperimentConfig.from_args(args)
            train_and_eval(cfg)
    else:
        cfg = ExperimentConfig.from_args(args)
        train_and_eval(cfg)
"""

code_main_idx = code.find("if __name__ == '__main__':")
code = code[:code_main_idx] + new_main

with open('run_ablation_experiments.py', 'w') as f:
    f.write(code)
