# 现状（活文档，每次收尾更新；历史进日志，本页只写 now/next）

更新：2026-09-18 ｜ 基线：B7 `_e9ctl`（Q1，M1×35 best ep33，e 0.518/5.73% p 0.741/3.50% Cv 0.30）

## Now
- E9-P0：G1 挂起 → Qc1 park → Qc2 park（Q组关闭）→ L3 pilot 进行中（`_l3kl` 已落盘打平，`_l3now1/_l3nohub` 待落）。
- 数据：Q1 干净池 18706/2313/2287；旧缓存归档 `data/archive/train4ARPAT_20260916_preQ1/`。
- 默认配方：sumnorm + E0P0 + eta + dropout 0.05。

## Next（按序，一次一棒）
1. L3 verdict 落盘（vs `ctl_Q1-10`）：有臂赢→合并改默认，否则 L 组关闭。
2. E9-P0 下一候选：L3 后由总指挥在 L3/C5前移二选一。
3. E9 后队列（不提前开）：C5新读出 → warp网格 → 非均匀大窗 → 密度臂 → C1会师/D3/C3 → 连续谱场。

## Blocker / Watch
- `dataset.py` coords 默认断言回归已修（auto 兜底）。C2b 重跑前先冒烟。
- 精简完成（09-18）：根目录仅 3 py；旧入口归档 `tools/legacy/`（E8关闭）；收官 Design（A3/A4/A6/B2）归档 `01_开发文档/archive/`，现行只剩 A7/C5/E/P0；
  死代码 `utils/rbf_encoding.py` 已删（零导入，功能在 `rp_encoding.py`）；`results/pred_*.npy` 已删（省约94MB，可重算）；`data/archive/` 已ignore。
- `output/` 约74G checkpoints 不动：删任一ckpt需总指挥逐项批（不可逆GPU成本）。`output/*/config_used.yaml` 不入库，以 `results/` 为准。
