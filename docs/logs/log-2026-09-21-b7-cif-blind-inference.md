# 工作日志 2026-09-21：B7 M1 CIF 盲推理入口

## 范围

- **目标：**把冻结的 B7 `_e9ctl` M1 作为一个可审计的 CIF-only blind 推理入口导出。
- **改动：**新增 `b7_cif_infer.py` 与 `utils/b7_cif_inference.py`；新增 3 项 CPU 合同测试，并把它们纳入
  E7 静态检查。旧 `cif2dos.py` 不变，仍仅兼容历史 M4。
- **边界：**不训练、不微调、不改 B7 checkpoint、默认 YAML、模型或数据缓存；checkpoint 必须由调用者经
  `--checkpoint` 明确提供。

## 证据

- B7 配置取自本地 `_e9ctl` 的有效配置：M1、512 维、6 encoder＋6 shared decoder、`legacy3`、
  0.05 dropout、SumNorm、H1 `scale_mode=eta`、E0/P0。
- blind 公式复用 B7 盲验证的已冻结口径：eDOS 为 `softmax(logits) * N_val(CIF) * gamma_hat / 0.09375`；
  phDOS 为 `softmax(logits) * 3*N_atoms(CIF)*eta_hat / 19.6875`。`N_val` 从版本控制的
  `index/z0_zval.json` 按 CIF 中每个原子相加，不读取标签或 Q1 缓存。
- `python3 -m unittest tests.test_b7_cif_inference`：**4/4** 通过，覆盖 82-row／`1/c` 特征、Z0、
  E0/P0 中心、H1 总量公式、临时 CIF 的 `npz`／JSON 写出，以及无序、超 80 原子和非 B7 checkpoint 拒绝。
- `bash tools/ci/check-static.sh`：**40 项**缓存无关合同测试通过，并包含上述 4 项。
- `python3 -m unittest discover tests -q`：完整本地回归 **91/91** 通过（31.44 秒）。
- 本地只读 B7 checkpoint CPU 烟测：`output/ablation_m1_e9ctl/checkpoint_best.pth` 元数据为
  `M1`／epoch 33／seed 42；合成 Si（2 原子、`N_val=8`）前向得到 `eta=0.857857`、
  `gamma=0.916209`、eDOS/phDOS 总量 `78.1832/0.261442`，数组形状 `(128,)/(64,)` 且有限非负。

## 结论

- **状态：完成。**`b7_cif_infer.py` 是 B7 M1 的正式 blind 导出入口；它严格认证 checkpoint 和 state dict，
  拒绝随机初始化、M4／其他 epoch checkpoint、无序 CIF、超过 80 原子或不在 Z0／embedding 支持范围的元素。
- 输出是 E0/P0 窗口内、CIF 单胞约定下的 B7 blind 谱。eDOS 的 H1 p99 blind gap 已知较大，不能将输出标为
  oracle 或完整全能区谱。

## 交接

- 使用示例：

  ```bash
  python3 b7_cif_infer.py --cif structure.cif \
    --checkpoint output/ablation_m1_e9ctl/checkpoint_best.pth \
    --output predictions/b7
  ```

- 下一项科学／工程候选尚未从 C3 PhysMoE、D3 辅助数据与尖峰／虚频审计中确定；不在此入口上继续叠加模型或
  数据因素。
- 已更新 `README.md`、`status.md`、`decisions.md`、索引和 E7 门禁。
