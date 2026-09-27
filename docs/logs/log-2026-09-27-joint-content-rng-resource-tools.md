# 日志：2026-09-27 — 候选1 RNG 公平前置与阶段 A 资源核查

## 范围

- 用户批准实现 `init_ckpt` 载入后的 RNG 重置、阶段 A 资源工具、阶段 B 判决工具及 CPU
  合同测试，并批准在 V100 上执行阶段 A；三臂 10 epoch pilot 不在本轮授权内。
- 本日志记录 RNG 前置、资源工具和阶段 A 的最终权威结果。判决工具另见
  `log-2026-09-27-joint-content-verdict-tool.md`。
- 实现初稿由 OpenCode Go / DeepSeek V4.1 Flash 完成；协调者复核后补上 B7 配置、三臂 batch
  指纹、六层 alpha 梯度和设备身份硬校验，再执行最终阶段 A。

## 改动与调用链

### RNG 公平前置

- `utils/experiment_config.py::ExperimentConfig.reset_rng_after_init`：新增默认关闭的显式开关。
- `run_ablation_experiments.py::validate_reset_rng_config`：开关未配合 `init_ckpt` 时，在创建目录和
  数据加载器之前拒绝。
- `run_ablation_experiments.py::train_and_eval`：模型构造和 `init_ckpt` 模型权重载入完成后、首次
  训练迭代前调用既有 `setup_ablation_seed(cfg.seed)`；默认关闭路径不变。
- `run_ablation_experiments.py::build_arg_parser`：暴露 `--reset_rng_after_init`；有效配置写入
  `config_used.yaml` 的 `cli.reset_rng_after_init`。
- CPU 合同同时确认 `DistributedSampler(seed=0).set_epoch(epoch)` 的顺序不受全局 RNG 重置影响。

### 阶段 A 工具

- 新增 `tools/eval/joint_content_resource_gate.py`，固定三臂 `control/radial/joint`、V100、batch 32、
  Q1 train `shuffle=False` 的 batch 0；代码没有 valid/test 加载入口。
- B7 预检同时锁定 checkpoint SHA-256、epoch 33、M1、seed 42、205 个无 G2 键的 state dict，
  以及配套 `config_used.yaml` 的 M1×35、batch 32、SumNorm、eta 和旧配置无 `use_g2` 字段。
- 初始化合同：control 严格载入；radial 只允许完整径向 G2 键集缺失；joint 复用生产
  `load_joint_initial_state`；优化前所有共享键与 B7 `torch.equal`、六层 alpha 全零、optimizer
  state 为空。
- 每臂在独立进程中载入 B7 后重置 RNG，只执行一次完整 `train_one_step`。主进程要求三臂
  `batch_sha256` 完全一致；所有四项损失有限，radial/joint 六层 alpha 梯度都存在且有限。
- 只有三臂全部成功后才生成
  `results/joint_content_resource_gate_v100.json` 与 `.csv`；计时明确是单次冷启动训练步，不代表
  稳态或整轮成本，也没有自动成本淘汰线。

## 阶段 A 结果

执行命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 tools/eval/joint_content_resource_gate.py
```

设备为 `Tesla V100-SXM2-32GB`。B7 checkpoint SHA-256 为
`cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40`。三臂 batch 指纹均为
`a0af8fd0fb3afe503f07472d7bb7d1d6377f934e5b5f4737893b9720d3a460cc`，batch 32 中共有 81 个
非零原子槽。三臂首步损失逐值相同：总损失 `0.68687594`、eDOS `0.25892028`、phDOS
`0.41948882`、eta `0.00846681`。

| 臂 | 可训练参数 | 冷启动 GPU 时间 | 峰值 allocated | 峰值 reserved |
|---|---:|---:|---:|---:|
| control | 71,152,964 | 1857.6 ms | 6554.7 MB | 7344 MB |
| radial | 74,504,522 | 1993.7 ms | 6805.0 MB | 7634 MB |
| joint | 79,230,794 | 1978.2 ms | 6916.4 MB | 7788 MB |

- joint/control：时间 `1.0649x`，allocated `1.0552x`，reserved `1.0605x`。
- joint/radial：时间 `0.9923x`，allocated `1.0164x`，reserved `1.0202x`。
- radial/control：时间 `1.0732x`，allocated `1.0382x`，reserved `1.0395x`。

冷启动单点中 joint 比 radial 快 `0.8%` 属于单次测量波动，不能解释为稳态加速。可确认的结论
只有：三臂都能在固定 V100 batch 32 上完成完整单步，无 OOM、NaN/Inf 或 G2 梯度断路；阶段 A
技术通过。

## 验证

- `python3 -m unittest tests.test_joint_content_pilot -v`：最终 **38 项通过**，覆盖 RNG 顺序、
  sampler 独立性、checkpoint/config/key/batch 合同、裁决边界与 mock 全流程原子落盘。
- 新工具、runner、配置和测试的 Ruff/`py_compile`：通过。
- `bash tools/ci/check-static.sh`：最终 **158 项通过**。
- `git diff --check`：通过。
- 未执行：valid/test 前向、三臂 10 epoch pilot、缓存重建、数据或依赖修改。

## 结论与下一关

- 阶段 A：`MEASURED_NO_OOM_SINGLE_STEP`，技术通过；该结论不包含精度或泛化收益。
- 阶段 B 判决工具已经实现并通过合成合同测试，但阶段 B 训练仍须用户另行批准。
- 历史 G2a 实测约为 control `189.9 s/epoch`、radial `241.7 s/epoch`；结合本轮 joint 冷步与
  radial 接近，三臂各 10 epoch 在单张 V100 串行的合理预算约为 **1.9 小时，另加少量启动与
  最终 valid 判决时间**。这是预算估计，不是阶段 A 冷步外推出的精确承诺。
