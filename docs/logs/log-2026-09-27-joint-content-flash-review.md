# 联合边内容：runner/恢复/配置链基线兼容性审查（Flash）

> **基线兼容性审查，不是最终补丁通过意见。** 只读 `git show HEAD:45c2e85` 的 runner / checkpoint / config
> 与现有恢复测试；未运行任何训练、测试、GPU 或网络操作，未改业务代码。范围仅：新增
> `g2_content_mode=radial|joint` 对初始化、恢复与配置落盘的要求。joint 默认关闭、旧 B7/G2a 行为须保留。

## 结论
新开关本身不改旧路径；风险集中在两类调用链：**同一配置被多处手工枚举**，以及**恢复/热启动没有显式模式标识**。

## 易漏接 / 宽松加载 / 副作用链（文件::符号）
1. **配置双重枚举（新风险，最易漏接）** `run_ablation_experiments.py::train_and_eval`：一个开关要写两处——
   `yaml_cfg['model']['params']['sub_model']['transformer'][...]`（现 `use_g2`/`g2_r_cut`，约 194–195 行）与落盘的
   `config_used.yaml` `'cli':{...}`（约 246/345 行）。只加一处 → 实跑 joint 而配置标 radial，违反“保存配置必须准确标新模式”。
   同时 `utils/experiment_config.py::ExperimentConfig` 必须有 `g2_content_mode` 字段，否则 `ExperimentConfig.from_args`
   按 `fields(cls)` 过滤会**静默丢弃**同名 CLI 参数。
2. **热启动宽松加载（新风险）** runner `if cfg.init_ckpt:` 块（约 251–257 行）：`load_state_dict(_st, strict=False)`
   不只容忍“缺新增分支键”，也容忍 **unexpected 键**。把 radial/G2a 权重灌进 joint 时，radial 内容键变 unexpected、
   joint 新键变 missing，会静默通过并让 joint 分支停在 α=0 初始化——正是必须拒绝的杂用。建议加载后检查 `_unexp`
   为空、`_miss` 全落在新增 joint 分支键前缀白名单；抽成可单测的小 helper（放 `utils/ablation_checkpoint.py`），
   不要只 `logger.info` 前 5 个键。
3. **恢复缺显式模式标识（新风险）** `utils/ablation_checkpoint.py::restore_ablation_checkpoint` 只校验 `use_amp`
   （`saved_amp != bool(use_amp)`，第 54 行）与 slope 标定，无模式参数。跨模式目前只靠第 70 行
   `transformer.load_state_dict(state)` 的 strict 默认 True，以形状差异拦截，错误不可读；若 joint 复用同名参数则无法区分。
   应新增 `g2_content_mode` 关键字，**在 `load_state_dict` 之前**做等值比较并抛 `ValueError`（与 `use_amp` 同模式），
   `build_ablation_checkpoint` 必须写入该标签。缺标签的旧 payload 必须按 **radial（legacy）** 解释；缺省不能跟随
   “当前请求值”，否则 joint 恢复旧 B7 会静默混用。
4. **恢复不重校验配置（原有副作用，放大新模式风险）** runner 的 `config_used.yaml` 写出被
   `if not os.path.exists(latest_p) or not os.path.exists(config_path)` 门控：恢复时磁盘配置被保留、不与 `cfg` 比对。
   `--g2_content_mode joint` 恢复 radial 运行时，只能靠第 3 条 checkpoint 标签拦截，否则磁盘 radial、cfg joint。
5. **非法组合未拦截（新风险）** 设计规定 joint 必须显式 `--use_g2`、不得静默忽略；但 `model/transformer.py::Transformer.__init__`
   只在 `use_g1 and use_g2` 时抛错（151–152 行），不会因 `use_g2=False` 拒绝 mode=joint，会静默不建模块。应在
   `train_and_eval` 早期（`ConfigBuilder` 之前、写任何产物之前）显式拒绝 joint 且 `use_g2=False`。
6. **原有问题（非本轮必须）** `g2_r_cut` 在 yaml_cfg 与 cli dict 都硬编码 `5.5`、未用 `cfg.g2_r_cut`；slope 校验
   仅在 `edos_slope_ratio>0` 时等值检查（0 时即使有非零元数据也不拒绝）。二者是既有模式陷阱，新增 mode 校验不要照抄
   “仅请求 >0 才检查”。`utils/b7_cif_inference.py` 第 166 行 `strict=True` 在 runner 之外，保持不动。

## 必须在新模式下拒绝的错误 + 最佳检查点
- 恢复/热启动 saved mode ≠ requested mode → `restore_ablation_checkpoint` 新增参数处（`load_state_dict` 前）。
- 热启动出现 unexpected 键，或 missing 键不在新 joint 分支白名单 → init_ckpt 加载后的小 helper。
- `mode=joint` 且 `use_g2=False` → `train_and_eval` 开头。
- 保存配置缺 `g2_content_mode` 或与实际不符 → 两处枚举点必须同步（可收敛为单一 config-dict 构造，但不做全局重构）。

## 验收条件（≤5，mock/合成 state_dict 可验）
- **A1 配置落盘**：合成 `ExperimentConfig(g2_content_mode="joint", use_g2=True)` 走 `train_and_eval`，断言
  `config_used.yaml` 的 `cli.g2_content_mode=="joint"` 且 `config...transformer.g2_content_mode=="joint"`；radial 同理。
- **A2 非法组合**：`mode=joint,use_g2=False` 抛错且未写任何产物；`mode=radial,use_g2=False` 合法。
- **A3 热启动**：仅含共享键的合成 `_st` 成功；含 radial 内容键（unexpected）时抛错，不得只记日志。
- **A4 恢复混用**：`build_ablation_checkpoint(..., g2_content_mode=...)` 写入标签；`restore_ablation_checkpoint` 传不匹配
  mode 抛 `ValueError`；无标签旧 payload 按 radial 恢复成功、按 joint 拒绝。
- **A5 回归**：`tests/test_e5_checkpoint_boundary.py::TestRunnerRecovery` 在默认 radial 下全通过；恢复分支不重写
  `config_used.yaml`（字节级）；`restore_ablation_checkpoint` 对旧 payload 仍成功。

---
依据 `git show HEAD:45c2e85`；未运行测试，以上为契约级审查建议，需实现方在其 CPU 合同测试中落实。

## 协调者最终代码复核纠正（2026-09-27）

本报告是旧源码上的风险清单，不能当作已实现代码的事实：
- 第2条/A3关于radial键变unexpected不成立：最终joint保留全部radial键，只新增W_i/phi1/phi2；radial→joint会缺6键/层，且可能继承非零alpha。
- 第3/4条漏掉严格恢复对键集合的检查。最终joint有独有参数键，现有strict恢复已拒绝跨模式，不需要扩展全局checkpoint格式或放宽恢复。
- 宽松初始化确是实际缺口，已在runner新增 `load_joint_initial_state`：完整joint或完整B7主干才允许；拒绝部分G2、缺失共享键、额外键、张量形状不符，且先校验再复制。旧radial初始化路径不变。
- 配置双枚举、非法组合建议已落实，CPU测试覆盖实际runner配置/初始化链。最终验收见 `log-2026-09-27-joint-content-implementation.md`。
