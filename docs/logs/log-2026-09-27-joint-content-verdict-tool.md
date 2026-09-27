# 日志：2026-09-27 — 候选1 valid-only 三臂判决工具与 CPU 合同测试

## 范围

- 任务与假设：为已冻结的“联合边内容（候选1）Q1 valid-only M1×10 三臂 pilot”实现阶段 B 前的
  **判决工具** `tools/eval/joint_content_pilot_verdict.py`，并用 CPU 合成合同测试锁定其预检、
  纯裁决与 CLI 边界。假设：阶段 B 训练完成后，只需三臂 epoch10 `checkpoint_latest.pth` 与
  `config_used.yaml`，即可在不读取 test、不重造指标语义的前提下给出 `joint vs control` 主裁决，
  并附带 `joint vs radial`、`radial vs control` 与辅助读出。
- 本轮不实现、不运行：三臂 10 epoch pilot、GPU、Q1 数据加载器或前向/训练、test 评估、
  缓存重建、判决工具的正常 CLI。执行者为确认冻结辅助表的列契约，只读查看了既有
  `results/edos_spectral_support_*` 的表头、行数和 split 计数；未运行数据加载或模型计算。
- 改动的文件：
  - 新增 `tools/eval/joint_content_pilot_verdict.py`；
  - 修改 `tests/test_joint_content_pilot.py`（新增判决合同测试，原 RNG／资源门合同保留）；
  - 新增本日志 `docs/logs/log-2026-09-27-joint-content-verdict-tool.md`。
  - 未修改 `tools/ci/check-static.sh`：`tests.test_joint_content_pilot` 已在该脚本的固定
    unittest 清单中（第 31 行），本轮测试位于同一模块内，不会重复登记。

## 证据

### 复用与调用链（未改既有指标工具）

判决工具严格复用既有语义，不另造 R² 或尺度：

- 模块重建与 Q1 valid 前向、逐样本 `oracle/blind` R²：
  `run_verdict` → `tools/eval/edos_error_attribution.py::run_audit`，固定
  `split="valid"`、`expected_epoch=10`、`verify_reference_r2=False`；列名沿用
  `r2_edos_oracle_unmasked` / `r2_edos_blind_unmasked` /
  `r2_phdos_oracle_unmasked` / `r2_phdos_blind_unmasked`。
- 配对 bootstrap 区间：
  `paired_metric_summary` → `tools/eval/edos_slope_pilot_verdict.py::paired_bootstrap_interval`，
  固定 `BOOTSTRAP_SEED=20260923`，`--bootstrap` 默认 2000。
- 冻结标签与键集：`TRAIN_ROUGHNESS_P90=0.351600`（直接导入）、
  `ARMS`/`expected_g2_keys`（直接导入 `joint_content_resource_gate`）。
- 失败定义沿用项目口径 `R² < 0`；`tie` 线沿用 `|Δmed|<0.02` 且 `|Δfail|<0.01`（0.01=1pt）。

### 主要符号

- 预检（纯函数，CPU 可测）：
  - `normalize_path`、`load_arm_config`；
  - `validate_arm_config`：逐臂校验 `model=M1`、`epochs=10`、`use_g2`、
    `batch_size=32`、`seed=42`、`norm=sumnorm`、`scale_mode=eta`、
    `g2_content_mode`、`skip_test_eval=true`、`use_amp=false`、
    `use_bucket_batch=false`、`reset_rng_after_init=true`、`init_ckpt` 规范化等于 B7 路径
    `output/ablation_m1_e9ctl/checkpoint_best.pth`，并要求生效 `config` 子字典与 `cli` 一致；
  - `strip_arm_specific_keys` 与 `validate_cross_arm_configs`：三臂配置剥离
    `use_g2`/`g2_content_mode` 后必须逐一相等（`tag` 未写入 `config_used.yaml`，不参与比较，
    作为 JSON 缺口记录）；
  - `validate_checkpoint`：`epoch==10`、`model_name=M1`、`use_amp=false`，且
    `encoder.g2_msgs.*` 键集恰等于该臂期望（control 空集）；
  - `load_and_validate_checkpoint`：要求文件名为 `checkpoint_latest.pth`（判决固定 latest，
    拒绝 best/其他名），随后 `torch.load` 并调用 `validate_checkpoint`；
  - `validate_arm_artifact_paths`：固定 `_jcctl/_jcrad/_jcjoint` 三个唯一输出目录，并要求
    checkpoint 与 `config_used.yaml` 为同目录产物；正式判决前再次核对 B7 checkpoint 哈希；
  - `validate_sample_frames`：每臂 n=2313、指标列齐全、`mpid` 唯一且三臂逐值同序，并提供
    与冻结 `valid_index.npy` 的同序校验入口。
- 配对与裁决：
  - `paired_metric_summary`：reference/candidate 中位 R²、失败率、差值及 2000 次配对区间；
    非有限值改为 NaN 以便裁决报 `technical_incomplete`，不臆造数字；
  - `build_comparison`：按四指标生成一组比较；`comparison_frame` 输出 CSV（失败率以 pp 表示）；
  - `_win_conditions`：7 条预注册 win 条件，边界严格；
  - `evaluate_joint_verdict`：**只看 `joint vs control` 点估计**，返回 `win` / `tie` /
    `degraded_or_guard_failed` / `primary_insufficient` / `technical_incomplete`；
  - `win_conditions_met`、`auxiliary_radial_classification`：`only_beats_radial` 仅辅助，
    不改变主裁决、不启动其他路线。
- 辅助读出：`build_auxiliary_readouts`
  - 高粗糙度 `train p90=0.351600` 分组（high/other）；
  - 谱支持距离四分位（复用 `results/edos_spectral_support_q1_train_valid_samples.csv`
    的 `split=valid`、`train_support_quartile`，并校验 mpid 同序）；
  - 320 同组成 pair 表：现有审计工具只持久化逐样本 R²/描述列，不保存逐样本预测谱，
    无法无歧义复算各臂预测谱差 TV；按约定在 JSON 记录缺口，不自创新定义。
- 输出与原子性：`formal_paths` 定义 3 份逐样本 CSV、3 份配对比较 CSV、1 份辅助读出 CSV、
  1 份 JSON；`run_verdict` 先在 `output_prefix.parent` 下的临时目录完成全部计算与文件生成，
  整套成功后才用 `os.replace` 逐个落盘；任一预检或计算失败时临时目录被清理，正式文件不出现。

### CLI 边界

- `build_arg_parser` 只提供 `--control/--radial/--joint-checkpoint`、
  `--control/--radial/--joint-config`、`--output-prefix`、`--device`（auto/cpu/cuda）、
  `--bootstrap`（默认 2000）。无 `--split`、无 test 入口、无 `--force`。

### 实际执行的验证

- `python3 -m unittest tests.test_joint_content_pilot`：协调者终审后 **38 项通过**（原合同 +
  新增判决合同）。
- `python3 -m ruff check tools/eval/joint_content_pilot_verdict.py tests/test_joint_content_pilot.py`：
  通过（仓库 `ruff.toml` 仅选 `E9/F63/F7/F82`）。
- `python3 -m py_compile` 两个文件：通过。
- `bash tools/ci/check-static.sh`：协调者终审后 **158 项通过**（含新增判决合同；ruff 致命错误、受跟踪
  Python 编译、无缓存 CPU 合同测试全部通过）。
- 新测试覆盖：win 各边界等号（blind eDOS Δmed=+0.02、oracle eDOS Δmed=0、
  phDOS Δmed=−0.02 均通过）、fail 恰为 +0.01 判失败、`tie`/`degraded_or_guard_failed`/
  `primary_insufficient`/`technical_incomplete` 状态、样本错序/n≠2313/期望 ID 不符、
  epoch 与 G2 键集、模式与各开关拒绝、跨臂非允许字段差异拒绝、`checkpoint_latest.pth`
  文件名与唯一 tag 目录合同、bootstrap 固定种子可复现、辅助读出缺口记录且不改变主裁决、
  CLI 无 split/test，以及 mock 三臂全流程成功/第三臂失败的正式产物边界。
- 未执行：判决工具正常 CLI、任何 GPU、真实 Q1 train/valid/test 读取、前向/训练、缓存重建、
  三臂 10 epoch pilot。以上均为本轮明确禁止项。

## 结论

- 状态：候选1阶段 B 判决工具与其 CPU 合同测试已实现；训练仍未授权。
- 原因：工具在代码层面锁定“epoch10 latest + Q1 valid-only + 点估计裁决 + 区间只报告 +
  辅助只解释”，并复用既有 `run_audit` 与 `paired_bootstrap_interval`，未引入新的 R²/尺度语义，
  也未添加新的科学阈值。是否启动阶段 B pilot 仍取决于阶段 A 资源报告与用户另行批准。

## 交接

- 下一项关卡工作：阶段 A 已于同日通过；由用户决定是否批准阶段 B 三臂 10 epoch pilot；
  本轮不启动。
- 已知边界：`config_used.yaml` 不记录 `tag`，因此配置字典本身无法比较 tag；协调者终审改为用
  checkpoint/config 的固定父目录 `_jcctl/_jcrad/_jcjoint` 补足该合同。320 同组成 pair 的逐臂
  预测谱差读出因缺少逐样本预测谱而在 JSON 记为缺口。
- 协调者已在阶段 A 完成后更新 `status.md` 与 `index.md`；本任务不产生新的科学决策，
  `decisions.md` 不变。

## 2026-09-28 后续补齐与正式执行

- 阶段 B 获批并完成后，首次正式判决为 `tie`。随后按冻结设计补齐 320 个同组成材料对读出：
  `run_audit` 增加默认关闭的 SumNorm 预测 shape 临时侧车，正式工具复用
  `g2_structure_path_audit.py::contrast_metrics`，输出 `*_composition_pairs.csv` 并在辅助 CSV 中
  汇总；该读出不参与主裁决。
- OpenCode Go / DeepSeek V4.1 Flash 完成初稿；协调者发现最后一个 9 样本 batch 的数组拼接边界，
  改为显式拼接可变 batch，并新增 32+9 合同。专项测试由 38 项增至 **43 项通过**。
- 正式 valid 判决重跑后仍为 `tie`，主点估计与首次运行的最大差约 `8.4e-7`；JSON 中辅助缺口已
  清空，320×3 行配对表完整落盘。最终实验事实见
  `logs/log-2026-09-28-joint-content-pilot.md`。
