# 日志：2026-09-27 — 候选1联合边内容下一阶段实验设计（只读核查 + 设计）

## 范围

- **任务与假设：**在马尚酱指定的只读核查后，为候选1（联合接收/发送/径向边内容）产出下一阶段
  实验设计：先资源可运行性核查，后 Q1 valid-only M1×10 三臂 pilot。假设性质不变：
  表达差别／归纳偏置假设，不是已选定根因。
- **本轮只读动作：**阅读 `AGENTS.md`、`docs/status.md`、`docs/index.md`、`docs/workflow.md`、
  `docs/glossary.md`、`docs/decisions.md`，候选取舍草案、实施与兼容性日志、G2a 预检／阈值更正／
  pilot 日志、eDOS slope-loss pilot 日志，`run_ablation_experiments.py` 的初始化/种子/训练/
  checkpoint 选择/`skip_test_eval` 调用链，`utils/experiment_config.py`、`utils/builder.py`、
  `utils/ablation_checkpoint.py`、`model/transformer.py::PeriodicEdgeMessage`，以及
  `tools/eval/` 下既有资源门禁与判决工具。
- **改动的文件：**仅新增
  `docs/design/design-joint-content-pilot.md` 与本文。未改 `status.md`、`index.md`、
  `decisions.md`、业务代码、测试、结果、`output/`、数据或依赖；工作区既有改动全部保留；
  未执行 git commit/push/checkout/switch。

## 证据（只读事实）

- **runner 种子与 RNG：**`setup_ablation_seed`（`run_ablation_experiments.py` 第 22–34 行）只在
  `train_and_eval` 开头调用一次（第 191 行），早于 `ConfigBuilder`（第 290 行）与
  `builder.get_model()`（第 296 行）；模型构造后、`init_ckpt` 载入后均**无** RNG 重置。
- **数据顺序：**`utils/builder.py::get_sampler` 训练集为
  `DistributedSampler(..., shuffle=True, seed=0)`；runner 每 epoch 调 `sampler.set_epoch(epoch)`
  （第 415–419 行）。其顺序与全局 RNG 无关。
- **G2 代码面：**`--g2_content_mode {radial,joint}`、`--use_g2`、`--init_ckpt`、`--skip_test_eval`
  均已实现；joint 必须显式配合 `--use_g2`（`validate_g2_config`，第 64–80 行）。`init_ckpt`
  只消费 `ck['model']`（第 304–312 行）。
- **B7 checkpoint：**`output/ablation_m1_e9ctl/checkpoint_best.pth` SHA-256
  `cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40`，`epoch=33`、
  `model_name=M1`、`seed=42`、无 `use_amp` 键（FP32 旧格式），含 `optimizer`/`best_val_score`
  键（`init_ckpt` 不读），205 个 state dict 键、无 G2 键。配套旧 `config_used.yaml` 生成于
  G2 开关加入前，没有 `use_g2` 字段；不能写成配置显式 false。
- **历史臂：**`output/ablation_m1_g2ctl/checkpoint_best.pth`（ep10，205 键）与
  `output/ablation_m1_g2edge/checkpoint_best.pth`（ep10，259 键，含径向 G2）存在，但都不是从
  B7 初始化，只能作背景，不能作因果对照。
- **冻结证据文件：**`results/edos_spectral_support_q1_valid_pairs.csv`（320 对）、
  `results/edos_error_attribution_q1_valid_samples.csv`、`results/g2_edge_audit_q1.csv`、
  `results/c2_1b_valid_samples.csv`、`data/train4ARPAT/manifest.json`（valid n=2313）均存在。
- **历史资源事实：**G2a V100 batch 32 无 OOM，稳态单步耗时 +29.5%，pilot 平均每轮 +27.3%、
  峰值显存 +16.5%（阈值更正日志）。这些是成本报告，不是自动淘汰线。

## 设计结论（写入 `design-joint-content-pilot.md`）

- **项目决策：**判断联合内容函数整体是否值得进入 M1×35 确认；结果不得解释为单独证明接收端
  条件或参数容量是根因。
- **两阶段：**先资源单步可运行性核查，通过后才进入 Q1 valid-only M1×10 pilot。
- **资源门禁：**只报告时间、峰值显存及相对 B7/G2a 比值；硬停止仅 OOM、NaN/Inf、固定 batch 32
  无法完成完整单步（资源不可运行），或预测超用户批准预算（**暂停请用户决定，不自动 park**）；
  不擅设成本淘汰线。
- **三臂同代码：**`_jcctl`（G2 关）/`_jcrad`（径向 G2a）/`_jcjoint`（joint），均从 B7 ep33
  warm-start；给出准确命令草案并标注未授权执行。
- **预飞冻结：**checkpoint 哈希/epoch/config；三臂共享键逐值相同；G2 alpha 全零；joint/radial
  新增键集合符合预期；optimizer 无恢复状态。
- **随机公平：**把“`init_ckpt` 载入后、创建训练迭代器/首次前向前重置 Python/NumPy/Torch/CUDA
  RNG”列为开工前代码前置（另需实现 + CPU 合同测试，本任务不改代码）；`DistributedSampler`
  每 epoch `set_epoch` 保持；明确相同 seed 本身不足。
- **判决口径：**正式用三臂 epoch10 `checkpoint_latest.pth`；全体 Q1 valid n=2313、同序配对、
  2000 次 bootstrap、不读 test；点估计裁决，区间只报告。
- **后续顺序（推荐）：**若 win，先另行设计 seed 43/44 的 M1×10 复现，两个 seed 均满足完整
  joint vs control win 判据才晋级 M1×35 确认；复现阶段只跑 control/joint，不再重复 radial；
  本轮不授权。
- **缺口：**RNG 前置、`tools/eval/joint_content_resource_gate.py`、
  `tools/eval/joint_content_pilot_verdict.py` 及对应 CPU 合同测试／CI 均需另行实现。

## 结论

- **状态：pending（待用户批准）。**设计已冻结为一个推荐方案，未执行任何实验。
- **原因：**训练、GPU、真实数据前向、缓存重建、test 评估均未授权，本轮只做只读核查与设计。

## 未执行事项（明确边界）

- 未运行 `run_ablation_experiments.py`；未启动任何 `_jcctl/_jcrad/_jcjoint` 训练或资源门禁。
- 未运行真实数据前向、未做 GPU 测量、未重建缓存、未读取或评估 test。
- 未实现 RNG 前置、资源工具或判决工具（仅在设计中列出）。
- 文中所有命令均为草案，不是已执行记录。

## 交接

- **协调者复核修正：**旧 B7 配置没有 `use_g2` 字段；修正参数张数、seed 复现条件、章节引用，
  并明确阶段 A 单次冷启动时间不等于稳态成本。阶段 A 结束后必须报告成本并再次由用户决定是否
  批准阶段 B，成本本身不是自动淘汰线。
- **下一项关卡工作：**用户审阅本设计并决定是否批准 RNG 前置、资源工具实现与阶段 A 核查；
  批准前不进入任何训练或 GPU 作业。
- **对 status/backlog/decisions 的更新：**本任务按限制未改共享页面，由协调者处理；本轮不产生
  可固化的科学决策。

## 结束自检

- 只新增/修改两个允许文件：`docs/design/design-joint-content-pilot.md`、本文。
- 文中引用的路径均已核对存在；Markdown 结构正常；未写入任何“已执行”的计划命令。

协调者随后复核并修正文中两项事实／逻辑，且按 `docs/workflow.md` 更新共享 `status.md` 与
`index.md`：旧 B7 配置没有 `use_g2` 字段；seed 43/44 的二臂复现不再要求未重跑的 radial 条件。
此外澄清参数张数、阶段 A 冷启动计时边界与阶段 B 必须另行批准。上述共享页更新不属于 OpenCode
执行者的两文件写入范围。
