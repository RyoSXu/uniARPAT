# 日志：2026-09-27 — 联合边内容（候选1）实施与CPU合同测试

## 范围

- 任务与假设：按 `docs/design/design-model-upgrade-candidates.md`「2026-09-27实施授权与冻结范围」
  与候选1公式，仅实现联合边内容（`content_mode="radial"|"joint"`）并通过CPU合成合同测试。
  假设性质不变：表达差别／归纳偏置假设，不是已选定根因；本轮不含训练、真实数据、GPU或资源测量。
- 改动的文件或配置（均为授权列表内）：
  - `model/transformer.py`：`PeriodicEdgeMessage` 增加 `content_mode` 与 joint 分支；`Transformer`
    末尾新增 `g2_content_mode="radial"` 关键字参数与组合校验。
  - `utils/experiment_config.py`：`ExperimentConfig.g2_content_mode: str = "radial"`。
  - `run_ablation_experiments.py`：`validate_g2_config` 早失败、yaml 参数写入、`config_used` 记录、
    `--g2_content_mode {radial,joint}`、parser 抽出为 `build_arg_parser()`（机械移动，供合同测试调用）。
  - `tests/test_g2_joint_content.py`（新文件，16 项 CPU 合同）。
  - `tools/ci/check-static.sh`：把 `tests.test_g2_joint_content` 纳入无缓存CPU合同列表。
  - 本日志。`status/index/设计文档` 由协调者维护，未动；默认 YAML 未动；工作区既有改动全部保留
    （含会话期间出现的他人未跟踪日志 `log-2026-09-27-joint-content-flash-review.md`，未触碰）。

## 实现（文件 → 符号 → 调用链）

- `model/transformer.py::PeriodicEdgeMessage`
  - 新符号：类常量 `JOINT_HIDDEN = 256`（固定宽度，无扫描口子）；构造参数
    `content_mode="radial"`（追加在 `r_cut` 后，旧位置参数不变）；joint 分支参数
    `W_i: Linear(D,D)`、`phi1: Linear(3D,256)`、`phi2: Linear(256,D)`（`SiLU` 激活）；
    `_forward_joint`。
  - joint 公式逐条落实：`u_i=W_i h_i`（先算节点 `[B,L,D]` 再按 `edge_dst` gather）、
    `v_j=W_v h_j`、`e=W_g RBF(d)`、`c_ij=phi2(SiLU(phi1(concat(u_i,v_j,e))))`，
    消息乘原 quintic cutoff，复用原 `index_add`、`1/sqrt(clamp(deg,min=1))`、`W_o` 与 `alpha`。
  - radial 路径逐字保留：原 forward 主体、参数键（`W_v/W_g/W_o/alpha` + `rbf_*` 缓冲）和
    构造顺序未改；joint 新参数只在 `content_mode=="joint"` 时注册，因此 radial/off 路径的
    参数键与 RNG 消耗顺序与改动前一致。
  - `alpha=0` 无早退：代码中不存在按 `alpha` 的分支，首步梯度始终到达 `alpha`。
    整批空边沿用 G2a 早返回 `h` 的既有守卫；非空批中无入边行按公式为 `h + alpha*W_o(0)`。
- `model/transformer.py::Transformer.__init__`：签名末尾（`macro_lattice_std` 之后）新增
  `g2_content_mode="radial"`；校验（值域、joint 必须 `use_g2=True`、G1/G2 互斥沿用原错误消息）；
  `encoder.g2_msgs` 构造把 `content_mode` 传入 `PeriodicEdgeMessage`。
- 调用链：CLI `--g2_content_mode` → `ExperimentConfig.from_args`（`utils/experiment_config.py`）
  → `run_ablation_experiments.train_and_eval::validate_g2_config`（在 `os.makedirs(save_dir)`、
  `configs/default.yaml` 读取和 `ConfigBuilder` 之前显式拒绝非法组合）→
  `yaml_cfg['model']['params']['sub_model']['transformer']['g2_content_mode']` →
  `utils/builder.ConfigBuilder.get_model` → `model/model.py::basemodel.__init__` 的
  `Transformer(**sub_model["transformer"])` → 实际模型；同值写入 `config_used.yaml` 的
  `cli.g2_content_mode` 与 `config.model.params.sub_model.transformer.g2_content_mode`。
- 非法组合（joint 无 `--use_g2`、joint+G1、未知模式、`--use_g1 --use_g2`）在 runner 顶层
  `ValueError` 早失败，不静默回退 radial；模型层同规则独立校验。旧 G2/B7 checkpoint 的
  `strict=True` 恢复原则未动（`utils/b7_cif_inference.py::load_b7_model` 未改），未新增
  通用 warmstart 框架。

## 证据

- 命令（CPU，`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` 限制线程）：
  - `python3 -m unittest tests.test_g2_joint_content`：16/16 通过（约 1.2s）。
  - `python3 -m unittest tests.test_e5_checkpoint_boundary`：14/14 通过（runner 回归合同）。
  - `python3 -m unittest tests.test_g2_periodic_edges.TestG2PeriodicEdges.<15项合成合同>`：
    15/15 通过（`test_q1_smoke`、`test_production_smoke` 为真实数据测试，按禁令未运行）。
  - `bash tools/ci/check-static.sh`：ruff + compileall + 119/119 无缓存CPU合同通过
    （含新纳入的 `tests.test_g2_joint_content`）。
- 测试辨别力（`tests/test_g2_joint_content.py`，全部合成输入，无 Q1 缓存／真实标签／GPU）：
  - 键合同：radial 默认与显式模式 state dict 键集一致、严格 roundtrip（radial 与 joint 各一）；
    默认构造无 `W_i/phi1/phi2` 键；关闭分支无 `g2_msgs`；radial 参数键集硬编码为 7 参数 + 2 缓冲。
  - 参数解析数：`PeriodicEdgeMessage(512, content_mode="joint")` 实测 1,346,305，
    与冻结解析数一致；radial 为 558,593；差值 787,712 恰为新分支；`phi1/phi2` 宽度断言 256，
    传入宽度参数报 `TypeError`（无扫描口子）。
  - 加载合同：B7 共享权重（`use_g2=False` state dict）→ joint，缺失键集合**恰好**等于
    `g2_msgs.0` 的 radial+branch 全集；→ radial 缺失恰好 radial 键集；radial → joint 缺失
    恰好 `W_i/phi1/phi2` 六键；unexpected 均为空；共享键逐张量 `torch.equal`；
    严格加载 B7 字典仍抛 `RuntimeError`（未放宽全局 strict）。
  - alpha=0 eval 等价：base/radial/joint（含经 radial 二级加载的 joint）在
    `edos/phdos/eta`（H1 eta/gamma）上 `torch.equal` 逐值一致。
  - 非 alpha=0 的行为合同（防恒等假通过）：joint 消息对接收态敏感（同发送态、同距离下改动
    `h_i` 改变消息，配套 radial 对照在同一 fixture 上接收态不变）、发送态敏感、排列等变、
    padding 不影响有效原子（边端点断言 + 输出逐值隔离，无边行核对 `h+alpha*W_o(0)`）、
    多镜像语义（k 个同距镜像相对单边按 `sqrt(k)` 缩放、两个不同发送端先求和再除 `sqrt(2)`，
    `W_o` 置单位阵直读聚合）、空边整批恒等、公式参考实现（经 `g2_rbf_features`/
    `g2_quintic_cutoff` 独立路径）在多镜像边上与模块输出差 < 1e-5。
  - 梯度合同：非零 alpha 下接收／发送／边距离梯度有限且非零，`W_i/phi1/phi2/W_v/W_g/W_o`
    梯度有限非零；alpha=0 首步 `alpha` 梯度有限非零且 joint 分支留在图中（grad 非 None）；
    单步 SGD 后 alpha 离开零（全套至多这一步 optimizer）；手动置 alpha=0.7 后内部参数收到信号。
    若实现按 alpha=0 早退、去掉接收路径或禁用 joint，上述断言分别失败。
  - 接线合同：CLI choices 拒绝未知值（exit 2）、默认 `radial`；runner 非法组合在
    `ConfigBuilder` 构造与数据入口之前抛 `ValueError`（mock 数据入口断言未被触达、
    目录未创建）；joint 传递链经真实 `ConfigBuilder→basemodel→Transformer(**params)` 边界
    （仅把 Transformer 构造替换为同参数小模型 spy、`epochs=0` 零训练步、mock 数据入口）到达
    实际模型 joint 分支，并在 `config_used.yaml` 两处记录 `g2_content_mode: joint`。

## 结论

- 状态：closed（代码与合同测试已交付；训练未授权，不在本日志结论范围）。
- 原因：候选1按冻结公式实现，radial 路径键与数值行为保持（旧 G2a 合成合同 15/15、
  E5 runner 合同 14/14 复核通过）；新合同 16/16、CI 门禁 119/119 通过。
  这只证明实现与合同一致，不构成任何精度或机制结论。

## 交接

- 下一项关卡工作：协调者复核并更新 `status/index/设计文档`；若进入训练，仍须先冻结
  训练初始化配对、判定阈值与资源停止线（设计文档缺口分层已列），另行批准。
- 对 status、backlog 和 decisions 的更新：由协调者处理。
- 未验证项（明确边界）：
  1. 显存／耗时增量未测（设计预算中的 +27.3%／+16.5% 是 G2a 实测参照，不是本提案承诺）；
  2. `alpha` 是否离开零、内容 MLP 梯度量级、valid 全指标均须训练后才有；
  3. 真实 Q1 数据路径未触达（`test_q1_smoke`、`test_production_smoke` 未运行，需在允许
     读数据的回归中由协调者安排）；
  4. config_used 传递链验证用小模型 spy 替换了 512 维生产构造（同一 kwargs 边界），生产
     全尺寸构造链未在本合同内实例化；
  5. 语义备注：整批空边沿用 G2a 早返回 `h`，而非 `h + alpha*W_o(0)`（与非空批的无边行
     语义不同），这是逐字保留原模块行为的结果，已在测试中固化并在此标明。

## 协调者验收与补齐（2026-09-27）

已复核MiMo实现及DeepSeek V4.1 Flash基线审查。Flash提示的宽松初始化风险成立，但“radial键变unexpected/严格恢复仅检查形状”解释不成立，纠正保留在审查日志中。

修改与调用链：
- `model/transformer.py::PeriodicEdgeMessage._forward_joint` 实现接收/发送/径向联合内容；`Transformer.__init__` 传递模式并拒绝非法组合。固定隐藏宽度256，默认radial。
- `utils/experiment_config.py::ExperimentConfig.g2_content_mode` → runner `build_arg_parser` / `validate_g2_config` → 模型构造及config_used两处配置。
- 协调者补充runner `load_joint_initial_state`，由 `train_and_eval` 的joint初始化路径调用：只接受完整joint或完整B7主干；缺失集合须为空或恰好等于全部G2分支键。键与形状在复制前检查；正常checkpoint恢复仍strict。
- `tests/test_g2_joint_content.py` 新增坏权重不污染参数测试，并通过实际runner mock链验证初始化调用；`tools/ci/check-static.sh` 纳入新模块。共享status/index/design已同步。

实际执行（CPU，CUDA不可见，合成输入）：
1. `python3 -m unittest tests.test_g2_joint_content -q`：17/17通过。
2. `bash tools/ci/check-static.sh`：lint/编译通过，120/120合同测试通过，49.442秒；含既有合成优化器步测试，不涉及真实材料训练。
3. 对未跟踪新测试文件另行执行ruff与py_compile，均通过。
4. 与改动前保存的独立合成快照比较：B7关闭G2、radial alpha=0、radial alpha=0.37三种设置，参数键/同seed初始化/严格载入/全部输出逐值一致。临时证据位于 `/tmp/uniarpat-joint-implementation-fpjtasno/`，非正式实验或生产checkpoint。
5. 写前哈希快照比对：变化局限于授权生产文件和协调者文档，原R1/R2/R3及纠正报告未变。未commit/push，未改依赖、数据、缓存或真实checkpoint。

本段17/120项是最终验收计数，取代执行者早期16/119项。代码与CPU合同小单元完成；精度、实际GPU显存/耗时、生产数据路径未验证。后续先冻结初始化配对、资源停止线与全体valid验收设计，再请求对应资源核查/pilot授权；不会自动进入训练。
