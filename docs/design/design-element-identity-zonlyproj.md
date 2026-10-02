# 元素表示投影（ZP100）对齐初始化候选：实验设计记录

日期：2026-10-01。状态：**已实施并完成判读**（本轮提示词授权 M1、seed 42、100 轮、`--atom_feat z_only_proj`）。
2026-10-02 用户已选择固定 ZP 用于后续结构研究；当前职责、证据与使用限制见
[元素初始化说明](design-element-initialization.md)。本页继续保留历史实验的设计与原采用标准。
本页记录实施前冻结的实验设计；结果与判读见
[`../logs/log-2026-10-02-eid-zonlyproj.md`](../logs/log-2026-10-02-eid-zonlyproj.md)
（结论：未达采用标准，eDOS oracle 失败率上升 1.04pp 越线；对 Z100 亦未观察到可区分改善）。
本候选是 [`design-element-identity-zonly.md`](design-element-identity-zonly.md) 的 Z100 判读后追问：
在 Z100 的纯原子序号表示上加一个可学习投影，且**与 Z100 共享初始化与随机流**，从而把差异收窄到投影本身。
A100/Z100 是既有对照臂，不重训；旧 B7 只作历史参考。

## 1. 要回答的问题

1. 与 Z100 比：在「共享初值 + 共享随机流 + 共享数据顺序」的对齐初始化下，加一个可学习的
   `Linear(512→512)` 元素投影是否改善 Q1 valid 表现？
2. 与 A100 比：该候选是否达到新 baseline 的工程采用标准？

## 2. 冻结方案（ZP100）

元素输入退化为：

```
atom_src = atom_proj(atom_norm(tok_emb(atom_idx)))
atom_proj = Linear(512 -> 512, bias=True)
```

1. 仍不使用元素性质或常数分支：不实例化、不调用 `AtomFeatureEncoder`、`num_norm` 与 `fuse_proj`；
   `AtomFeatureEncoder(feat='z_only_proj')` 显式抛错，防止误入性质表读取路径。
2. 输入索引与 padding/mask 契约、晶体几何路径、encoder/decoder/谱头/尺度头（H1 eta/gamma）、
   损失、隐藏维度、层数、任务权重与选点规则（valid `balanced_score = 0.5×MAE_edos_median +
   0.5×MAE_phdos_median` 最小）全部保持不变。
3. 其他模式（`legacy3`/`legacy3_const`/`z_only`/`mendeleev24`）的分支语句、模块创建顺序、
   初始化与 checkpoint 加载兼容性逐值不变（训练前检查逐项复核）。
4. 唯一模型改动是上述 `z_only_proj` 分支；对齐初始化流程属于实验协议实现，不改变任何其他架构。

实现入口：`model/transformer.py::Transformer.__init__/forward` 的 `z_only_proj` 分支；
`utils/atom_feature.py::AtomFeatureEncoder` 的拒绝；`run_ablation_experiments.py` 的
`--atom_feat` choices 与受控对齐流程（`utils/zproj_alignment.py`，训练进程与训练前检查共用同一实现）。

## 3. 对齐初始化（协议）

1. **参考重建**：按 Z100 原配置（`output/ablation_m1_eidzonly100_s42/config_used.yaml` 的
   `config` 段）与 seed 42 重建其未训练模型（`z_only` 模式、同一构造路径
   `ConfigBuilder.get_model → basemodel → Transformer`），核对初始参数指纹等于 Z100 manifest
   记录的 `state_hash_z_only_init = 4eda5029…`。不加载 Z100 或其他模型训练好的权重。
2. **共享参数**：候选模型的全部共享参数（`z_only` 的 `state_dict` 键集）逐值复制自该未训练参考模型；
   新增投影 `atom_proj.weight = I(512)`、`atom_proj.bias = 0`（赋值初始化，不消耗随机数）。
   初始前向与 z_only 相同（检查中核对数值一致并记录最大绝对差）。
3. **训练随机状态**：Z100 的训练开始时随机状态 = `setup_ablation_seed(42)` 后、模型构造消耗后的
   状态（DataLoader/`Dos_Dataset` 构造不消耗随机数，构造之后到训练循环之间无随机抽样）。做法：
   参考模型构造结束时保存 python/numpy/torch CPU/torch CUDA 四路随机状态；候选模型构造与
   训练前检查完成后整体恢复该状态；进入训练循环前复核状态摘要哈希未变，否则报错中止。
4. **数据顺序**：训练顺序由 `DistributedSampler(seed=0)` + `set_epoch(epoch)` 决定，只依赖 epoch，
   与模型初始化随机数无关；检查中核对候选与 z_only 管线的 epoch 0/1 样本顺序逐值一致并记录哈希。
5. **对齐后的差别**：仅新增投影及其梯度（投影初值为恒等，初始函数与 Z100 相同；dropout 流、
   数据顺序、优化器与学习率调度一致）。注意投影改变了参数量（+262,656）与优化器状态数量。

## 4. 训练前检查（工具：`tools/eval/eid_zproj_preflight.py`）

- 参考重建与指纹：Z100 manifest 指纹 = 冻结常数；按其记录配置与 seed 42 重建的未训练模型指纹一致；
  runner 构造路径同样复现该指纹；Z100 的 `config_used.yaml` 与 best/latest checkpoint 哈希未漂移。
- 唯一模型改动：`z_only_proj` 恰好多出 `atom_proj.{weight,bias}` 两键；`atom_proj` 为
  `Linear(512→512)` 含偏置；`num_emb_encoder/num_norm/fuse_proj` 不存在；encoder 实际输入
  逐值等于 `atom_proj(atom_norm(tok_emb(atom_idx)))`（z_only 公式同样复核）；构造与前向不读性质表；
  `AtomFeatureEncoder(feat='z_only_proj')` 抛错；可训练参数 70,887,748 = 70,625,092 + 262,656。
- 对齐初始化：与训练进程同一实现跑通；共享参数逐值一致（独立复核）、投影=单位阵/零偏置（独立复核）、
  前向与梯度有限、对齐后前向与 z_only 参考数值一致（记录最大绝对差）、检查后随机状态恢复为 Z100
  训练开始时状态（摘要哈希一致），并有一条独立重放路径（不经 loader）互相印证。
- 数据顺序：候选与 z_only 管线 epoch 0/1 样本顺序逐值一致，记录顺序哈希；sampler 覆盖全部 18,706 条。
- 兼容性：`legacy3`/`legacy3_const`/`mendeleev24` 的 seed-42 初始 `state_dict` 哈希与改前快照逐值一致；
  A100/B100/Z100 checkpoint 各自 `strict=True` 载入自己的模式；`z_only_proj` 显式拒绝三臂训练权重。
- 数据/配方：性质表/split/数据 manifest 哈希与 A100 manifest 记录及文档常数一致；Q1 train 18,706 /
  valid 2,313；网格 128/64、dropout 0.05；计划 CLI 与冻结配方除 `atom_feat`/`tag`/`dropout` 外逐项一致；
  FP32、`skip_test_eval=true`、`init_ckpt=''`、pair_aux 关闭。
- tag `_eidzproj100_s42` 的目录、history、summary、日志、manifest 全部空闲（防 runner 自动续跑）；
  训练进程内的对齐流程同样拒绝已有 `checkpoint_latest.pth`。

溯源：manifest `results/eidzproj100_s42_manifest.json` 记录命令、计划 CLI、数据/划分/性质表指纹、
代码版本与训练相关文件哈希、检查结果、对齐细节（参考指纹、随机状态摘要、样本顺序哈希、数值检查）；
训练启动后 `--verify-run` 核对真实 `config_used.yaml` 与 `output/ablation_m1_eidzproj100_s42/zproj_align.json`
并登记进程 PID；训练后 `--finalize` 补记 best/latest checkpoint SHA-256 与 epoch 身份。
manifest 是溯源记录，不代替检查本身。

## 5. 运行与监控

```bash
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat z_only_proj \
  --skip_test_eval --tag _eidzproj100_s42
```

从零训练（共享初值来自未训练参考，不是任何 checkpoint）。预算约 5.3 GPU 小时（以 A100/Z100
实测约 189 s/轮、峰值约 7.1 GiB 估计）；产物：`output/ablation_m1_eidzproj100_s42/`
（`checkpoint_best.pth`/`checkpoint_latest.pth`/`config_used.yaml`/`zproj_align.json`）、
`results/history_m1_eidzproj100_s42.csv`、`results/train_m1_eidzproj100_s42.log`。
每约 5 轮监控 epoch、训练损失与分量、valid 指标、最佳 epoch、耗时、显存与产物更新；
启动、25/50/75/100 轮及最终判读同步。遇 NaN/Inf、OOM、异常退出、配置错位或意外续跑，
保留记录并报告，不自动改参数、重启或追加训练。

## 6. valid 判读与采用标准

判读工具：`tools/eval/eid_zproj_valid_verdict.py`（独立产物名 `results/eid_zproj_s42_valid_*`，
不覆盖 A100/B100/Z100 任何产物）。只用 Q1 valid，不构建或使用 test；选点规则保持 `balanced_score`
（已知其受原始 MAE 尺度影响，本轮记录该限制，不改规则）。主结果用各自 best，另报第 100 轮；
同 valid 样本、同 oracle/blind 公式；A100/B100/Z100 逐样本结果在身份、样本 ID、配置与指纹核对通过后复用。

报告四项中位 R²（eDOS/phDOS × oracle/blind）、四项失败率（逐材料 R²<0 比例）、学习曲线
（25/35/50/75/100 轮）、best epoch 与 balanced_score、参数量与运行成本；
Δ = ZP100−A100 与 Δ = ZP100−Z100 各 2,000 次逐样本配对 bootstrap 95% 区间；
固定 320 对同组成材料的预测谱差 TV 与对比误差（结构响应诊断，不单独决定采用）。

**工程采用标准（单种子临时 baseline）**：相对 A100，四项中位 R²下降均 <0.02，四项失败率上升均
<1 个百分点，且实现、配置与数值检查全部通过。达标则登记为「单种子临时 baseline」并保留 A100/Z100
历史对照；未达标完整报告、保留 A100、不改门槛、不启动下一轮。结论限于本次配方与初始化，
不声称跨种子稳定；bootstrap 区间与 seed 波动边界如实说明。
