# 日志：2026-09-28 — 联合边内容 Q1 valid-only 三臂 pilot

## 范围

- 任务：执行已冻结的候选1阶段 B，判断联合边内容整体是否值得进入复现或 M1×35 确认。
- 三臂：`_jcctl`（G2 关）、`_jcrad`（径向 G2）、`_jcjoint`（联合内容 G2）。
- 共同配方：Q1、M1×10、seed 42、batch 32、FP32、非分桶、SumNorm、H1 eta、学习率
  `5e-5`、dropout `0.05`，均从 B7 epoch 33 的同一 checkpoint 初始化，并在载入后重置
  RNG。训练与判决均未读取 test。
- 本实验只判断联合内容函数组合包；不能把结果单独归因于接收端条件或新增参数容量。

## 执行与资源

- 单张 `Tesla V100-SXM2-32GB` 串行执行，2026-09-27 22:55:55 至 2026-09-28
  01:00:56，共约 2 小时 5 分钟。三臂都完成 10 轮，无恢复、OOM、NaN 或 Inf；日志逐臂记录
  `Test loader and automatic test evaluation skipped by request`。
- 完整训练轮平均耗时：control `189.91 s`、radial `241.36 s`、joint `257.31 s`；
  radial/control `1.271x`、joint/control `1.355x`、joint/radial `1.066x`。
- 正式判决固定使用三臂 epoch 10 的 `checkpoint_latest.pth`，全体 Q1 valid 2,313 条、同序
  `mpid` 配对、2,000 次 bootstrap；区间只报告，点估计决定结论。

## 主判决

`joint - control`：

| 指标 | control 中位 R² | joint 中位 R² | Δ中位 R² | 95% 配对区间 | Δ失败率 |
|---|---:|---:|---:|---:|---:|
| eDOS oracle | 0.51734 | 0.51853 | +0.00119 | [−0.00417, +0.00338] | −0.086 pt |
| eDOS blind | 0.47920 | 0.47865 | **−0.00055** | [−0.00458, +0.00359] | +0.043 pt |
| phDOS oracle | 0.74265 | 0.74276 | +0.00012 | [−0.00474, +0.00521] | +0.000 pt |
| phDOS blind | 0.73166 | 0.73469 | +0.00303 | [−0.00403, +0.00618] | +0.303 pt |

- 主门槛要求 blind eDOS `Δmedian ≥ +0.02`；实测为 `−0.00055`，未通过。
- 所有四类指标均满足项目平局线 `|Δmed| < 0.02` 且 `|Δfail| < 1 pt`，保护项没有越线。
- 正式裁决为 **tie**。按预注册行动表，候选1 park，不做 seed 43/44 复现，不进入 M1×35，
  不改 B7 默认。

辅助比较也为平局：

- `joint - radial`：eDOS oracle/blind `−0.00172/−0.00131`，phDOS oracle/blind
  `+0.00100/+0.00146`；没有“只胜 radial”的情况。
- `radial - control`：eDOS oracle/blind `+0.00291/+0.00076`，phDOS oracle/blind
  `−0.00088/+0.00158`；新代码线复现了历史 G2a 的平局，而非异常越线。

## 辅助机制读出

- 高粗糙度 205 条中，joint-control eDOS oracle 为 `+0.01678`，但 blind 只有 `+0.00336`；
  phDOS blind 为 `−0.01735`。这是局部、互不一致的读出，不能推翻整体判决。
- 谱支持距离四分位没有单调或跨线的 joint 优势。
- 冻结 320 个同组成材料对已经用原设计公式补齐：真实谱差 TV 中位数为 `0.26975`；
  control/radial/joint 的预测谱差 TV 只有 `0.01918/0.01829/0.01871`，谱差误差 TV 为
  `0.26731/0.26714/0.26918`。joint 相对 control 的预测 TV `−0.00047`、谱差误差
  `+0.00187`，没有修复结构条件响应过弱。
- 联合内容代码默认关闭，不能由本结果推出结构信息无用；结论只是否定该内容函数组合在本预算和
  配方下的采用价值。

## 产物与验证

- 主结果：`results/joint_content_pilot_q1_valid.json`。
- 三臂逐样本：`results/joint_content_pilot_q1_valid_{control,radial,joint}_samples.csv`。
- 三组比较：`results/joint_content_pilot_q1_valid_{joint_vs_control,joint_vs_radial,radial_vs_control}.csv`。
- 辅助结果：`results/joint_content_pilot_q1_valid_auxiliary_readouts.csv` 与
  `results/joint_content_pilot_q1_valid_composition_pairs.csv`（320×3 行）。
- 完整训练输出：`results/joint_content_stage_b_training.log`；训练历史为
  `results/history_m1_{jcctl,jcrad,jcjoint}.csv`；检查点保留在对应 `output/` 目录。
- OpenCode Go / DeepSeek V4.1 Flash 补齐预测 shape 临时侧车与同组成对读出；协调者发现并修复
  最后一个小 batch 的拼接边界，增加 32+9 合同。`tests.test_joint_content_pilot` 最终 43 项通过；
  仓库 `tools/ci/check-static.sh` 最终 163 项通过，Ruff、`py_compile` 与 `git diff --check`
  通过。正式重算与首次主判决的点估计最大差约
  `8.4e-7`，裁决及所有门槛不变。

## 结论与交接

- 状态：**park**。
- 候选1关闭；不自动转入谱监督、角信息、数据扩充或新的 G2 拆分诊断。
- 当前没有已授权的下一项模型训练。后续若继续模型升级，应重新选择一个能连接到独立 valid 检验的
  具体缺陷，先写设计和停止条件，再申请执行。
