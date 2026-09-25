# 项目状态与计划

更新日期：2026-09-25。此页是唯一的整体计划：记录当前基线、已完成工作、当前关卡、待办顺序和
阻塞项。任务细节见 `logs/` 与 `results/`，固化规则见 `decisions.md`。

## 当前基线与默认方案

- **B7 `_e9ctl`：**Q1、M1×35、最佳 epoch 33；oracle 测试集 eDOS 0.518 / 5.73%，phDOS
  0.741 / 3.50%，Cv 0.30。
- **数据与训练：**Q1 为 18,706 / 2,313 / 2,287；E0/P0 网格、SumNorm KL/W1/Huber、H1
  eta/gamma 盲推理缩放和 0.05 dropout。pre-Q 仅作归档，不能直接对照。

## 已完成

- Q1 清洁数据池、Z0 价电子审计和 H1 盲推理闭环已定稿。
- B7 是唯一的当前对照；完整 blind 指标见 `decisions.md`。
- G1 周期图编码器、E9-Q1 坐标 MLP、E9-Q2 Fourier 和 L3 损失消融均已按平局规则搁置或关闭；
  不得无新假设重跑。
- **R1a：逐点读出头技术通过。**Q1 M1×10 对照中 eDOS/phDOS 变化为 −0.0102/−0.0098，失败率为 +0.22/0.00pt，均在平局线内，单轮耗时 −2.7%，显存 −2.0%，无 NaN/Inf；作为非默认载体准入 R1b。详细见 `logs/log-2026-09-18-r1a-pointwise-readout.md`。
- **R1b：坐标生成 query 实验技术失败并 park。**Q1 M1×10 对照中 `_r1bcoord` 相对 `_r1bctl` 的 eDOS/phDOS 中位 R² 为 −0.0546/−0.0359（均劣于 −0.02 平局下限），失败率为 −0.09/+0.44pt，Cv MAE 恶化 0.1596；标量 MLP 坐标嵌入严重限制了 cross-attention query 的特化表达力；代码默认关闭，下游网格/窗口/bin 实验不启动。详细见 `logs/log-2026-09-19-r1b-coordinate-query.md`。
- **C5：token 级 decoder MoE 已 park。**详细证据见 `logs/log-2026-09-18-c5-token-moe.md`。
- **G2a：周期多镜像边条件消息成对 pilot 完成并 park。**Q1 M1×10 成对对照中，`_g2edge` 相对 `_g2ctl` 的 eDOS/phDOS Oracle 中位 R² 为 −0.0066/−0.0048，失败率为 +0.22/+0.44pt；Blind 中位 R² 为 −0.0024/−0.0033，失败率为 +0.26/+0.57pt，Cv MAE 恶化 0.0104；单轮耗时 +27.3%（1.273x），峰值显存 +16.5%（无 OOM）。双任务指标均落在平局线内，未出现正向跨线 win；代码保持默认关闭，不进入 35-epoch 确认，不扫描 cutoff、层数或超参。详细见 `logs/log-2026-09-20-g2a-pilot.md`。
- **结构信号诊断完成。**Q1 test 的 176 个同约化化学式、但缓存结构表示不同的组（402 样本）中，
  真实谱形组内中位总变差（TV）为 eDOS 0.252、phDOS 0.320；组内谱形离散度与 B7 组中位 R²呈负相关
  （−0.507/−0.650）。约化组成不足以唯一决定谱形，G2a 平局不等于结构无用。详细见
  `logs/log-2026-09-20-结构信号诊断.md`。
- **C2.1b：验证集损失归因审计完成并 park。**B7 epoch 33 的 Q1 valid 2,313 条中，eDOS/phDOS
  主谱形梯度没有负冲突（encoder/decoder 中位余弦 0.074/0.009）；eta 梯度未压制主任务。高熵 phDOS
  更难，但已承受更大当前损失，不能推出有方向的改权重机制。默认 SumNorm KL/W1/Huber 与 H1 不变，
  不启动 pilot 或扫描。详细见 `logs/log-2026-09-21-c2-1b-loss-attribution.md`。

## 当前关卡／等待执行

- **R1 完成：**R1a 技术通过（非默认载体），R1b 坐标 query 技术失败并 park；依赖连续坐标 query 的网格/窗口/bin 数据表示实验按设计搁置。
- **E10 已完成并 park。**Q1 M1×10 中，宏观状态相对对照的 eDOS/phDOS 中位 R² 为
  −0.0060/−0.0058，phDOS 失败率 +0.74pt；无 accuracy win，不进入 35 epoch。
- **G2a 已完成并 park。**资源实测耗时 +27.3%~+29.5%、显存 +16.5%；Q1 M1×10 成对 pilot 双任务指标均在平局线内（Δmed 约为 −0.007/−0.005），无 accuracy win，代码默认关闭。
- **R2a：共享 decoder 6→3 层 pilot 已作为低成本载体通过。**Oracle 的 eDOS/phDOS 中位 R²变化为
  −0.0164/−0.0011，失败率 +0.48/+0.35pt，均未越过负向平局线；平均每轮耗时 −17.9%、峰值显存
  −18.8%。它不是 accuracy win，B7 6 层仍是参考与默认；R2a 3 层仅准入后续原子加性 phDOS 读出。
  详细见 `logs/log-2026-09-20-r2a-pilot.md`。
- **R2b：固定 P0 原子加性 phDOS 读出已 park。**相对 R2a `_r2a3`，phDOS Oracle 中位 R²
  下降 0.0328（0.7209→0.6881），越过 −0.02 保护下界；虽参数 −52.0%、每轮耗时 −22.6%、峰值显存
  −17.6%，仍无 accuracy win。默认关闭，不确认、不扫描。详细见 `logs/log-2026-09-21-r2b-pilot.md`。
- **C4 AMP：技术通过。**`--use_amp` 默认关闭；B7 epoch 33、V100、同一 Q1 batch 32 的 20-step
  门禁中，概率最大绝对差不超过 3.42e-4，单步耗时 0.457x、峰值显存 0.949x，scaler 正常更新。它是
  后续训练的可选低成本载体，不改变 B7 FP32 默认或声称 accuracy win。详细见 `logs/log-2026-09-21-c4-amp.md`。
- **E5a checkpoint 代码边界：完成。**checkpoint payload、AMP 一致性检查、恢复和原子写入已提取为
  唯一模块；FP32 旧 checkpoint 兼容，AMP scaler 可恢复。无训练或数值行为变化。详细见
  `logs/log-2026-09-21-e5a-checkpoint-boundary.md`。
- **E5 代码边界：完成。**checkpoint 责任已提取；生产 `evaluate_split` 与历史
  `basemodel.test_one_step` 语义不同，明确隔离而不强行合并。详细见
  `logs/log-2026-09-21-e5a-checkpoint-boundary.md` 与 `logs/log-2026-09-21-e5b-evaluation-boundary.md`。
- **E6 分桶 batch：技术通过。**`--use_bucket_batch` 默认关闭；Q1 batch 32 的 V100 门禁中，平均原子槽
  为固定宽度的 0.171x、单步耗时 0.539x、峰值显存无变化。它改变 batch 组成，只作为可选成本载体，
  不产生 accuracy 结论。详细见 `logs/log-2026-09-21-e6-bucketed-batches.md`。
- **E7 lint/CI：完成。**`bash tools/ci/check-static.sh` 固定 Ruff 致命错误检查、所有受跟踪 Python 文件的
  编译及 48 项无缓存 CPU 合同测试；GitHub Actions 在 push/PR 上复现。它不读取 Q1、不启动训练，完整的
  本地数据回归仍由 `python3 -m unittest discover tests` 覆盖。详细见 `logs/log-2026-09-21-e7-lint-ci.md`。
- **B7 M1 CIF 盲推理入口：完成。**`b7_cif_infer.py` 严格加载 B7 epoch 33、按 Z0 的 CIF `N_val` 和
  H1 eta/gamma 重建 E0/P0 blind 谱；旧 M4 `cif2dos.py` 已退役（Git 历史可恢复）。详细见
  `logs/log-2026-09-21-b7-cif-blind-inference.md`。
- **D4 phDOS 尖峰与负频坐标质量审计：完成，有可行动关联。**B7 phDOS 失败在 train p90 以上的负频
  坐标质量代理中为 16.09%（other 2.09%，差 14.00pt、95% CI 9.41–18.73pt）；尖峰集中度不相关，coverage
  外质量为零。负频坐标不等同于已证实虚频。详细见 `logs/log-2026-09-21-d4-phdos-spike-imaginary-audit.md`。
- **D4b 负频坐标来源与稳定性审计：完成，无全库数据处理授权。**D4 high 的 230 条都精确回连冻结原始
  phDOS（MP `pheasy` 210、JARVIS 20），其原始负坐标 DOS 质量也高，故非 P0 重分箱伪影；但 MP 与
  PhononDB 没有逐材料稳定性／收敛标识。JARVIS 的 `min_fd_phonon_mode` 仅覆盖该来源 209 条，不能外推
  为全库虚频真值。详细见 `logs/log-2026-09-21-d4b-negative-coordinate-provenance.md`。
- **D4c JARVIS `min_fd_phonon_mode` 语义审计：closed。**该字段来自 `MAIN-ELAST` 有限位移来源，但
  209 条中有 167 个 `-0.0`（数值为零）；它仅在 7/209 条等于可见 `phonon_modes` 最小值、在 0/209 条
  等于 phDOS 网格最小频率。因此不能作全局最低模式、稳定性或收敛真值。详细见
  `logs/log-2026-09-21-d4c-jarvis-min-fd-semantics.md`。
- **eDOS Q1 错误归因与形状诊断：**test/valid 均显示高粗糙度/高熵谱形 oracle 表现更差；valid 冻结 B7 epoch 33 的归一化预测形状显示，高粗糙度组粗糙度偏差中位数 `−0.318`（95% bootstrap CI `−0.328…−0.312`），高梯度边差分误差占比相对其他组 `+0.142`（`+0.121…+0.166`），与预测过度平滑相容；但峰位偏移差异不稳，且不能据此证明根因或因果。全 coverage 结果方向一致。gamma、coverage 与结构标量审计边界不变。详见
  `logs/log-2026-09-23-eDOS归因.md`、`logs/log-2026-09-23-eDOS-valid复核.md` 和 `logs/log-2026-09-23-eDOS形状诊断.md`。
- **eDOS slope-loss pilot 已完成并 park。**Q1 valid、M1×10、seed 42 的成对结果中，高粗糙度组 eDOS oracle Δmedian R² 为 `−0.00167`（95% bootstrap CI `−0.01788…0.01379`），未达 `+0.02` 主门槛；整体保护项通过，但 slope-error 机制不支持。两臂均跳过 test；不进入 M1×35 确认或权重扫描，默认损失不变。详见 `logs/log-2026-09-23-eDOS-slope-loss-pilot.md`。
- **eDOS 高粗糙度组梯度归因完成（只读）。**在冻结 B7 epoch 33 的 Q1 train 全集上，高粗糙度组占 10.0%，但其每样本 eDOS 梯度范数为其他组的 2.06–4.94 倍；按样本比例加权后，在 encoder/decoder/eDOS head 探针中的梯度范数占比为 35.4%/20.6%/18.6%，两组梯度方向余弦均为正。样本比例稀释假设不获支持，不做高粗糙度样本加权 pilot。未访问 valid/test；详见 `logs/log-2026-09-25-eDOS-gradient-group-attribution.md`。

## 待办顺序

1. **下一步待讨论：**eDOS slope-loss 与样本稀释两条假设均未支持新的训练 pilot；不做 M1×35 确认、样本加权或常规损失扫描。与马尚酱讨论另一条可证伪的 eDOS 机制假设，优先检查谱形表达能力或结构条件信息；新设计获批前不启动训练。记录见 `logs/log-2026-09-23-eDOS-slope-loss-pilot.md` 和 `logs/log-2026-09-25-eDOS-gradient-group-attribution.md`。C4 AMP 与 E6 分桶 batch 保持默认关闭。
2. C2.1b 已 park；除非出现区别于 L3 的有方向机制，不重开或做常规权重扫描。C3 PhysMoE、D3 eDOS
   辅助数据和 phDOS 尖峰/虚频路线仍不进入近期顺序。
3. 网格、窗口与 bin 的数据表示实验保持搁置；只有新的可变 query 假设在固定 E0/P0 上技术通过后，
   才能重新设计并启动。
4. 独立工程队列：C4 自动混合精度、E5 代码边界、E6 分桶 batch、E7 lint/CI，以及将 B7 M1 接入
   CIF 推理入口均已完成；M4 `cif2dos.py` 已退役。

长期方向是在任意能量或频率上查询连续谱场：先证明读出接口，再验证数据表示，最后才进入连续谱场。

## 阻塞与关注点

- `dataset.py` 中坐标默认断言的回归已修复（自动回退）；重跑 C2b 前须做冒烟测试。
- 旧 M4 `cif2dos.py` 已删除；B7 M1 盲推理唯一入口为 `b7_cif_infer.py`。
- `output/` 中约有 93 GB 检查点（92 个目录）；未经逐项明确批准不得删除。正式复现以 `results/`、完整命令和
  带日期日志为准，`output/*/config_used.yaml` 仅为本地产物。
