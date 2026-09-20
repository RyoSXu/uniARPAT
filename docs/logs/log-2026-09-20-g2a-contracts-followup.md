# 日志：2026-09-20 — G2a 合同补齐（生产冒烟与等变输出；未训练）

## 范围

- 任务与假设：补齐 G2a 预检的三处缺口，不改变已实施的模块与设计口径：
  完整训练配方下的 Q1 单步数值合同、平移的整模型输出合同、基矢交换与幺模变换的
  G2 残差层输出合同。
- 改动的文件或配置：仅 `tests/test_g2_periodic_edges.py`（新增 4 项测试并更新文件头
  的关卡说明）。未改模型、训练入口、数据、缓存、损失、默认配置或既有结果文件。

## 证据

- 命令、标签、随机种子和 epoch 预算：未运行 `run_ablation_experiments.py` 训练
  （`_g2ctl/_g2edge` 均未启动）。仅运行
  `python3 -m unittest tests.test_g2_periodic_edges -v` 与
  `python3 -m unittest discover tests`（CPU）。
- 结果文件与测试：
  - G2a 文件：17/17 通过；全量：68/68 通过（约 22 秒）。
  - 新增 `test_g2_production_train_one_step_sumnorm_h1`：按 `_g2edge` 生产配方构造
    `basemodel`（M1：d_model 512、6+6 层、dropout 0.05、`loss_form='sumnorm_klw'`、
    `scale_mode='eta'`、seed 42），Q1 train 真实 batch（batch 2、
    `dos_sumnorm=True`、nvalence 侧车存在），CPU 单步约 1.5 秒；返回的全部 14 个
    损失键有限：`loss 4.674737、loss_edos 1.070166、loss_phdos 3.356380、
    loss_eta 0.248191`（H1 激活），其余键为 0.0 且有限；首层 G2 `alpha`
    取得有限梯度。
  - 新增 `test_g2_full_model_translation_invariance`：启用 G2 的整模型在分数坐标
    整体平移（+0.37 mod 1）前后 eDOS/phDOS 读出一致，最大差约 3.6e-07/5.6e-09
   （容差 1e-5）。
  - 新增 `test_g2_module_basis_swap_invariance` /
    `test_g2_module_unimodular_invariance`：共享隐状态、开启残差（`alpha=1`）时，
    `PeriodicEdgeMessage` 在两种晶胞重表达下的输出一致，最大差约 2.8e-07
   （容差 1e-5）；边距离多重集一致性由既有测试覆盖。
- 指标：`e med/fail + p med/fail + (test|valid, epN, oracle|blind, Q1|pre-Q)` ——
  本轮无训练，故无指标结果。
- 首次使用 `docs/glossary.md` 未定义的术语时，给出中文解释：本轮无新增术语；
  “帧（frame）”指晶胞基矢在实空间的取向约定。
- 范围界定证据（为何基矢交换/幺模不要求整模型 1e-5）：两种变换经
  `(a,b,1/c)+角度` 参数化重建后隐含整体旋转，方向敏感的 B7 主干（球谐相对编码）
  随帧变化；实测未训练 B7 在基矢交换下 eDOS 差约 1.9e-02（phDOS 约 1.7e-04），
  故整模型不变性不是 G2a 单因素合同；G2 径向残差本身与帧无关，模块层满足 1e-5。

## 结论

- 状态：pending。
- 原因：实现、单测（含本次补齐的生产冒烟与输出合同）、Q1 边数审计已按设计通过；
  但设计要求的 V100 batch 32 资源门禁（同一批次比较 B7 与 G2 单步峰值显存和耗时，
  任一超 1.25 倍或 OOM 即停）尚未执行，故 `_g2ctl/_g2edge` 的 Q1 M1×10 成对 pilot
  不得启动。

## 交接

- 下一项关卡工作：在同一 V100、batch 32、同一输入批次下实测 B7 与 G2 的单步峰值显存和
  耗时；通过后才运行
  `python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _g2ctl` 与
  `--tag _g2edge --use_g2`。
- 对 status、backlog 和 decisions 的更新：未改 `status.md`（当前仍为“G2a 设计完成，
  等待实施预检”；待资源门禁通过后再更新预检状态）；未改 `decisions.md`（尚无可用结论）。
