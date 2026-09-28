# 同组成谱差辅助 Q1 valid-only Pilot

日期：2026-09-28
状态：**完成并 park；未读取 test，不进入 M1×35。**

## 决策问题

在保留 B7 完整双谱主损失与 H1 的前提下，检验完整谱锚定的 eDOS 同组成谱差辅助监督能否改善
Q1 valid 的 blind eDOS 主指标，同时保护 oracle eDOS、phDOS 和配对材料的单谱质量。设计、单一因素、
校准、停止条件与五项门槛已在训练前写入
`../design/design-edos-pair-auxiliary-pilot.md`。

## 执行口径

- control：`_pcctl`；candidate：`_pcaux`。
- 两臂均从 B7 `_e9ctl` epoch 33 同一文件初始化，Q1、M1×10、seed 42、batch 32、FP32，载入后
  重置 RNG；使用各自 epoch 10 `checkpoint_latest.pth`，不按 valid 选择 epoch。
- 两臂均执行相同主 batch、pair 计划和辅助前向。唯一因素是 candidate 加入校准后的 pair loss 梯度；
  `pair_ratio=0.10`，`lambda_pair=0.25936880707740784`。
- 冻结 pair 宇宙哈希为
  `f668e9d938d91081626c69d070faaa40df2cfe0c9de3d3aae6cf93befefef730`；epoch 0 计划哈希为
  `0f5a390ac0cc5c9fbb65f6bb57ac76641995bcac13c76f30a8ce866a5baebf81`。两臂配置逐项一致。
- 两臂 history 均为连续 epoch 1–10；训练脚本正常退出，自动 test loader 和 test inference 明确跳过。
- 裁决只读 Q1 valid 2,313 条，bootstrap 2,000 次，固定 seed 20260928。

训练命令：

```bash
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _pcctl \
  --pair_aux_arm control --pair_ratio 0.10 --skip_test_eval \
  --init_ckpt ./output/ablation_m1_e9ctl/checkpoint_best.pth --reset_rng_after_init

python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _pcaux \
  --pair_aux_arm candidate --pair_ratio 0.10 --skip_test_eval \
  --init_ckpt ./output/ablation_m1_e9ctl/checkpoint_best.pth --reset_rng_after_init
```

裁决命令：

```bash
python3 tools/eval/edos_pair_aux_verdict.py \
  --control-checkpoint output/ablation_m1_pcctl/checkpoint_latest.pth \
  --control-config output/ablation_m1_pcctl/config_used.yaml \
  --candidate-checkpoint output/ablation_m1_pcaux/checkpoint_latest.pth \
  --candidate-config output/ablation_m1_pcaux/config_used.yaml \
  --output-prefix results/edos_pair_aux_q1 --device cuda --bootstrap 2000
```

## 结果

| 门槛 | control | candidate | candidate − control | 结论 |
|---|---:|---:|---:|---|
| 全体 eDOS blind 中位 R² | 0.47980 | 0.48139 | +0.00159 | **主门失败**；要求 ≥+0.02 |
| 全体 eDOS blind 失败率 | 9.166% | 9.382% | +0.216pt | 通过保护线 |
| 全体 eDOS oracle 中位 R² | 0.51864 | 0.52026 | +0.00162 | 通过 |
| 全体 eDOS oracle 失败率 | 6.399% | 6.528% | +0.130pt | 通过 |
| 全体 phDOS oracle 中位 R² | 0.74544 | 0.74293 | −0.00251 | 通过 |
| 全体 phDOS blind 中位 R² | 0.73902 | 0.73721 | −0.00181 | 通过 |
| 117 组／163 pair 组等权谱差误差 | 0.29110 | 0.28974 | 相对下降 0.465% | **机制门失败**；要求 ≥5% |

机制改善的 95% bootstrap 区间为 `0.104%…0.880%`，下界为正但幅度没有达到预注册的 5%。
paired-256 保护门通过：oracle/blind eDOS 中位 R²差为 `−0.00186/−0.00280`，失败率差为
`+0.781/−0.391pt`。五项门禁结果为：总体主门失败、机制门失败，其余三项保护门通过。

完整机器可读结果见 `../../results/edos_pair_aux_q1.json`；`../../results/edos_pair_aux_q1_pairs.csv`
保存两臂各 320 个 valid pair，其中每臂 `primary_valid=true` 的 163 pair／117 组用于机制门。两臂训练历史见
`../../results/history_m1_pcctl.csv` 和 `../../results/history_m1_pcaux.csv`。

## 结论与行动

事实是候选在总体 blind eDOS 上只产生 `+0.00159` 的平局幅度变化，且同组成谱差机制改善只有
`0.465%`。保护项通过说明该辅助项在本次预算下没有造成明显破坏，但不能把“安全”解释为精度收益。

按预注册行动表，本路线 **park**：不进入 M1×35，不做 seed 扩展，不扫描 ratio、pair 频率、损失形式
或 sampler，也不追加逐层或训练集诊断。默认 B7、损失和推理接口不变。下一项模型工作必须回到项目
决策层，结合已关闭路线重新提出一个能直接进入独立 valid 检验的新假设；本结果不自动授权替代候选。
