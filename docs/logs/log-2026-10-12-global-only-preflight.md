# 2026-10-12 — 无局部更新 G0 对照实施与训练前验证

用户要求先整理并提交仓库，再给 OpenCode 正式训练与监测提示词。
已有工作已由 `36ee443` 和 `eb1d193` 分别归档实现/测试和实验记录/当前状态。
本轮另行实施[六层全局无局部更新对照](../design/design-2026-10-12-global-only-attribution.md)。

## 实施范围

G0 复用新 `ScalarGlobalLayer`，仅创建六层全局模块，局部调用和内部局部块均为零；
`Transformer.forward` 直接传入原子态和 mask，不构建周期记录或几何缓存。
新增 `global_only_config`、`validate_global_only_config`、`global_only_reference_params`、
`initialize_global_only_model`、`global_only_contract`，经既有初始化/训练/评估分派接入。
旧 A/B+/C0/C1、原 Encoder、局部公式与旧 checkpoint 契约保持。

运行入口默认只打印一条命令；`--run` 要求成功且仍匹配源码/参考/配置的真实预检。
未完成组沿用原训练入口恢复 optimizer/RNG 与配置；已完成组直接校验、复用，不重训。
`--analyze` 或训练同时指定 `--out` 时，对 G0 与已有 C0/C1/C2/原 baseline 做全量 valid 配对。
`comparison_report` 明确以 G0 为参照，并把原 baseline 标为 `original_baseline`，
避免复用配对工具时将两种 baseline 的元信息混为一组。

## 实际执行的验证

- 完整 `tools/checks/check-static.sh`：ruff、内存编译、**341 项测试**通过，749 个可入库文件
  通过边界检查。新增 9 项 G0 测试覆盖零局部执行、共同权重/RNG、几何独立、重排/填充、
  配方边界、真实训练配置组装、checkpoint 恢复、已完成组复用与配对报告的方向/参照。
- 最终源码真实 GPU 预检 **44.93 秒**：两批 train 的完整原目标前后向通过，全部参数梯度存在且有限；
  一批 valid 的 blind/oracle 逐样本指标有限。内存 checkpoint 的权重、optimizer、RNG 往返通过。
- 一批真实 valid 的六种 E(3) 变换，两种计算精度 FP32/FP64：eDOS、phDOS、eta、shape 与
  blind 重建谱最大相对差均为 **0**。这是所执行变换检查的结果，不代表遍历了所有输入。
- C2 未训练参考初值完整摘要复现：`dbdbbdf50a076083…`。G0 剩余 **219 个张量**逐值复制，
  全模型 **75,552,068** 参数，状态摘要 `a0ac3a913e0604e7…`；未读入已训练权重作为新组初值。
- 默认命令核对：seed42 / 35 轮 / batch32 / lr5e-5 / G0 / K0 / G6 / staged /
  `balanced_oracle_mae` / ZP / Q1 / `skip_test_eval`。
- 复核最终预检仍匹配当前源码和参考；已有 C0/C1/C2/原 baseline 的数据契约、35 轮历史、
  选点及 checkpoint 身份通过检查，best epoch 分别为 **29 / 34 / 35 / 34**。

最终准入证据：`results/runs/global_only_attribution_20261012/preflight_final.json`。
早期 `preflight.json` 保留为过程证据；其后报告入口已修正，正式训练只使用最终准入报告。

## 固定批次成本与正式预算

V100 32 GiB、FP32，预热后各三次**完整 optimizer 步骤**：

| train 批次 | 中位步骤耗时 | 最大 CUDA 分配 | 最大 CUDA 保留 | 实际微批次 |
|---|---:|---:|---:|---|
| 32 材料 / 352 原子 / padded width40 | 0.18074 s | 3,883 MiB | 4,214 MiB | 32 |
| 32 材料 / 380 原子 / padded width94 | 0.25114 s | 3,660 MiB | 4,262 MiB | 23 + 9 |

含 AdamW 状态和一次性模型的实际更新；不含加载全数据、完整轮验证、保存与最终分析。
已核对 train/valid 实际材料数 18,704/2,313；正式队列仍为一组 seed42、35 轮。
按 C0 历史每轮 train 120 秒及此次两批短测，预留 **1–2 小时**，正式开始后由前几轮更新估计。
未因显存空余改变有效 batch 或学习率。

## 当前状态与边界

**正式训练尚未启动**，G0 没有 history 或训练 checkpoint；预检更新仅作用于丢弃的一次性模型。
未实际执行完成后的五模型全量 valid 分析，因为 G0 尚未训练；报告方向/参照已由测试检验。
当前证据只表明入口和已执行的准入检查通过，没有得出局部更新、插入顺序或全局骨架的性能结论。
正式执行和监测交给 OpenCode，不自动扩大实验、读取 test、更换选点或追加预算。
