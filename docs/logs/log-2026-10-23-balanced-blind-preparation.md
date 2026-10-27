# 日志：2026-10-23 — 归一化 blind 联合目标准备

用户确认新主指标为归一化 blind 全谱误差，双任务分别保护失败率；
公式、保留的 R² 保护及兼容边界见[设计](../design/design-2026-10-23-balanced-blind-objective.md)。
本页记录2026-10-23准备阶段：当时代码与短检完成、正式35轮训练未启动，I1/A64/LR2历史结论未改写。
进度同步（2026-10-27）：正式35轮现已完成，best ep32，见[运行与核验](log-2026-10-23-balanced-blind-run.md)；
后续谱形辅助也已完成，当前配方以[状态页](../status.md)为准。以下保留原准备证据与交接方式。

## 已落地

- 显式 `--spectral_balance_config`＋`--scale_mode per_atom`＋`--selection_metric balanced_blind_energy`；
  缺少匹配协议即拒绝，冻结完整 A64、seed42、35轮，其余 Encoder 不扩展。
- 仅有效 train 拟合 C 与数值总量参照；数据 manifest、train 载荷、valid 固定参考导出与 checkpoint 身份绑定。
- loss 和 valid 选点同用无量纲 blind 全谱误差，任务等权；旧 H1/KL/W1/Huber 路径保留。
- best 先过滤两任务失败率与原有 R² 保护，再最小化误差；无合格轮次保留 latest、拒绝 best 导出。
- 配套初值复用、按材料加权的微批、恢复、逐材料导出与源码快照。

## 已有结果重算，仅为诊断

同一 train 常数重算旧规则选出的两个 checkpoint，使用完整 valid：

| 旧选中模型 | mean E_eDOS | mean E_phDOS | 等权 mean E |
|---|---:|---:|---:|
| A64 ep31 | 0.341748 | 0.215207 | 0.278477 |
| LR2 ep29 | 0.341128 | 0.207539 | 0.274333 |

这不是按新规则重新选点，也不宣布 LR2 采用。其 ep29 eDOS 失败数212，高于固定 A64 的196，
故不满足新协议失败率门槛。没有保存每一轮预测，不能仅从旧 history 追溯计算新目标最优轮次。
两类谱的真实误差差异仍存在；不通过定义让两列人为相等。

现有 LR2 train 冻结预测上的旧谱形 loss 均值约 eDOS0.18815、phDOS0.11621；
这仅描述 eval 模式的 loss 数值，不证明共享梯度强弱或原训练动态。

## 短检与验证

V100-SXM2-32GB、FP32、batch32 两个代表性 train 批的一次性更新约1.89秒、0.79秒，
峰值已分配显存7505.5 MiB。额外一次 lr=0 检查的训练 loss 与评估分数差小于1e−7。
新头使用未训练 A64 张量映射；复建的 A64 原始初始化 hash 与既有运行记录相同：
`2a2f9a507cf9a6955fe6c299be995b9fec78e5fe3b5d254148eb27aa680fe3fa`。

单批共享 Encoder 梯度范数 eDOS1.369、phDOS0.496，余弦−0.078；
只覆盖两次一次性更新后的一个批，不证明全程梯度失衡、负迁移或修正方案的精度。
本方案平衡目标定义与任务系数，不强制梯度范数一致。

解析与调用链检查覆盖补零/单位/晶胞倍率、零谱、形状与总量梯度、过滤/并列/无合格 best、
微批单次更新、恢复与导出；保留旧选择与残差路径回归。
78项相关测试通过（其中19项新协议检查），当前`model/utils/tools/tests`的Ruff与内存语法编译通过，134个文档本地引用有效；
手动改写C的反例被训练入口拒绝。未宣称全仓测试通过。
旧缓存完整迁移检查有一项无法运行：迁移前 `data/train4ARPAT/train/train_index.npy` 不在当前服务器；
不以伪造旧文件或放松检查补齐。当前完整晶体数据检查与新协议检查另行执行。

预检只创建一次性内存模型；未写实验 checkpoint、未评估 test。既有数据契约校验会验证
manifest 所列字节哈希，这不属于使用 test 标签调参。短检不能代替整轮成本或精度验证。

## 复现与交接

```bash
python tools/eval/spectral_balance_prepare.py \
  --analysis-dir results/runs/local_internal_residual_20261021/analysis \
  --out results/runs/spectral_balance_20261023_final_v2 --tag-prefix energy20261023
python tools/checks/check_spectral_balance.py \
  --preparation results/runs/spectral_balance_20261023_final_v2/report.json \
  --out results/runs/spectral_balance_preflight_20261023_final_v2
```

输出目录必须全新；初版准备/短检保留，不覆盖。准备入口打印唯一完整训练命令，也写入
`report.json.command`；短检通过后才执行该命令。总量和数据修正包从头训练，不能载入旧权重续训。
训练后使用 `tools/eval/lg_valid_export.py` 的现有入口导出完整 valid 新指标。

长期训练交给 OpenCode：记录两任务 E、中位/P10 R²、失败率、eligible、best 与 latest；
首轮实测后更新剩余时间。沿用完整 A64 约数小时预算，两个批不能可靠估算整轮与总耗时。
若始终无合格 best，原样报告，不追加预算或放松保护线。

5条电子谱负能量待核样本、价电子定义差异与 coverage 物理含义继续保留为数据问题；
本轮不认证/删除标签、不重分箱、不改 coverage mask、不追加 Encoder 公式或头数对照。

## OpenCode 执行提示词

> 阅读本日志和对应设计，执行已准备的新版联合目标实验。核对
> `results/runs/spectral_balance_preflight_20261023_final_v2/report.json` 的 passed、配置和全部源码哈希
> 与当前一致，再执行 `results/runs/spectral_balance_20261023_final_v2/report.json` 中的 command 列表。
> 显式设置 `PYTHONUTF8=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`，
> 仅完整A64、seed42、35轮、batch32、lr5e-5、FP32、train/valid；
> 使用冻结protocol，保持当前Encoder，不改数据、mask、门槛或预算。
>
> 启动后持续监测至完成：每轮记录两任务blind mean E、中位/P10 R²、失败率、eligible、best/latest、
> 耗时与显存；首轮后更新预计总耗时。异常即停并保留证据，恢复须校验契约与完整history。
> 没有合格best则报告no_eligible_best，不以latest代替、不临时放松约束。
>
> 有合格best时，用既有lg_valid_export入口导出完整valid及谱；使用同一protocol重算固定A64旧选中点，
> 按材料配对bootstrap 2000次（seed20261023），给出两任务与等权mean E差、R²中位数差、失败率差及区间，
> 同一次抽样同时用于两任务和两模型。单列5条待核材料对误差的贡献，正式评分仍保留它们。
> Oracle与同组成谱差仅作诊断。注明A64旧选点不同，不能将完整修正包归因于单独loss、头或Encoder，
> 区间只描述已选checkpoint的材料抽样，不代表跨seed稳定性。不读取test作评估、不自动换I1或LR2、
> 不追加实验、不提交Git；结果写入新日志，状态页只留简短结论。
