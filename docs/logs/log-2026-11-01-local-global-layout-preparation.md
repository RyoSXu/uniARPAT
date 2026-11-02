# 局部—全局中段编排：准备与短检

日期：2026-11-01。[单候选设计与判据](../design/design-2026-11-01-local-global-layout.md)。
用户同意检验第二次局部更新的位置。本轮已准备并短检，正式训练未启动，未提交Git。

## 实现范围

对照`L1 L2 G1 G2 G3 G4 G5 G6`，候选`L1 G1 G2 G3 L2 G4 G5 G6`。
两次完整A／四个内部块、六层全局注意力与FFN、LR2内部公式、外层残差及E＋S联合训练均保持。
原`staged`和`alternating`含义保持；不把新编排默认为已采用的工作点。

模块／调用链 → 文件 → 符号：

- 执行顺序 → `model/local_global_encoder.py` → 新增`MIDPOINT_ORDER`，
  `LocalGlobalEncoder.__init__`限制LR2 L2/G6，`execution_order`仅新增分支，不增删模块或参数。
  `model/transformer.py::Transformer.__init__`将该布局限制为LR2 per_atom路径。
- 干预定义／校验 → `utils/spectral_layout_experiment.py` → 新增`MIDPOINT_SCHEMA`、
  `layout_ablation_contract`、`validate_midpoint_model`；检查真实顺序后用无副作用代理复用原LR2模块校验。
- 配方与初值 → `utils/spectral_balance_experiment.py` → `legacy_balance_config`、`_backbone_params`、
  `validate_balance_config`、`balance_model_contract`、`initialize_balance_model`支持这一个新编排。
  共同初值仍在canonical staged参考上构建；新schema为`joint-spectral-balance-layout-model-v1`。
- 准备入口 → `tools/eval/spectral_layout_prepare.py::prepare/main`薄封装原准备器。
  `spectral_residual_prepare.prepare`新增默认false的`midpoint`参数；
  `spectral_balance_prepare.completed_spectral_control`新增默认I3的`local_route`参数，以绑定完成的staged LR2组。
- 训练／导出 → `tools/train.py::build_arg_parser`增加现有布局参数的新选项；
  `tools/eval/lg_valid_export.py::load_run`拒绝布局、schema及执行定义漂移；训练loss与选点实现未改。
- 短检 → `tools/checks/check_spectral_balance.py::check`复用原真实train批次检查，核对新编排和共同初值。
- 测试 → `tests/pipeline/test_spectral_layout.py::SpectralLayoutTests`六项，加入静态检查入口。

## 准备与绑定

```bash
python tools/eval/spectral_layout_prepare.py \
  --control-preparation results/runs/spectral_residual_20261030/report.json \
  --out results/runs/spectral_layout_20261101 --tag-prefix midpoint20261101
python tools/checks/check_spectral_balance.py \
  --preparation results/runs/spectral_layout_20261101/report.json \
  --out results/runs/spectral_layout_preflight_20261101
```

实际解释器`/opt/tefscloud/miniconda3/envs/torch/bin/python`；`PYTHONUTF8=1`，
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`。

- 绑定主对照为已采用的staged LR2 E＋S原选中ep29，已完成35轮。没有重训或改选点。
- 新tag：`_sbal_midpoint20261101_lr2_midpoint_s42_e35`。
- protocol逐字复制；training record只变化`local_layout`和相同内容的protocol路径。
- 准备75项输入／源码、短检56项源码分别指纹绑定，新报告保存源码快照且拒绝覆盖。
- 新模型与主对照初始化SHA256一致：
  `924bd18a26ba7fe5d266713df9d463bdb6d1b6547787fce699944327e208eeb6`。
  参数量76,077,252，调用计数仍为2／4／6／6。
- 新增编排需要新的源码指纹；旧实验报告原样保留。已用当前代码重建旧H1 LR2、A64 E＋S及
  staged LR2 E＋S三份正式运行契约，与各自保存契约逐项相同；不把旧预检的源码哈希冒充当前哈希。

## 已执行验证

72项相关测试通过：新编排、LR2 E＋S、谱形辅助、balanced pipeline/data、旧LR2、
完整local/global encoder及原比较入口。维护目录Ruff与`git diff --check`通过。

新编排测试确认实际L2输入等于G3输出，所有局部／全局模块各执行一次；手写固定顺序的前向与
全部Encoder参数梯度在FP32／FP64下匹配。共同初值、参数量、初始化RNG及恢复后的AdamW状态一致，
同形状的staged／midpoint checkpoint被契约隔离。拒绝未批准配方、错误布局／schema／公式。
六种E(3)变换及原子重排在两种dtype的小型完整模型上通过，含物理blind输出；不宣称覆盖所有材料或晶胞／超胞。

V100-SXM2-32GB上执行两个可丢弃的真实train batch32更新，分别352／380个原子：

| 项目 | 实测 |
|---|---:|
| 更新耗时 | 1.8344／0.7987秒 |
| 峰值已分配显存 | 7509.40 MiB |
| 训练loss减辅助S与blind E评估差 | 6.16×10⁻⁸ |

loss和梯度有限；没有实验checkpoint或正式history，test未评估。生产组装短检只创建了目标空run目录。
一次train批的共享Encoder梯度余弦为−0.0392，仅作局部诊断，不据此调loss或推断全局任务冲突。
两批短检不保证全数据峰值显存／耗时；正式35轮仍按旧组成本估计约4–5小时，首完整轮再更新ETA。

## 后续执行

OpenCode核对本轮准备和短检哈希后，原样执行准备报告的唯一command，从头训练单组seed42、35轮。
监测进度、数值、资源与退出情况；只允许同契约、原tag的latest恢复，不调参或重训对照。
按原双任务保护选择完整valid best，与staged ep29配对比较；总体误差与同组成谱差分别判读，
材料bootstrap seed20261101，组成组bootstrap seed42。无合格best不能用latest替代。
结果归档到本轮analysis目录和新正式运行日志，是否采用交用户决定。
