# LR2迁移E＋S：准备与短检

日期：2026-10-30。用户同意单候选迁移；[设计和预设判据](../design/design-2026-10-30-spectral-residual.md)。
本轮已完成实现、对照绑定及短检，尚未启动正式训练；未提交Git。

## 变更与证据

已有LR2模型公式直接复用，没有新增消息计算或数据字段。相对当前完整A64 E＋S，只把四个内部块
从`LN64(s+F(s,R))`改成`LN64(F(s,R))`；外层残差、宽度、两次完整局部调用及staged L2/G6均保留。

模块／调用链 → 文件 → 符号：

- 干预定义 → `utils/local_residual_experiment.py` → 新增`residual_ablation_contract`，
  由原`residual_contract`和新E＋S契约共用；旧LR2契约内容不变。
- 配置／共同初值 → `utils/spectral_balance_experiment.py` → `validate_balance_config`、
  `legacy_balance_config`、`_backbone_params`、`initialize_balance_model`、`balance_model_contract`；
  LR2仅允许当前E＋S和共享decoder，schema为`joint-spectral-balance-residual-model-v1`。
- 对照绑定 → `tools/eval/spectral_balance_prepare.py` → 新增`completed_spectral_control`，
  原`completed_energy_control`仍绑定纯E；不重新选择主对照checkpoint。
- 单候选准备 → `tools/eval/spectral_residual_prepare.py` → 新增`prepare`、`main`；
  检查完整35轮对照、所选best、valid导出和配置，只打印命令。
- 短检／导出 → `tools/checks/check_spectral_balance.py::check`、
  `tools/eval/lg_valid_export.py::load_run`；核对共同初值和新路线／公式／schema一致性。
- 验证 → `tests/pipeline/test_spectral_residual.py::SpectralResidualTests`六项测试；加入静态检查入口。

## 实际准备

```bash
python tools/eval/spectral_residual_prepare.py \
  --control-preparation results/runs/spectral_shape_aux_20261025/report.json \
  --out results/runs/spectral_residual_20261030 --tag-prefix lr2es20261030
python tools/checks/check_spectral_balance.py \
  --preparation results/runs/spectral_residual_20261030/report.json \
  --out results/runs/spectral_residual_preflight_20261030
```

实际解释器为`/opt/tefscloud/miniconda3/envs/torch/bin/python`；启用`PYTHONUTF8=1`，
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`。
准备和短检均保留输入哈希与源码快照，拒绝覆盖已有目录。

- 已完成主对照：`ablation_m1_sbal_shape20261025_shape_a64_s42_e35`，按原规则选中的ep25。
- 新tag：`_sbal_lr2es20261030_lr2_s42_e35`；正式run目录目前仅为组装短检创建的空目录。
- protocol逐字相同；training record仅变化`local_route`及内容相同的protocol路径，tag单独隔离。
- 73项准备输入／源码哈希及54项短检源码哈希重新核验全部一致。
- 新模型与已完成主对照初值SHA256均为
  `924bd18a26ba7fe5d266713df9d463bdb6d1b6547787fce699944327e208eeb6`。

## 已执行验证

48项相关测试通过：`test_spectral_residual`、`test_spectral_shape_aux`、`test_spectral_balance`
（pipeline及data）和`test_local_residual`。覆盖：

- 所有初值张量／参数量相同，初始化不消耗调用方RNG；开启四个加法后输出逐项恢复A64。
- E＋S实际训练／评估路径一致，两类任务联合反传，含真实角消息的测试几何产生有限局部参数梯度。
- 配置拒绝未批准组合；CLI和生产组装正确；相同形状的A64及旧LR2契约不能混用。
- 同契约恢复模型和AdamW状态；valid导出拒绝schema、route、公式或辅助loss标志漂移。
- 原纯E、谱形辅助、旧LR2相关检查继续通过。

另对已保存的旧LR2及A64 E＋S正式运行，从其training record重建契约，与保存契约逐项相同。
维护目录Ruff通过；191个Python文件内存语法检查、五份相关文档的相对链接、`git diff --check`
及仓库边界检查通过（899个prospective文件）。没有宣称已运行完整历史测试套件。

V100-SXM2-32GB上执行两个可丢弃的batch32更新，分别含352／380个原子：

| 项目 | 实测 |
|---|---:|
| 两次更新耗时 | 1.8761／0.7874秒 |
| 峰值已分配显存 | 7505.50 MiB |
| 训练loss减谱形辅助与blind E评估差 | 1.32×10⁻⁸ |

loss和梯度有限，完整训练未启动，未写实验checkpoint，未评价test。
一次train批上的共享Encoder梯度范数为eDOS 1.4269、phDOS 0.4862，余弦−0.0248；
这是局部诊断，不能推断全局任务冲突、调loss权重或认定两个任务已经平衡。
两批显存及时间不代表全数据峰值／整轮耗时，不能据此扩大batch。

测试和短检仅确认当前路径可运行及契约一致，不证明LR2在新配方有效。正式结果仍未知。

## 交接

OpenCode读取本轮准备`report.json.command`原样启动一组35轮，沿用双任务保护，
监测退出、epoch进度、数值及显存。先核验哈希；保持脱离会话运行、保存退出码，
遇异常停止并保留证据，仅原参数／原tag／同契约latest允许恢复。
完成后复核完整history与best，导出完整valid，与E＋S ep25做预设材料配对比较及同组成谱差诊断。
无合格best不能改用latest。结果按设计归档，本轮不自动增加候选或调整预算。
