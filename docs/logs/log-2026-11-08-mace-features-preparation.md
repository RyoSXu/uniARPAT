# 日志：2026-11-08 — MACE 冻结特征参照（M1）的实施与预检

对应[设计](../design/design-2026-11-08-mace-features.md)。本轮只实施、抽取特征、测试和短时预检，**未启动正式训练**，未读 test。

## 环境与特征

- 独立环境 `/root/home/newstudy/mace_env`（venv，torch 2.5.1+cu124、mace-torch 0.3.17、e3nn 0.4.4），训练环境未改动。
  首次安装因镜像缺少 pandas 3.x／matscipy 1.3 的预编译包而转向源码编译，改为 `--prefer-binary` 并限制 `pandas<3 numpy<2.3 matscipy<1.2` 后完成。
- 权重 `mace-mp-0b3-medium.model`（hf-mirror `mace-foundations/mace-mp-0`，MIT），SHA256 `2f2be696…`，存仓库外 `/root/home/newstudy/mace_models/`。
  GitHub release 经代理返回 503，改用 hf-mirror。
- 抽取 `tools/data/extract_mace_descriptors.py` → `data/external/mace_mp0b3_medium_v1/`（本地）：每原子 256 维（两层各 128 个 L=0 通道），
  train／valid／test 原子数 220,785／27,533／26,731，用时约 15 分钟（V100）。
  自检：8 个 valid 结构旋转后最大差 3.3e−15，2×1×1 超胞一致性最大差 1.3e−15；无近常数通道，train 标准差 0.030–1.31。

## 实施

| 位置 | 改动 |
|---|---|
| `utils/external_atom_features.py` | 特征 manifest 校验（绑定晶体缓存 manifest 与各划分数组指纹）、train 统计量标准化、按原子偏移切片 |
| `utils/dataset.py`、`utils/bucket_batch.py` | 开启时 batch 追加第 17 字段（每原子特征）；collate 按原子补齐，计算分块时按原子长度裁剪 |
| `utils/crystal.py`、`model/model.py` | `FeaturedCrystalGeometry`：元组仍是 (cell, frac)，特征作为属性在 `data_preprocess` 上设备后附加，所有调用点无需改动 |
| `model/transformer.py` | `external_atom_dim`：`atom_src = ZP(Z) + Linear(特征)`，零初值、在主随机流外构造；配置与输入不一致时报错 |
| `tools/train.py`、`utils/experiment_config.py` 等 | `--external_atom_features`；只在开启时写入模型参数、数据集配置与训练记录；非 E＋S 配方拒绝 |
| `utils/spectral_balance_experiment.py` | `validate_external_features`（只允许 seed42 staged LR2＋E＋S，可配合头深度）；初始化补零投影；契约新增 `external_atom_features` |
| `tools/eval/lg_valid_export.py` | 导出时核对特征的契约、训练记录与三个划分的数据集配置一致，并重新验证指纹 |
| 准备／预检／分析 | `tools/eval/mace_features_prepare.py`、`tools/checks/check_mace_features.py`、`tools/eval/mace_features_analyze.py` |

## 冻结准备

`results/runs/mace_features_20261108/report.json`（本地），tag `_sbal_mace20261108_m1_s42_e35`。
相对 h2 只改 `external_atom_features` 与 protocol 路径；冻结 h2 的 best checkpoint、history、valid 导出与 train 推理指纹作为对照证据。

## 验证

- `tests.pipeline.test_external_atom_features` 6 项通过；全量 `unittest discover -s tests -t .` 521 项，失败 4 项均为本机缺旧 `data/train4ARPAT`（与此前相同）。ruff 通过。
- GPU 预检 `results/runs/mace_features_20261108/preflight/report.json`：不加特征时复现 h2 初值 `264d0a7d…`；M1 除投影外张量哈希相同；
  一个 train 与一个 valid batch 上初始输出与 h2 最大差 3.3e−6，与 h2 自身重复前向的噪声 2.9e−6 同量级（CUDA scatter 非确定性）；
  投影梯度非零；新增参数 131,584（全模型 47,891,140）；热缓存 0.475 秒每批，约 278 秒每轮训练，峰值 12.6 GiB；lr=0 训练目标与评估之差 5.6e−8。
- 成本：约 4.5 GPU 小时（含每轮 valid 评估，参照 h2 实测每轮 402 秒）。
