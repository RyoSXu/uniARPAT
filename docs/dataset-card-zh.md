# uniARPAT crystal-q1-v1 数据说明

[English](dataset-card-en.md) · [来源、引用与许可核查](dataset-sources.md) · 文档更新：2026-10-27（数据版本未变）

## 数据用途与版本

这是从晶体结构联合预测电子态密度（eDOS）和声子态密度（phDOS）的**处理后计算数据集**。
每条记录是一种材料的 MP 标识、完整晶体输入、两类离散谱、覆盖掩码和价电子数。
它适合固定划分下的模型开发；标签来自计算，不是实验测量，也不是统一计算条件下重新计算的结果。

版本名 `crystal-q1-v1` 同时固定 Q1 成员、E0/P0 网格和完整结构修复口径。
数据清单 SHA-256 为 `8e1b0adff516835d52d9fb0321dae5e66fb9a2ac5fd964d317e1dd8647de4b80`。
本地发布候选标签为 `data-crystal-q1-v1`，尚未上传。各来源的许可与署名要求见来源说明；本次不新增整包统一许可证。
MP 的 23,306 个材料已在官方 2026-04-13 快照逐条核实为 `BY-C`，无 `BY-NC`。
本包面向当前 ZP + 局部更新主线，不包含旧周期表、mendeleev 元素特征快照、权重或训练输出。

## 来源与规模

结构和全部 eDOS 来自 [Materials Project](https://docs.materialsproject.org/downloading-data/aws-opendata)。
phDOS 按可用性从 MP、[PhononDB / NIMS MDR](https://doi.org/10.48505/nims.4197)、
[JARVIS-DFT / NIST](https://pages.nist.gov/jarvis/databases/) 选择。
当前计数由冻结 ID 与历史处理表逐行关联得到，不是上游数据库的总规模。

| 划分 | 存储材料数 | 默认使用数 | MP phDOS | PhononDB phDOS | JARVIS phDOS |
|---|---:|---:|---:|---:|---:|
| train | 18,706 | 18,704 | 14,946 | 1,901 | 1,859 |
| valid | 2,313 | 2,313 | 1,863 | 246 | 204 |
| test | 2,287 | 2,287 | 1,835 | 243 | 209 |
| 合计 | 23,306 | 23,304 | 18,644 | 2,390 | 2,272 |

来源列按**存储记录**统计；默认 train 排除两条 PhononDB 记录，故实际使用 1,899 条该来源记录。
存储结构共有 275,049 个原子，最大单结构 192 原子；默认 train 最大 160 原子。
原始获取与父处理池形成于 2026 年 9 月，结构补全于 10 月 4 日；这不是上游实时镜像。
父表记录 MP 2026.04 系列、PhononDB PBEsol 和 JARVIS optB88vdW 等不同来源；本包没有完整的逐任务计算参数快照。

## 从来源到训练输入的处理逻辑

`来源获取与 ID 映射 → 配对双谱 → 统一坐标轴与离散化 → 尖峰处理 → 固定划分 → Q1 质量筛选 → 完整结构修复 → loader 归一化`

1. **统一材料身份。** 通过 MP summary 将历史 MP ID 及外源映射到 canonical MP ID，按材料去重。
   输入结构取 MP summary；eDOS 按可归属的 MP DOS task 选择。映射和血缘筛查不等于已经证明结构与两类谱的所有计算条件一致。
2. **选择谱来源。** eDOS 只取 MP；phDOS 优先级为 MP `pheasy` → MP `dfpt` → PhononDB → JARVIS。
   PhononDB 由发布的 phonopy 输入在本地以 `20×20×20` 网格、四面体法复算 DOS。
   外源 phDOS 不意味着 eDOS 也来自该外源。
3. **统一轴与纵轴变换。** eDOS 使用其**自身 DOS task** 的费米能级作为零点，合并 up/down 通道；缺少 down 时只保留已有总量。
   THz 声子轴乘 `33.35641` 转为 cm⁻¹，密度同时除以该因子，以保持这一步的积分。
   JARVIS 声子原强度按完整输入原胞的 `3N / 原生全谱面积` 缩放；MP/PhononDB 保留原生 DOS 计数尺度，不再统一强制重归一到 3N。
4. **离散化与覆盖。** 有限源点按左闭右开区间分箱。至少两个点时，取箱内点的梯形积分除以箱宽；
   只有一个点时取该点值；没有点时存 0 并置 `mask=False`。
   箱边界不插值，因此这不是严格积分守恒的重新分箱；mask 表示采样点覆盖，不能把未覆盖的 0 当成测得的零态密度。
5. **尖峰处理。** 父处理池按每个 bin 的已覆盖值计算 p99.9；超阈且相邻值低于峰值一半的孤立尖峰截到阈值，保留宽峰和肩峰。
   这些阈值是**历史父池级**预处理，未限定为 train 拟合；本次发布保留既有结果，不重算标签。
6. **划分与 Q1 筛选。** 父池 24,988 条先作约 8:1:1 分层分组划分，再排除 1,682 条 Q1 不合格记录，保留原划分和顺序。
   其中 1,668 条满足全谱积分 `C_full/N_val < 0.5`；另 14 条原谱不可读且尺度标签异常低。
   `C_full` 不是 E0 窗口总量；`N_val` 优先由占据电子谱及晶胞换算审计，必要时由元素价电子表回退。
7. **补全结构。** MP 结构采用 pymatgen 标准原胞，`symprec=0.1 Å`。旧缓存曾截到 80 原子；当前版本补全其中 55 条，
   修正 54 条组成回退价电子数，谱值和原行序保持不变。
   `mp-aaabidtd`、`mp-aaabilxy` 修正后的完整率低于既定阈值，默认 train 过滤，但磁盘仍保留。
8. **训练时变换。** 磁盘存处理后的 DOS 密度值。当前 SumNorm 在 loader 中逐谱除以离散值之和，得到无量纲谱形；
   全零谱在加小量的分母下仍保持全零。这个和不是直接的物理积分。窗口总量需要结合箱宽；
   模型另行学习谱尺度，blind/oracle 不是两套发布标签。

当前研究工作配方使用每原子非负无固定上限的窗口总量，以及train谱能量归一化的blind全谱误差加独立谱形辅助。
零谱的辅助项为0、全谱误差仍监督；新blind重建不使用逐材料`N_val`，正式loss/评分不使用coverage删箱。
这是模型协议更新，未改发布数组、mask、划分或历史质量筛选；具体公式与适用范围见[当前状态](status.md)。

父处理方法可在 Git `e24d2a4^:tools/getdata/a4_process.py`、`a4_pass2.py`、`recompute_phonondb.py` 核对。
Q1 和完整结构步骤分别见 `tools/data/q1_rebuild.py`、`tools/data/migrate_crystal_q1.py`。

## 坐标轴、单位与字段

| 标签 | 横轴 | 网格 | 磁盘纵轴含义 |
|---|---|---|---|
| eDOS | `E−E_F`，eV | E0：[-6, 6]，128 箱，宽 0.09375 eV | MP 原 DOS 计算晶胞的 states/eV，经分箱及尖峰处理；未按原子数统一相除 |
| phDOS | 频率，cm⁻¹ | P0：[-280, 980]，64 箱，宽 19.6875 cm⁻¹ | 换算后的模式密度，含 JARVIS 的 3N 幅度约定；有限窗口与分箱后不保证积分恰为 3N |

纵轴是**密度而非每箱积分**；不能直接称为“每原子 DOS”。原 DOS 计算晶胞与当前输入晶胞未逐条认证一致。
负声子频率区域保留上游的虚频表示；不能据此把全库视为动力学稳定材料。

每个 `data/crystal_q1_v1/{split}/` 含以下 11 个数组，`s` 为 split，`M` 为材料数，`S` 为该划分原子总数：

| 文件模式 | 形状 / dtype | 含义 |
|---|---|---|
| `cell_s.npy` | [M,3,3] / float32 | 晶格行基矢，Å |
| `frac_coords_s.npy` | [S,3] / float32 | 分数坐标，无量纲；笛卡尔坐标为 `frac @ cell` |
| `atomic_numbers_s.npy` | [S] / uint8 | 原子序数 |
| `atom_offsets_s.npy` | [M+1] / int64 | 第 i 个结构取 `[offsets[i]:offsets[i+1]]` |
| `n_atoms_s.npy` | [M] / int32 | 完整原子数 N |
| `nvalence_s.npy` | [M] / float32 | 审定/回退价电子数，用于历史H1尺度监督；非严格独立于标签的输入特征，新blind重建不用它 |
| `s_index.npy` | [M] / Unicode | canonical MP 材料 ID，所有材料字段按此顺序对齐 |
| `edos_tgtdos_s.npy`、`phdos_tgtdos_s.npy` | [M,128]、[M,64] / float32 | 处理后的密度标签 |
| `edos_mask_s.npy`、`phdos_mask_s.npy` | [M,128]、[M,64] / bool | True 表示该箱存在有限源采样点；是否用于损失由训练配置决定 |

磁盘不保存原子 padding 或晶格哨兵。`manifest.json` 记录 shape、dtype、哈希、修复和质量隔离。
包内 `data/reference/` 的 8 个文件提供固定划分、网格、价电子回退表、排除清单及历史价电子/截断审计：
`split_v2.yaml`、`grids.json`、`c2b_grids_report.json`、`q1_excluded_records.json`、
`valence_by_element.json`、`edos_valence_by_material.parquet`、`edos_truncation_by_material.parquet`、`edos-label-audit.md`。
其中 split 与审计表可能覆盖父池 24,988 条；应以当前 ID 过滤，54 条修复值以当前 manifest/数组为准。
加上当前缓存的 33 个数组和 manifest，压缩包共 42 个文件，均与本地冻结版本逐字节一致。

## 划分的实际边界

父划分按**绝对原子计数组成**分组，兼顾带隙档、晶系和谱难度，记录 seed42；不是每个材料独立随机划分。
历史算法使用当时的最多 80 原子组成。此次对补全后的全体存储结构重新核查，材料 ID 和完整绝对组成均无跨集重复。
但约分化学式并未完全隔离：train–valid / train–test / valid–test 分别共享 **254 / 253 / 44** 个约分组成。
这些是组成集合交集的大小，不是重复材料数，不能宣传为未见化学式泛化基准。

模型开发和选点只用 train/valid。历史 test 曾参与诊断，不能称作从未触碰的最终独立留出集。
本次仅核查 test 的文件、来源、组成和标签一致性，没有运行模型或计算性能指标。

## 获取、恢复与来源追溯

代码由 Git 保存，数据以 GitHub Release 附件候选准备；数据压缩包不进入普通 Git 历史，准确大小与哈希见 `release-manifest.json`。
候选目录包含 `README-data.md`（本页）、`README-data.en.md`、`SOURCE_NOTICE.md`、数据包、
`material-provenance.csv.gz`、`provenance-audit.json`、`release-manifest.json`、`verification.json` 和 `SHA256SUMS`。
另附 `mp-license-records.csv.gz`、`mp-license-audit.json`，保留逐条 MP 许可核查；旧元素表及其专项审计不作为本次发布附件。
逐材料来源表含 ID、split、行号、默认可用标记、谱来源/引用、全零窗口标记、历史结构引用及历史 `ph_weight`。
**历史权重是追溯元信息，当前 loader 不读取它，不代表训练已按来源降权。**
MP 声子引用仅记录方法类别，未恢复唯一 task ID；表中的来源 URL 按标识构造，不保证链接长期有效。

将全部附件放到干净的代码目录。新克隆中已有 manifest 和参考文件属于正常情况；下面先检查同名文件一致，
只允许补齐缺失文件或覆盖相同内容，发现不同版本立即停止：

```bash
sha256sum -c SHA256SUMS
python - <<'PY'
from pathlib import Path
import hashlib, json
for name, item in json.loads(Path('release-manifest.json').read_text())['files'].items():
    path = Path(name)
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], name
print('Existing payload files match; ready to extract.')
PY
tar -xzf uniARPAT-crystal-q1-v1.tar.gz
```

装好项目依赖后，使用 `utils.crystal_cache.crystal_data_contract` 校验数组，再经 `utils.dataset.Dos_Dataset` 加载。
无需旧权重或原服务器绝对路径；恢复验证已通过。完整上游档案与所有处理环境未收录，
所以“恢复本次冻结数据”可复现，不等于“联网重新采集可逐字节重建”。
来源元信息可由 `tools/data/export_release_provenance.py --cache ... --processed ... --out <新目录>` 重导，
该工具只读既有缓存/父表、校验谱与 mask 一致性，不重算标签或划分。

## 已知限制、引用与维护

多源泛函、磁态、计算几何及细致设置未完全统一。历史 valid 诊断的 394 个材料中，125 个输入结构与 DOS 结构未通过严格匹配；
此统计仅适用于该子集，不外推成全库错误率。有限频率窗口、采样覆盖和尖峰处理均会影响谱面积与细节。
当前 phDOS 全零窗口谱有 train 7 条、valid 1 条、test 0 条，均与父处理表相同，默认视图仍保留；
eDOS 无全零谱。全零不代表已认证的物理零谱，原因尚未逐条归因；可由来源 CSV 的 `phdos_zero_window` 定位，改变过滤须另定数据版本。
引用时写明 `crystal-q1-v1`、数据清单哈希、实际 train 过滤、网格和归一化，并引用 MP、PhononDB、JARVIS 原来源。
本数据集尚无独立论文/DOI，不虚构作者或 DOI；正式发布后补充 Release 的固定 URL。原来源引用与许可说明见来源说明。

文档组织参考 [Matbench 的数据集元数据](https://github.com/materialsproject/matbench/blob/main/matbench/matbench_v0.1_dataset_metadata.json)、
[MPtrj 的版本与引用说明](https://github.com/CederGroupHub/chgnet#mptrj-dataset)和
[Hugging Face Dataset Cards](https://huggingface.co/docs/hub/datasets-cards)。这些是说明方式参考，不是本包数据来源。
问题与更正通过本项目仓库 Issues 记录；改变成员、谱值、网格或完整结构契约须发布新数据版本。
