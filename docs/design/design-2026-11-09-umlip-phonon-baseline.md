# 方案：2026-11-09 — uMLIP＋phonopy 的 phDOS 基线（路线阶段 A）

## 问题

文献显示通用机器学习势（uMLIP）配合 phonopy 已能接近 DFT 精度地计算声子（[Loew et al., npj 2025](https://www.nature.com/articles/s41524-025-01650-1)：
约 1 万个材料上，MatterSim-v1 的 phDOS MAE 3.3，PBE 与 PBEsol 之差 2.9；MACE-MP-0 为 4.2）。
直接回归 phDOS 的价值须正面回答：在我们的 valid 上，它与“uMLIP 直接算声子”相比是否更准，或同等准确而快多少。

## 方法

- `tools/data/umlip_phonon_baseline.py`（MACE 环境，含 phonopy 2.46）：对 valid 每个材料，用模型输入的同一晶胞与原子序，
  MACE-MP-0b3 medium（float64）→ 有限位移（0.01 Å，超胞各垂直高度 ≥ 10 Å）→ 力常数对称化 → 网格（长度 100）四面体法总 DOS →
  按数据卡约定转 cm⁻¹（密度除以 33.35641），在 P0 窗口按数据发布的分箱规则得到 64 箱、每输入晶胞的 DOS。
- 两个变体：
  - **relaxed**：先用 MACE 弛豫晶胞与原子（FIRE＋FrechetCellFilter，fmax 0.005 eV/Å，≤500 步），即标准 uMLIP 声子流程；
  - **input**：不弛豫，直接用模型输入结构，与学习模型的输入条件完全相同。
    2026-11-09 运行中调整（全量 relaxed 实测约 9 秒/材料，长于预检估计）：input 只跑随机 500 个 valid 材料（seed 20261109）；对比分两次——relaxed 单独在全部完成的材料上（主结论），两变体在共同的 500 个材料上（辅助）。
- `tools/eval/umlip_phonon_compare.py`（训练环境）：在两个变体都成功的材料上，用训练公式算 phDOS blind E、按目标窗口总量缩放后的 E（只看谱形）、
  中位 R²，与 h2、M1 的 valid 导出做材料配对 bootstrap；另报失败数、虚频比例（最低网格频率 < −20 cm⁻¹）、弛豫未收敛数与耗时。

## 口径与局限

- 标签 80% 来自 MP（pheasy/DFPT），其余来自 PhononDB（PBEsol）与 JARVIS（optB88vdW）；MACE 训练于 PBE（MPtrj）。泛函差异会使 uMLIP 偏软。
- 标签窗口总量与 3N 不一致（valid 上窗口积分/3N 的中位数 0.86，1%–99% 分位为 0.39–0.98），uMLIP 输出的是物理的 3N 尺度，
  因此另报只看谱形的缩放 E。
- 本地没有逐材料的谱来源表，暂不分来源报告。
- 只用 valid，不读 test，不用于任何选型；MatterSim 等更强的 uMLIP 留作后续补充。

## 预检

6 个 valid 材料试算（relaxed）：每个 2–6 秒，弛豫均收敛，体积比 0.97–1.06，2/6 出现虚频；最大 valid 材料（136 原子，272 原子超胞，102 个位移）不弛豫约 28 秒。
全量 valid 估计每个变体 2–4 小时（V100）。测试 `tests/pipeline/test_umlip_phonon_baseline.py`。

## 判读（事先说明，不作采用决定）

- uMLIP 的 phDOS E 明显低于 h2（配对区间不含 0）→ 直接回归在精度上不占优，phDOS 定位为“快速筛选”或改为以 uMLIP 声子为先验的混合方案；
- 两者分不开或 h2 更好 → 直接回归在同等或更高精度下快约 3 个数量级，phDOS 主线成立。
- 结论会同时参考按谱形缩放的 E，以区分尺度约定差异与谱形本身的差异。
