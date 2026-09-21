# 设计：D4c JARVIS `min_fd_phonon_mode` 语义与来源限定审计

> 对应 `docs/status.md` 的当前候选。D4c 是只读审计，不删除、重标、重建 Q1，也不启动训练。

## 目标与成功判据

- **问题：**JARVIS 原始 phDOS 的 `min_fd_phonon_mode` 是否可明确追溯为有限位移声子计算的数值字段？
  它的 `-0.0` 能否当作负值，以及它是否实际等于同一记录可见的最低 `phonon_modes` 或 DOS 频率？
- **已知边界：**D4b 已确认此字段只覆盖 Q1 test 中 JARVIS 的 209 条。它不是 MP/PhononDB 的共同字段，
  也不是显式的计算收敛 flag；任何结论均不能跨来源推广。
- **预注册判定：**
  - 原始字符串精确为 `-0.0` 归为 **zero-serialized**，数值上等于零，绝不归为 negative；
  - 仅数值严格小于零的值归为 **negative**；零与正值分别单列；不事后选择数值阈值；
  - 分别比较 `min_fd_phonon_mode`、`min(phonon_modes)` 和 phDOS 最小频率。只有三者在数值上逐行一致，
    才可称前者为该存储记录的全局最低模式；否则只使用来源原样字段名；
  - 报告各类的 D4 high 与 B7 phDOS 失败率，但仅为探索性描述，不作为新的删样本阈值或显著性结论。
- **完成判据：**所有 209 条 JARVIS test 记录与 D4b 的冻结 `ph_ref` 精确匹配；字段来源路径、字符串类别、
  三种最小值的逐行一致性和缺失数都写入机器可读结果。

## 改动

- 新增 `tools/eval/d4c_jarvis_min_fd_semantics_audit.py`，只读取 D4b 逐样本来源表、Git 外 JARVIS 原始
  JSONL 和本地 `jarvis.db.webpages.Webpage.get_dft_phonon_dos` 的源码位置；流式保留所需的 209 条。
- 输出逐样本字段比较、字符串分类和描述性分层到 `results/`；缺失、解析失败或引用不匹配必须显式记录，
  不能以零填充。
- 新增合成单测，锁定 `-0.0` 不为负、逗号数列解析及“字段相等”判断；不将 Git 外大归档放进 CI。

## 测试关卡

- 合成合同测试及 E7 静态门禁。
- 本地完整只读命令：

  ```bash
  python3 tools/eval/d4c_jarvis_min_fd_semantics_audit.py \
    --raw-root /root/home/newstudy/getdata/raw
  ```

- 审计前后 Q1 缓存、标签和划分的哈希／文件均不得被写入或改变。

## 成本与风险

- JARVIS raw JSONL 约 14 GB；必须顺序流式读取，且只保留预先固定的 JARVIS test 引用。
- 即使数值小于零，也可能受有限位移、结构弛豫、超胞、声子路径或显示精度影响；D4c 不将其解释为实验
  不稳定性、收敛失败或标签错误。
- 若该字段与 `phonon_modes`／DOS 最小坐标不一致，D4 在此关闭，不提出数据策略；只有字段语义、来源边界和
  独立数据政策都清楚时，才能另起设计。
