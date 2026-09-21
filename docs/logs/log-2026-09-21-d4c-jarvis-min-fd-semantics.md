# 日志：2026-09-21 — D4c JARVIS `min_fd_phonon_mode` 语义审计

## 范围

- 任务与假设：核验 JARVIS-only 的 `min_fd_phonon_mode` 来源、`-0.0` 语义，以及它是否是同一条记录的
  全局最低 `phonon_modes` 或 phDOS 频率；不将字段转成数据质量标签。
- 改动的文件或配置：新增 D4c 只读审计、合成测试和设计；没有改动 Q1 缓存、标签、划分、模型或训练。

## 证据

- 本地已安装的 JARVIS 客户端 `jarvis.db.webpages.Webpage.get_dft_phonon_dos` 将该 payload 追溯至
  `basic_info.main_elastic.main_elastic_info`，即 JARVIS-DFT 的 `MAIN-ELAST` 有限位移声子 DOS 路径。
  这说明字段来自特定来源计算，而不是 uniARPAT 自行生成的稳定性 flag。
- 命令：

  ```bash
  python3 tools/eval/d4c_jarvis_min_fd_semantics_audit.py \
    --raw-root /root/home/newstudy/getdata/raw
  ```

  以 D4b 冻结的 `mpid + ph_ref` 为键，209/209 JARVIS test 原始 payload 都唯一匹配；没有读取或写入 Q1
  缓存。
- 按原始字符串预注册分类：35 条严格小于零、7 条普通 `0.0`、**167 条 `-0.0`**。`-0.0` 按 IEEE 数值为
  零，保留其序列化方式但不归为负值。严格负值组的 B7 phDOS 失败率为 25.71%（9/35），只是探索性描述，
  不是新阈值或新显著性检验。
- 语义反证：`min_fd_phonon_mode` 与同记录 `min(phonon_modes)` 只有 7/209 条完全相等；与
  `min(phonon_dos_frequencies)` 为 0/209 条完全相等。严格负值组的两项后者中位最小值为
  −72.25 和 −104.69，而 `min_fd` 本身并不相等。因此它**不能**被称作当前存储声子模式表或 DOS 网格的
  全局最低频率；只能保留为 JARVIS `MAIN-ELAST` 来源字段。
- 机器可读结果：`results/d4c_jarvis_min_fd_test_samples.csv`、
  `results/d4c_jarvis_min_fd_category_summary.csv` 和
  `results/d4c_jarvis_min_fd_audit_summary.json`。
- 验证：`python3 -m unittest tests.test_d4c_jarvis_min_fd_semantics_audit -v` 为 3/3 通过；相关 Ruff 与
  编译通过；其后运行 E7 静态门禁和全量测试。

## 结论

- 状态：**closed。**`min_fd_phonon_mode` 可追溯为 JARVIS `MAIN-ELAST` 的来源元数据，但其 167 个
  `-0.0` 不是负值，且它不等于记录中可见的全局最低模式或 phDOS 网格下界。它既不是跨来源真值，也不是
  计算收敛 flag；D4 系列不授权任何 JARVIS-only 或全库的删样本、重标、重建、损失改动或训练。

## 交接

- 下一项关卡工作：D4 数据路线关闭。C3 PhysMoE 与 D3 eDOS 辅助数据仍没有获批的单因素设计，不能自行
  启动；需要先选择新的研究假设。
- 对 status、backlog 和 decisions 的更新：D4c 的字段边界写入 `status.md` 和 `decisions.md`；默认 Q1/
  B7 不变。
