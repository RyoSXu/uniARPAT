# 日志：2026-09-21 — D4b 负频坐标来源与稳定性审计

## 范围

- 任务与假设：对 D4 的 B7 phDOS 高负频坐标质量组进行逐材料来源回连，辨别原始频率/DOS、稳定性字段
  与收敛字段的实际可用性；不把 P0 标签形状当作虚频真值。
- 改动的文件或配置：新增只读脚本
  `tools/eval/d4b_negative_coordinate_provenance_audit.py`、合成测试、D4b 设计和本日志；没有改动 Q1
  缓存、P0、标签、划分、损失或训练配置。

## 证据

- 命令：

  ```bash
  python3 tools/eval/d4b_negative_coordinate_provenance_audit.py \
    --raw-root /root/home/newstudy/getdata/raw
  ```

  审计流式读取 Git 外冻结原始 archive，并以 Q1 `test_index.npy` 对齐 D4 的 2,287 行。全部 2,287
  条记录都一对一匹配 `v2_processed.parquet` 的 `mpid/src_ph/ph_ref`，并找到与冻结 `ph_ref` 相符的原始
  phDOS；不存在静默回退或缺失填补。
- D4 的 230 条 high 中，210 条是 MP `pheasy`、20 条是 JARVIS、0 条是 PhononDB；全 test 的来源数为
  MP 1,835、JARVIS 209、PhononDB 243。high 并非单一来源的全部样本（MP/JARVIS 的 high 占比分别
  11.44%/9.57%）。
- 原始频率-DOS 负坐标质量的中位数与 D4 分层一致：MP high/other 为 0.25478/约 0，JARVIS high/other
  为 0.21111/约 0。因此该关联可在重分箱前的冻结原始谱上复现，但它依然只是 DOS 坐标质量，不是模式
  稳定性标签。
- B7 phDOS 失败率的来源内描述性分层仍存在：MP high/other 为 11.90%/2.03%，JARVIS high/other 为
  60.00%/2.65%。这些子组不是 D4 预注册的新的显著性检验，尤其 JARVIS high 只有 20 条，故不作为
  删除或重训判据。
- MP 原始记录含 `method/run_type/has_structure`，PhononDB 含 `mesh/tetrahedron/born/dielectric`；两者
  均没有逐材料二元稳定性或收敛标识。JARVIS 含数值型 `min_fd_phonon_mode`，覆盖其 209 条：其中严格
  小于零的 35 条有 25.71% 失败率，非负的 174 条有 4.60%；D4 high 的 20 条中 16 条小于零。这是来源
  提供的有限差分模式最小值，不是跨来源可比的收敛 flag，也不覆盖 MP/PhononDB。
- 机器可读结果：
  `results/d4b_phdos_provenance_test_samples.csv`（逐样本）、
  `results/d4b_phdos_provenance_summary.csv`（来源×D4 分层）、
  `results/d4b_phdos_provenance_field_catalog.csv`（字段边界）和
  `results/d4b_phdos_provenance_audit_summary.json`（结论）。
- 验证：`python3 -m unittest tests.test_d4b_negative_coordinate_provenance_audit -v` 为 3/3 通过；
  `ruff check tools/eval/d4b_negative_coordinate_provenance_audit.py
  tests/test_d4b_negative_coordinate_provenance_audit.py` 通过；`tools/ci/check-static.sh` 已纳入这 3 项，
  合计 45 项无缓存合同；脚本的完整 archive 审计成功结束。

## 结论

- 状态：**完成，且无全库数据处理授权。**D4 的高组可追溯到真实的原始频率-DOS 质量，并非 P0 缓存
  伪影；但全库没有可用的逐材料二元稳定性或收敛真值。JARVIS 的 `min_fd_phonon_mode` 是一项仅限
  209 条来源样本的数值元数据，不能推广到 MP/PhononDB，也不足以直接删除、重标或重建 Q1。

## 交接

- 下一项候选工作：若要继续数据方向，必须先单独设计 **D4c JARVIS 模式最小值语义与来源限定策略审计**，
  核验该字段的计算约定、`-0.0` 语义和其是否适合形成仅 JARVIS 的数据政策；否则 D4 在此关闭。
- 对 status、backlog 和 decisions 的更新：D4b 的可追溯性结论写入 `status.md` 与 `decisions.md`；不改
  Q1 或默认训练方案。
