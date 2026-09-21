# 工作日志 2026-09-21：D4 phDOS 尖峰与负频坐标质量审计

## 范围

- **问题：**B7 的 phDOS 失败是否集中在 P0 负频坐标质量高或单 bin 尖峰集中的标签？
- **范围：**只读 Q1 train/valid/test 的 phDOS、coverage mask、P0 网格和 B7 `_e9ctl` test 明细；不读
  checkpoint、不重新预测、不训练、不改标签或划分。
- **语义边界：**负频是 P0 的坐标而不是负 DOS 值；缓存没有可核验的动力学稳定性真值，故这里的
  `negative_mass_fraction` 是标签形状代理，不能被表述为已证实虚频或标签错误。

## 证据

- 新增 `tools/eval/d4_phdos_spike_imaginary_audit.py`。P0 的 64 个 bin 中有 14 个中心小于零
  （`−270.156` 至 `−14.219 cm⁻¹`）。按 train p90 固定阈值：负频坐标质量为 **0.129287**，单 bin
  尖峰占比为 **0.271307**；阈值不使用 valid/test 结果选择。
- 正式命令：

  ```bash
  python3 tools/eval/d4_phdos_spike_imaginary_audit.py
  ```

  输出 `results/d4_phdos_{descriptor_summary,b7_strata,b7_test_samples}.csv` 与
  `results/d4_phdos_audit_summary.json`；test 明细为 2,287 行并以固定 test index 对齐 B7 明细。
- B7 phDOS 分层结果（固定 seed 42、2,000 次分层 bootstrap）：

| train p90 描述量 | test high / other n | high / other 中位 R² | high / other 失败率 | 失败率差 95% CI | D4 门槛 |
|---|---:|---:|---:|---:|---|
| 负频坐标质量 | 230 / 2,057 | 0.574 / 0.752 | 16.09% / 2.09% | +9.41 至 +18.73 pt | 通过 |
| 单 bin 尖峰占比 | 226 / 2,061 | 0.833 / 0.734 | 4.42% / 3.40% | −1.58 至 +4.07 pt | 不通过 |

- 三个 Q1 split 的 coverage 外目标质量均为 0，故它只作为完整性结果保留，不构造退化的 high/other
  失败率比较。
- `python3 -m unittest tests.test_d4_phdos_spike_imaginary_audit`：2/2 通过；
  `bash tools/ci/check-static.sh`：42 项缓存无关合同通过；
  `python3 -m unittest discover tests -q`：完整本地回归 93/93 通过（31.75 秒）。

## 结论

- **状态：完成，有可行动关联。**B7 phDOS 的失败强烈集中于 train p90 以上的负频坐标质量代理；它满足
  `n≥100`、失败率差≥3pt、bootstrap 区间下界>0 的预注册门槛。尖峰集中度不满足门槛，不能以“尖峰损失”
  为由启动模型或损失改动。
- 该关联只授权下一步核验该 230 条样本的原始来源／稳定性／单胞语义，不能据此删除样本、改变 P0、重标标签
  或启动 D3／训练。

## 交接

- 下一关卡为 **D4b 负频坐标来源与稳定性审计设计**：先确认是否有可追溯的原始频率、虚频标识或计算收敛
  元数据；若没有，D4 的结论只能保留为失败分层事实。
- 已更新 `status.md`、`decisions.md`、索引和 E7 静态合同。
