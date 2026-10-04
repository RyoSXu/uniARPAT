# 诊断钩子清理的静态检查修复

日期：2026-10-04。用户提供 `60acc74` 对应的 CI 失败日志：仓库静态检查报告七项 F821，
均位于 [`element_identity_diagnosis.py`](../../tools/eval/element_identity_diagnosis.py) 的 `phase_c`。

## 已核实事实

- 直接读取远端 `main` 得到 `60acc7452620cf733ec7260f16227a8eb2888955`，该提交已到达远端。
  失败日志来自 `.github/workflows/ci.yml` 中的 `check-static.sh`，不能将 CI 失败等同于推送未到达。
- GPU 批次共 12 个文件，未压缩内容合计 1,575,154 字节（约 1.50 MiB）；最大文件
  `cases.json` 为 641,380 字节（约 0.61 MiB）。确实归档了 CSV/JSON/JSONL 明细，其中 JSONL
  显式纳入 Git；本次提供的失败证据是 F821，没有体积限制报错。
- 原清理代码在六批归档之前的 `921f468` 已存在，对应文件最近改动来自 `78fcfd0`。
  本地 Ruff 0.16.8 运行仓库检查，复现同样七项错误。归档时只查了语法与针对性证据，
  漏跑仓库完整静态检查，本次补齐。

## 修改与边界

只修改 `phase_c` 的批次末尾清理：原先 `del clean, hooked, layer_outs, captured` 删除了
钩子闭包引用的变量名称。现在在钩子移除、统计使用完毕之后执行 `layer_outs.clear()` 与
`captured.clear()`，释放字典持有的张量并保留名称绑定，随后仍删除 `clean`、`hooked`。
没有新增/删除公共符号，也未添加 F821 忽略或修改 Ruff 配置。

模型计算、局部 A/B+ 核心、数据格式、历史结果与 manifest 未修改；旧诊断源码仍可从 Git
历史追溯。没有重跑历史 `phase_c` 或据此改写其科学结论。输入拆分讨论稿保留未提交。

## 验证

修复前运行同一入口得到七项 F821；修复后运行：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONWARNINGS=ignore::DeprecationWarning bash tools/ci/check-static.sh
```

结果为 Ruff 通过、所有受版本控制的 Python 文件编译通过，**182 项契约测试通过**，
测试主体用时 51.479 秒，无失败。检查使用 CPU；部分既有契约用例会用合成小数据做几步
模拟更新，没有启动真实数据训练、历史诊断或 GPU 作业。

本次将修复单独追加提交，保留原六批历史；未执行推送。远端 CI 对新修复的结果尚待后续运行，
本地通过不能替代远端环境的完成状态。忽略规则和已归档结果均保持原样。
