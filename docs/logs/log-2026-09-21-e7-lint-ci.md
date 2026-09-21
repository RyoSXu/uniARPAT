# 工作日志 2026-09-21：E7 最小 lint/CI 门禁

## 范围

- **目标：**把不依赖 Q1 本地缓存的致命静态错误和核心 CPU 合同固定为本地与 GitHub Actions 可重复执行的门禁。
- **改动：**新增 `ruff.toml`、`tools/ci/check-static.sh` 和 `.github/workflows/ci.yml`；新增设计页并更新索引。
- **边界：**没有修改模型、损失、默认 YAML、数据缓存、checkpoint 或实验命令；没有启动训练。

## 证据

- `bash tools/ci/check-static.sh`：通过。它对所有受 Git 跟踪的 Python 源码运行 Ruff 的
  `E9,F63,F7,F82`、`compileall`，再执行 11 个无数据测试模块，共 **36 项**测试通过（1.42 秒）。
- `python3 -m unittest discover tests`：完整本地回归 **87/87** 通过（24.27 秒）。完整套件仍包含必须读取
  Q1 缓存的集成测试，故不被纳入干净 CI；E7 自身没有改变其代码或数据前提。
- workflow 经 Python YAML 解析，作业名 `static-contracts` 存在；`git diff --check` 通过。
- 初次引入 Ruff 发现 `model/model.py` 以 `locals()` 安全暴露可选损失槽时触发 4 个 `F821` 静态误报。
  `ruff.toml` 仅对这个文件、这个规则作注释化排除；其余受检路径保持严格检查。

## 结论

- **状态：完成。**仓库已有一个缓存无关、可由 `bash tools/ci/check-static.sh` 重现的最小 CI 门禁；
  GitHub Actions 将在 push 与 pull request 上安装声明的依赖后运行它。
- E7 不是训练、数值或准确率结论；Q1 相关集成回归仍须在有缓存的本地环境运行
  `python3 -m unittest discover tests`。

## 交接

- 下一项候选为 B7 M1 的 CIF 盲推理入口：先审计现有 `cif2dos.py` 的 M4 假设和 checkpoint／输出边界，
  再写设计，不直接修改或删除旧兼容工具。
- 已更新 `status.md`、`decisions.md` 与 `index.md`。
