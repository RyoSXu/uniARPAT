# 设计：E7 最小 lint/CI 门禁

> 对应 `docs/status.md` 的当前工程关卡。E7 保护已存在的代码和合同；它不训练模型、不读取
> Q1 缓存，也不改变任何实验配置。

## 目标与成功判据

- 在干净 clone 的 GitHub Actions Ubuntu runner 上，以 Python 3.11 安装项目声明的运行依赖后，
  自动完成 Python 编译、最小静态 lint 与无数据单元测试。
- 同一检查应能由开发者通过一个仓库脚本在本地重现；测试选择必须显式列出，而不是依赖当前机器
  恰好存在的 Q1 缓存。
- 所有检查通过，且 B7、训练命令、默认 YAML、数据缓存和 checkpoint 格式均不变。

## 改动

- 新增 `tools/ci/check-static.sh`，按固定次序执行：
  1. `ruff` 的致命语法／未定义名称规则（`E9,F63,F7,F82`）；不将尚未统一格式化的历史代码伪装成
     全仓风格门禁。`model/model.py` 中已有的 `locals()` 条件损失槽有一个受注释的 `F821` 例外，
     因为静态分析无法判定其动态保护。
  2. 对受版本控制的 Python 入口、`model/`、`datasets/`、`utils/`、`tools/eval/` 和 `tests/` 运行
     `compileall`。
  3. 运行 11 个不加载 Q1、checkpoint 或标签的测试模块。
- 新增 `.github/workflows/ci.yml`：push 与 pull request 均运行该脚本，使用 Python 3.11、`pip install
  -r requirements.txt` 和独立安装的 `ruff`。
- 无数据测试包含模型头、P0、热力学、注意力／归一化、G1、Q2、C5、C4、E5 和 C2.1b 合同；明确排除
  读取 Q1 缓存的 R1、R2、G2、E6、E10、Q1 坐标与 L3 测试。这些测试继续由本地
  `python3 -m unittest discover tests` 覆盖。

## 测试关卡

- 在当前环境运行 `bash tools/ci/check-static.sh`。
- 运行 `python3 -m unittest discover tests`，确认 E7 没有破坏完整本地回归套件。
- 对 workflow 与 shell 脚本做 YAML／shell 可执行性和路径审查；CI 本身在推送后由 GitHub 执行，
  不把未推送的工作流误报为已远端验证。

## 成本与风险

- CI 安装完整科学依赖，首次运行比只检查语法更慢；换来与项目真实导入边界一致的检查。
- `ruff` 仅检查确定会造成运行失败的规则，避免以一次 E7 改动引入无关的全仓格式重写。
- CI 不携带 Q1 数据，也不应下载或生成它；所有数据相关集成测试留在有意配置缓存的本地环境。
