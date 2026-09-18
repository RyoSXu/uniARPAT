# 日志-2026-09-18-审计与workflow打磨

## 改了什么
- 修回归：`datasets/dataset.py` 加 `coords="auto"`，生产网格（128/64）照常返回 17 元组，非生产网格（C2b E1/E2/P1/P2）回退 15 元组，不再硬断言。`model/model.py` 本就兼容 `len>15` 判空，无需改。
- 补悬空引用：`index/z0_REPORT.md` 加附录 D1–D4（trunc 定义/1682 名单/口径/影响）。
- 导航层（参考 ml-research-template / docflow / ARA / AGENTS.md 标准）：
  新建 `docs/INDEX.md`（中英地图）、`docs/GLOSSARY.md`（口径+五态+人话缩写）、`docs/STATUS.md`（活状态 now/next）、`docs/README.md`；
  重写 `AGENTS.md` 为导航页（先读表+命令+地图+纪律+Session协议）；
  新建 `docs/03_工作日志/_TEMPLATE.md`、`docs/01_开发文档/_DESIGN_TEMPLATE.md`、`tools/eval/README.md`（verdict 脚本入库处）。
  中文历史目录名冻结保留，只加英文索引，不改名不搬家。
- Backlog：头栏基线改为 B7 Q1；Phase 2 补 B6 行、Phase 4 对照 B5→B7；执行状态推到 09-18；模型坐标 Q1/Q2 改称 Qc1/Qc2（区别数据 Q1 隔离）。
- Decisions：网格改 C2b 毕业 E0+P0；实测数分现行 B7/存档 v0；数据源 PhononDB/CRD/h1 移归档；新增第14条（H1正统、0.92退役、掩膜留评估端）。
- README：结构树重写；Quickstart 加数据前置、34项单测、Q1实测 190s/7.2GB、B7对照配方+双禁令；基线表换 B7+pre-Q存档行，旧1371数降级存档注；M1-M5 NOTE 去 `in progress`；M5 作废注+h1三否决注；ScaleHead 图改 H1 η/γ；Citation 改在投；License 改待补。

## 实测
- `python3 -m unittest discover tests`：34/34 OK（约87秒，CPU 机）。
- 数据集冒烟：生产 `test` 批长 17、有坐标；非生产 edges（162/114）回退批长 15，无崩。
- L3 pilot 仍进行中（`_l3kl/_l3now1` 已落，`_l3nohub` ep1），STATUS 保持“进行中”，未提前判。

## verdict
- 审计结论：数字主链一致可信；2真错误+1回归已修（断言/D1-D4/8.75GB 按历史不改+本志正名 7228MB）；可读性阻断项已改。
- workflow：AGENTS 稳定页 + STATUS 活页分离，符合 AGENTS-COLLAB（一静一活）与“导航页不记流水”原则。

## 下一步
- L3 verdict 落盘后按模板记日志 + 更新 STATUS + 结论同步 Backlog。
- 后续可选（未做）：`src/` 迁移不做（研究仓扁平布局可接受，先文档化边界）；`output/config_used.yaml` 归档、`results/README.md` 登记、verdict 脚本从 `/tmp` 迁 `tools/eval/`。
