# Design-A3：census收尾

**状态**：进行中（`census_rebuild.py`后台跑） ｜ **Backlog**：[Backlog待开发总清单.md](./Backlog待开发总清单.md)

## 背景
eDOS全集271,321已齐；声子映射历经id_format空跑→静默丢数→新老id双记三重bug，现队列式重跑。

## 执行
`python3 getdata/v2/census_rebuild.py`（resume-safe；失败进`census_queue.json`，永不静默丢）。

## 输出/验收
`census_phonon_map_canonical.json` + 双谱构成表（28–32k）；He复核；日志DONE且failed=0。
