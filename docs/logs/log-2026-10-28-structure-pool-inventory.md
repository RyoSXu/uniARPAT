# 日志：2026-10-28 — 本地额外结构盘点

只读。不下载、不训练、不把 eDOS-only 候选改成获准数据。

## 查了什么

当前可直接读取几何的晶体只有 `data/crystal_q1_v1/`。
2026-09-28 记录的外部结构目录 `/root/home/newstudy/getdata/raw/pretrain_structures`、
`census_effective_ids.json` 和 `v2_processed.parquet` 在这台机器上都不存在。
仓库内没有另一套 `structures_*.jsonl` 或结构 parquet。

## 计数

组成按原子序号和个数。绝对组成不约分，约化组成把计数除以最大公约数。

| 划分 | 材料 | 绝对组成 | 约化组成 |
|---|---:|---:|---:|
| train | 18,706 | 16,922 | 15,745 |
| valid | 2,313 | 2,089 | 2,073 |
| test | 2,287 | 2,076 | 2,061 |

train 与 valid/test 的材料 ID 重叠为 0，绝对组成重叠为 0。
train 有 1,021 条材料和 valid/test 的约化组成相同，涉及 487 个约化式。
去掉 valid/test 的 ID 和绝对同组成之后，本地还剩的就是这 18,706 条 train，没有额外结构。

`q1_excluded_records.json` 有 1,682 个材料 ID，只是旧排除名单，旁边没有对应几何。

## 结论

预训练方案停在数据这一步。历史日志里的 121,420 是当时外部目录的名单计数，不是现在可训练的结构数。
不临时更换来源，也不把那批 eDOS-only 候选自动纳入训练。
