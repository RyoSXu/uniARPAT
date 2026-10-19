# 日志：2026-10-19 — MP 许可与旧元素表专项核查

后续范围更正：本页保留旧表的历史调查证据。用户确认当前主线不使用旧元素性质表后，
已将旧周期表、mendeleev 特征快照及旧表审计排除出新发布候选；其授权问题不再作为本包待办。
本记录不为旧表新增授权，当前发布范围见[重新打包记录](log-2026-10-19-mainline-data-repack.md)。

## 范围与依据

用户要求继续核清 MP 完整条款与旧元素表来源/许可，并补充来源线索为 JARVIS、PhononDB。
开始时工作区包含上一轮数据卡及导出工具改动；全部保留。本轮只更新中英文说明和本地审计附件。
没有训练、模型评估、换表、重划分、修改数据数组、提交、发布或联系外部人员。

## MP 事实与结论

- 官网完整 Terms 页面仍返回 403；不伪称已逐字读完实时页面。一般条款历史正文及官方维护者解释来自
  [2022 快照答复](https://discuss.matsci.org/t/45437)、[2024 加工数据答复](https://discuss.matsci.org/t/56836)。
  明确允许按 CC BY 4.0 署名再分发，建议注明 snapshot/数据库版本，并直接引用 MP。
- [2026 官方受限数据文档](https://docs.materialsproject.org/materials-project-data-lakehouse/access-controlled-data)
  及[维护者说明](https://discuss.matsci.org/t/67394/4)指出 GNoME 等贡献数据不能笼统套用普通许可。
  只通过旧 Terms 或 API key 获取成功，均不足以认证一整包的逐条许可。
- 本地两个 summary JSONL 覆盖当前全部 23,306 个 ID，但当时只保存结构/物性，没有 `builder_meta.license`。
  没有从许可字段缺失推断 `BY-C`；改用官方 S3 summary 快照核对。
- 固定源为 `s3://materialsproject-build/collections/summary/`，Delta version=1、分区 `version=2026-04-13`。
  该分区一个 Parquet、280,159 行、6 个 row groups；仅读取 ID 和三个 builder_meta 子字段，
  对应压缩列数据 1,039,771 bytes（另有文件元数据/请求开销），未下载全量结构或谱。
- 与当前三个 split ID 关联：23,306 个全部命中且唯一，`BY-C=23,306`、`BY-NC=0`、未知/缺失=0，
  `gnome_r2scan_statics` batch=0。因此本包 MP 材料的 summary 许可已核实，普通数据署名再分发有明确依据。
  summary batch 是许可/来源元信息，不用于重新判断 DOS 的泛函或几何一致性。
- CC BY 4.0 法律正文已读：保留来源/署名及所供版权、许可和免责信息，注明修改，不暗示背书、不额外限制既有权利。
  历史一般条款另含隐私安全、计算精度/责任、禁止网站抓取与大批量 API 获取联系规定。
  不将后者解释为每次离线发布必须另获批准，也未发送任何消息。

逐材料记录、快照 URI、列投影、许可计数、对象 ETag 和记录文件 SHA-256 保存为独立审计附件。
这些证据核清本数据版本的数据许可适用范围，不等于直接读到了现行网站所有附加服务条款。

## 元素表来源链与数值事实

当前 `data/reference/periodic_table_v2.csv`（118 行、29 列，16,264 bytes）与
ARPAT Git `b4b6756ef463b57b862ab39dab62c29643a6ca12`（2025-07-16）原文件逐字节一致，
SHA-256 为 `8cfba286b85f175c73431011a5c24bb8b110223ed60e81bae0357a02af9318fc`。
随后经 uniARPAT `19c21ed` 引入再移动到当前目录，原数据未修改。ARPAT 原仓库未找到独立许可证。

与官方 PubChem CSV 逐格比较，前 17 列共 2,006 格，1,996 格数值/字符串一致，15 列全 118 行相同。
10 格差异：AtomicMass 的 Gd/Lu 两格；Electronegativity 的 He/Ne/Ar/Pm/Eu/Tb/Yb/Rn 八格。
这比“字段名称相似”更强，但仍不证明原下载日期、转手路径或差异的编辑者。

后 12 列 1,416 格均可独立复算且无差异：周期/族由原子序数和表内既定 f 区 group3 约定得到；
四轨道计数由显式电子排布的最高壳层获得；未填满数按 2/6/10/14 容量计算，空轨道记 0；两个总数分别求和。
这是数值构造认证，没有把新写的复算代码冒称当年的生成脚本。

用户记忆指向 JARVIS/PhononDB，按该线索检查了官方 JARVIS Elements/Magpie 数据和 PhononDB 公开索引：
JARVIS Magpie 存在大量相同字段/数值，但 Lr/Ds/Rg 配置等不同，不能证明整表直接复制；
PhononDB 所检查的公开索引没有提供同名表。材料谱来源不自动证明额外元素表的授权。

## 元素表许可的剩余边界

[NLM](https://www.nlm.nih.gov/web_policies.html#copyright)和
[NCBI](https://www.ncbi.nlm.nih.gov/home/about/policies/)明确区分政府公共领域作品与第三方内容。
PubChem 元素条目本身也列出 IUPAC、LANL、NIST、PubChem Elements 等不同来源及许可链接。
因此没有把 PubChem 数值匹配推成整表 CC0/CC BY 4.0，也没有套用 PhononDB 的 CC BY 4.0。
当前待补的具体凭据为原始获取/生成记录、10 个基础差异单元格的来源和对整个加工表的授权范围。
不据此认定表一定有侵权，也不宣称整个数据包已完成统一开源许可。

## 文件与验证

- 更新 `docs/dataset-sources.md`、两份数据卡及首次准备日志的后续证据入口。
- 独立新增本地附件 `mp-license-records.csv.gz`、`mp-license-audit.json`、`element-table-audit.json`，
  不改缓存格式、45 个数据载荷文件及原 tar 包；更新发布清单和校验文件。
- 核对逐记录许可集合与当前 ID 严格相等；全量 118 元素基础值比较及 12 列公式复算通过，差异明确记录。
  收尾复核文档及发布副本文档内部链接通过；11 个附件 SHA256SUMS 通过；原包及全部 45 个载荷文件指纹不变。
  不重复训练或模型测试。

当前结论：**MP 数据许可已核清到当前版本逐记录；旧表直接来源及构造已查明，整体再分发授权仍未完全确认。**
