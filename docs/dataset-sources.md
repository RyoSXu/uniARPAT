# 数据来源、引用与许可 / Sources, attribution and licensing

核查日期 / Reviewed: **2026-10-19**。配套说明：[中文](dataset-card-zh.md) · [English](dataset-card-en.md)。

## 发布状态 / Release status

本说明区分已确认来源、许可证据和未补齐凭据，不对整包额外授予统一许可证。
MP 本版本的数据许可已补核：23,306 个材料在官方 2026-04-13 summary 快照中均为 `BY-C`，无缺失、无 `BY-NC`。
发布范围已收回当前主线所需的冻结数据和 8 个配套参考文件；旧周期表、mendeleev 特征快照及旧表审计不纳入本次附件。
这些历史表的来源问题不再作为本包的许可待办。尚未上传、创建远端标签或发布。

This notice distinguishes provenance, verified licensing evidence, and missing evidence. It does not grant a new uniform
license over the archive. All 23,306 materials have `BY-C` metadata in MP's official 2026-04-13 summary snapshot,
with no missing IDs or `BY-NC` entries. Release scope is limited to the frozen current data and 8 companion reference files.
The legacy periodic table, mendeleev feature snapshots, and element-table audit are excluded; their historical provenance
questions are outside this package. Nothing has been uploaded or published.

| 组成 / Component | 当前证据 / Evidence | 边界与待办 / Boundary and remaining evidence |
|---|---|---|
| MP 结构及全部 eDOS、18,644 条 phDOS / MP structures, all eDOS, 18,644 phDOS | 官方快照逐材料 `BY-C` 核查通过；[MP 维护者确认可再分发快照](https://discuss.matsci.org/t/45437/2)，适用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) / Per-material `BY-C` metadata and maintainer confirmation support attributed redistribution | 保留 MP 署名、版本、许可及修改说明；不能推广到全部 MP 数据。完整现行网页仍返回 403，正文获取边界见下 / Retain attribution, version, license and changes; do not generalize to every MP product. Full live webpage remains inaccessible |
| PhononDB，2,390 条 phDOS / PhononDB, 2,390 phDOS | 数据 DOI [10.48505/nims.4197](https://doi.org/10.48505/nims.4197) 的 [DataCite 注册元数据](https://api.datacite.org/dois/10.48505/nims.4197)明确为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) / Registered metadata explicitly identifies CC BY 4.0 | 保留 Atsushi Togo / NIMS 署名、许可链接及本地复算/换算说明；保留逐条 MDR 标识 / Retain attribution, license link, recomputation/conversion notice, and record-level MDR IDs |
| JARVIS，2,272 条 phDOS / JARVIS, 2,272 phDOS | [NIST 数据许可政策](https://www.nist.gov/open/license)区分 non-SRD 数据、SRD 与第三方作品，并规定署名和修改说明 / NIST policy distinguishes non-SRD, SRD, and third-party works | 不能把软件许可证直接当数据许可证，或未经依据改写为统一 CC0；保留 NIST 来源与处理说明 / Do not substitute the software license for data terms or relabel everything CC0 without evidence; retain NIST attribution and modification notes |
| 价电子表、划分、网格、审计 / Valence tables, splits, grids and audits | 项目处理产生，来源和生成规则见数据卡、审计及 Git / Project-derived, described by cards, audits, and Git | 不冒用第三方权利，也不代替作者决定自有内容的新许可证；正式发布时确定项目贡献的许可 / Do not relicense third-party rights or invent an author grant; specify the project's own contribution license when publishing |

MP 逐材料数据许可已核实，不再笼统标作“MP 许可未知”；也不声称已直接读取现行条款网页的每一项附加条款。
本次只缩小发布范围，不对保留的第三方内容或项目贡献新增统一许可。

MP record licensing is verified and is no longer described simply as unknown. This does not claim direct inspection of every current website-specific provision.
Narrowing release scope does not grant a new uniform license over retained third-party material or project contributions.

## MP 一般条款与本包适用范围 / MP general terms and applicability

取得完整一般条款的历史正文及维护者解释：[2022 年快照再分发答复](https://discuss.matsci.org/t/45437)，
[2024 年加工后公开发布答复](https://discuss.matsci.org/t/56836)。2026 年的
[官方受限数据说明](https://docs.materialsproject.org/materials-project-data-lakehouse/access-controlled-data)
及[维护者对 BY-NC 的说明](https://discuss.matsci.org/t/67394/4)用于核对现有适用边界。
`mp-license-audit.json` 和 `mp-license-records.csv.gz` 保存本版本的逐条核查。

Historical general terms and maintainer interpretations are available in the 2022 snapshot and 2024 redistribution replies.
Current access-control documentation and the 2026 maintainer explanation establish the restricted-data distinction.
The two license-audit companion files preserve this version's per-record check.

- **数据许可。** 一般 MP 内容按 CC BY 4.0 可复制、分发和加工；允许商业用途。须保留所提供的署名、版权/许可与免责信息、来源链接，并注明修改。
  不得暗示 MP 背书，或添加限制接收者原有许可权利的条款。完整许可法律正文见 [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)。
  / General MP content permits copying, redistribution, adaptation, and commercial use under CC BY 4.0. Retain supplied attribution,
  notices and links, identify changes, and avoid implied endorsement or downstream restrictions inconsistent with the license.
- **获取与责任。** 恢复的一般条款要求遵守 LBNL/MP 隐私安全规定，说明数据来自计算、精度不作保证，禁止直接抓取网站，并提供 API 获取途径。
  旧条款对大批量 API 下载提出提前联系的要求；它不是“每次分发都须额外批准”的依据。本轮未联系任何第三方。
  / Recovered general terms cover privacy/security, computed-data limitations, website scraping restrictions, and API acquisition.
  Their bulk-API contact provision is not evidence of an extra approval requirement for every redistribution. No third party was contacted.
- **本包核查。** 从官方 `collections/summary/` 的 `version=2026-04-13` 分区、Delta 版本 1，
  仅投影材料 ID 及 `builder_meta.license/batch_id/database_version`。23,306 个 ID 全部命中 `BY-C`，
  `BY-NC` 和 `gnome_r2scan_statics` 均为 0。投影列压缩数据约 1.04 MB，未下载全量结构或谱。
  / The official snapshot's four projected metadata fields match every current ID: 23,306 `BY-C`, zero `BY-NC`, and zero GNoME batch matches.
  This was a metadata query, not a bulk structure/spectrum download.
- **结论边界。** 这些是该 snapshot 的 summary 许可字段，不是 DOS 计算条件；summary 的 r2SCAN 等 batch 标签不改变已审计 eDOS 的泛函分类。
  现行完整 Terms 页面仍为 403，因此没有把历史正文冒充实时网页的逐字副本，也没有用未获答复的论坛提问代替官方规定。
  / Summary licensing metadata does not reclassify DOS calculation settings. The live full Terms page still returns 403;
  historical text is not presented as its verbatim current copy, and unanswered questions are not treated as policy.

## 引用与修改说明 / Citations and modification notice

建议同时注明本数据版本、manifest SHA-256、代码提交/正式 Release URL，并按实际内容引用下列原来源。
当前没有 uniARPAT 数据集 DOI 或独立数据论文，不提供虚构的作者/DOI。

Cite this data version, manifest SHA-256, code commit/permanent release URL, and the applicable original sources below.
There is currently no independent uniARPAT dataset DOI or dataset paper; no author list or DOI is fabricated.

- **Materials Project**：Jain et al. (2013), *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation*,
  APL Materials 1, 011002, [doi:10.1063/1.4812323](https://doi.org/10.1063/1.4812323)。
  同时引用实际使用的 MP 数据记录/任务及访问日期 / Also identify the MP records/tasks and access date used.
- **PhononDB**：Togo, Atsushi (2023), *MDR phonon calculation database*, National Institute for Materials Science,
  [doi:10.48505/nims.4197](https://doi.org/10.48505/nims.4197)。
  原作者的 [MDR 记录索引](https://github.com/atztogo/phonondb/tree/main/mdr/phonondb)提供原条目链接 / The author's index links individual records.
- **JARVIS**：Choudhary et al. (2020), *The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design*,
  npj Computational Materials 6, 173, [doi:10.1038/s41524-020-00440-1](https://doi.org/10.1038/s41524-020-00440-1)。
  逐材料保留 JVASP ID / Preserve per-material JVASP IDs.

uniARPAT 的处理包括 ID 映射与谱配对、PhononDB DOS 复算、轴与密度单位转换、分箱、尖峰截断、Q1 筛选、
固定划分、结构补全及价电子修复。当前包保留处理后的 FP32 数值；不是原来源未经修改的镜像。
请同时保留来源名称、链接、适用许可、上述修改说明及日期；第三方原权利不由本项目取代。

uniARPAT processing includes ID mapping and pairing, PhononDB DOS recomputation, axis/density conversion, binning,
spike clipping, Q1 filtering, frozen splits, complete-structure recovery, and valence-count repairs.
The package preserves processed FP32 values; it is not an unmodified upstream mirror. Retain source names, links,
applicable terms, these modification descriptions, and their dates. Original third-party rights are not replaced by this project.

## 证据边界 / Evidence boundaries

- 当前逐行来源来自外部冻结 `v2_processed.parquet`，由导出工具先对当前 33 个数组中的谱/mask、ID、划分和原子数作一致性检查。
  CSV 是补充来源附件，不改变训练缓存格式。父表 SHA-256 和来源 CSV 哈希记录在 `provenance-audit.json`。
  / The exporter validates spectra/masks, IDs, splits, and complete atom counts before attaching historical provenance.
  The CSV is a companion asset, not a cache format change; hashes are recorded in `provenance-audit.json`.
- 源表 `ph_ref=mp:pheasy/dfpt` 没有唯一声子 task ID；没有把缺失字段推断成已知任务。
  / MP phonon references identify method categories rather than unique task IDs; missing task identity is not inferred.
- 网页获取失败不等于数据不可用。PhononDB 许可基于实际读取的 DOI 元数据；MP 依据官方一般许可、维护者解释与逐记录字段，
  而完整现行网站条款全文仍有获取边界。
  / Fetch failures are not licensing conclusions. MP record metadata and maintainer statements support this version's data license;
  direct inspection of the entire live Terms page remains incomplete.
- 说明方式参考 Matbench、MPtrj 与 Hugging Face Dataset Cards；没有使用它们的谱或训练样本。
  / Matbench, MPtrj, and Hugging Face Dataset Cards informed documentation organization, not dataset membership.
