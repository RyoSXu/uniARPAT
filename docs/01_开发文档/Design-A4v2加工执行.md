# Design-A4：v2加工执行

**状态**：待讨论（等A3构成表） ｜ **规范**：[DataSpec数据加工规范.md](./DataSpec数据加工规范.md)

## 要点
- 接口骨架：`getdata/v2/process.py`（待填CAP_ATOMS/GRID）→ Parquet全集 → `build_v2_cache.py` → npy缓存。
- 锁定项：CIF六项、cm⁻¹单视图、泛函配对、自旋规则、原胞symprec、mask永不填0。
- 验收：round-trip抽1000 diff=0；视图相关>0.99；离群归因表清零。
