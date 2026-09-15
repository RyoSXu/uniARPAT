# data/ 目录说明

> 当前数据 = v2（24,988 双谱，E0+P2 网格，8:1:1 切分）。v1 已归档冻结。

| 路径 | 内容 | 状态 |
|---|---|---|
| `train4ARPAT/{train,valid,test}/` | 主训练集（20,040/2,477/2,471 + 掩膜 + manifest） | ✅ 当前 |
| `grids_c2b/{E1,E2,E3,P1,P2}/` | C2b 网格臂标签（E0/P0 沿用主集） | ✅ 实验用 |
| `grids_c2b/X_*/` | 配好的实验组合（全相对软链，不占地方） | ✅ |
| `grids_c2b/grids.json` | 七臂网格边定义（唯一真源） | ✅ |
| `archive/v1_train4ARPAT/` | v1 训练缓存（13,707，冻结） | 🔒 只读 |
| `archive/v1/` | v1 源 csv + csv2npy.py + test.ipynb | 🔒 只读 |

库外（`getdata/`，永不入库）：raw 档案、`v2_release/` 发布包（含 A9 描述符）。
裸 `train4ARPAT` 即 v2；历史文档里的 `train4ARPAT-v2` 指同一份（2026-09-15 转正）。
