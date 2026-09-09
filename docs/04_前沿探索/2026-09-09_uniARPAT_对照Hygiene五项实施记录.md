# uniARPAT 对照 Hygiene 五项实施记录（Week-3 前置）

**日期**：2026-09-09
**目标**：不动架构、不改目标函数，只修实验hygiene，使 M1–M5 重跑结果可复现、可归因。
**验证**：15/15 单测绿；M1 单轮冒烟通过（71.079M 参数，111.4s/epoch，收敛行为与旧首轮同量级）。

---

## H1 seed + set_epoch（`run_ablation_experiments.py`，`run_pilot_10epochs.py`）

- 新增 `setup_ablation_seed(seed)` / `setup_pilot_seed()`：random+numpy+torch(+cuda)，`cudnn.deterministic=True, benchmark=False`（消融可比性优先于吞吐）。
- 训练循环内 `sampler.set_epoch(epoch)`（旧代码从未调用 → 100 轮吃完全相同的 batch 序列）。
- `--seed`（默认 42）CLI 参数；seed 写入 history CSV 列 + 日志头 + pilot records。

## H2 评估唯一化（`utils/metrics.py::per_sample_spectral_metrics`）

- 两处独立实现的逐样本 MAE/MSE/R²（`model.test_one_step` 内函数、`evaluate_split`）收敛到同一函数，数学逐位一致（同为 `1 - ss_res/(ss_tot+1e-8)`），零行为变更，只杀分叉风险。

## H3 死参删除 + RP 缓存上提（`model/transformer.py`）

- 删除 encoder 每层未使用的 `self_attn` + `rbf_encoder/rel_proj/dir_proj`：**-6,556,648 参数（M4 75.85M → 69.30M；M1 77.63M → 71.08M，实测）**。
- `RPEncoding`（无参）上提至 `TransformerEncoder` 单例，每 batch 计算一次，经 `rp_base` 分发各层（各层保留可学习的 `rp_proj`）；单轮时间 117s → 111s。
- 兼容性：旧 checkpoint 含已删 key，`strict=True` 加载会报错——消融从零训练不受影响；`cif2dos.py` 用 `strict=False`，不受影响。
- `tests/test_p0_fixes.py` 同步升级：断言改为"死模块不存在"，层调用改传 `rp_base`。

## H4 scheduler 统一（`utils/builder.py::build_warmup_cosine_scheduler`）

- 新建 warmup（5 epoch，1e-5→base）+ cosine（floor 1e-6）组合，与 `configs/config.yaml` 语义一致；ablation 与 pilot 共同调用，替代原来的裸 cosine（无 warmup）。pilot LR 曲线至此可外推全量训练。

## H5 median + fail 口径

- 无代码变更（`evaluate_split`/history/test CSV 本就含 median 与 fail_rate）：选型维持 `balanced_score`（MAE median 双轨）以保证与旧基线可比；报告与决策一律用 median R² + fail rate，mean R² 停用。记录在此即为口径冻结。

## 附带修复（冒烟中新发现的 P0）：晶格断言误杀

- `utils/relative_features.py` 旧断言 `a,b,c ∈ (0.5,60)` 会把训练集 16 个 + 验证 3 个 + 测试 1 个**合法长轴样本**（c=65~110Å，层状/链状）判死刑，当前代码根本训不起来（Week-2 的成绩跑在断言加入之前）。
- 改为 `(0.1,1000)`：仍可捕获 1/c 契约违反类 bug（忘倒数→c~0.01；误用c作inv_c→c~0.14），合法长轴放行；报错信息补全上下界。

## 重跑基线须知

- 参数量变化（-6.56M）与 scheduler 变化（+warmup）意味着 hygiene 重跑 ≠ 旧成绩复刻——这是**更干净的对照**，旧 `history_m*.csv` / `test_m*_summary.csv` 归档为 v0-legacy，下轮输出加 `h1` 后缀区分。
