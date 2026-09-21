# 设计：B7 M1 CIF 盲推理入口

> 对应 `docs/status.md` 当前关卡。此项导出已冻结的 B7 `_e9ctl`；不训练、微调或修改 B7，
> 也不替换 `cif2dos.py` 的 M4 兼容职责。

## 目标与成功判据

- 从一个完整、有序的 CIF 和显式提供的 B7 epoch-33 checkpoint，得到只依赖 CIF 的 eDOS／phDOS
  预测、E0/P0 bin 中心及可审计元数据。
- 入口必须严格复现 B7 的 M1 结构：512 维、6 encoder＋6 shared decoder、`legacy3`、0.05 dropout
  （eval 模式）、`head_type=legacy` 和 H1 `scale_mode=eta`；checkpoint 必须是 M1、seed 42、epoch 33。
- 输出的尺度必须符合已经验证的 H1 blind 合同：

  ```text
  shape_e = softmax(e_logits)
  shape_p = softmax(p_logits)
  S_e = N_val(CIF) * gamma_hat / 0.09375
  S_p = 3 * N_atoms(CIF) * eta_hat / 19.6875
  eDOS = shape_e * S_e
  phDOS = shape_p * S_p
  ```

  `N_val(CIF)` 由版本控制的 `index/z0_zval.json` 逐原子相加，不从标签、训练缓存或全局常数取得。
- 成功时单 CIF 的输出数组为 `(128,)`／`(64,)`、全为有限非负值；输出 metadata 记录 checkpoint
  认证字段、CIF 原胞原子数、`N_val`、eta/gamma、总量、网格和代码合同版本。

## 改动

- 新增独立的 B7 入口与可导入核心模块，CLI 强制 `--checkpoint`、支持单 CIF 或目录，默认 CPU。
  不把随机初始化或 M4 checkpoint 当作可用回退。
- 使用 E0/P0 版本控制网格的 bin **中心**，保存机器可读 `npz` 与 JSON/汇总 CSV；可视化和热力学
  后处理不是本轮入口的职责，避免把旧 M4 的物性图和尺度语义混入 B7 结果。
- `cif2dos.py` 保持不动、仍只标记为 M4 兼容工具；两入口的特征布局均是 82 行格式，但 B7 对
  `>80` 原子、无序占位、超出 token／Z0 表的元素直接报错，绝不静默截断。

## 测试关卡

- 合成 Si CIF／`Structure` 测试 82-token 与 `1/c` 输入合同、E0/P0 中心和 Z0 价电子总数。
- 用确定性假模型测试 H1 公式、非负／有限性和 eDOS/phDOS 总量；测试错误 checkpoint 元数据、缺少
  `eta`、无序／超长 CIF 与未知价电子元素均被拒绝。
- 对本地 B7 checkpoint（若存在）做只读 CPU 烟测：严格加载、一次 CIF 前向和输出元数据；它不进入
  无 checkpoint 的 CI。
- 保持 E7 的缓存无关检查可执行，并运行完整本地套件；不运行训练。

## 成本与风险

- B7 checkpoint 约 854 MB，CPU 首次加载／前向会较慢；入口不下载权重、用户须明确给出路径。
- CIF 的原胞选择影响原子数与 eDOS 总量。入口采用 CIF 文件给出的单胞，metadata 明示该约定；
  它不擅自 primitive 标准化，也不为有序性／占位问题猜测结构。
- H1 只预测 E0/P0 窗口内的总量；输出应标为 **B7 blind prediction**，不能被解释为标签可得的
  oracle 结果，尤其 eDOS 的 p99 blind gap 已知较大。
