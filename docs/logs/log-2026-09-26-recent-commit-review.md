# 日志：2026-09-26 — 近期提交与研究方向审阅

## 范围

- **任务：**回应“现在该做什么、最近提交是否走弯路”，审阅最近 18 个提交
  （`e62a3e9` 至 `09ae78b`，9 月 18–25 日），重点核对最新 eDOS 改动、实验结论与默认结构信息通路。
- **既有工作：**开始时 `docs/status.md` 已修改；9 月 26 日的 train–valid、谱形支持两份日志和两份
  谱形支持 CSV 尚未提交。它们按工作区证据审阅，不能算入 `09ae78b` 的提交成果。
- **本轮写入：**新增本日志，在 `status.md` 标注待讨论建议和恢复保护问题。未修改生产源码、既有
  结果或检查点；没有实际训练或新增 test 评估。

## 证据与发现

### 1. 训练恢复保护未贯穿调用链，应在下一次训练前修复

- **调用链：**`run_ablation_experiments.train_and_eval` →
  `utils.ablation_checkpoint.restore_ablation_checkpoint` → runner 的异常处理与训练循环。
- **代码事实：**恢复模块会因 `use_amp` 或 `edos_slope_ratio` 不一致抛出异常，但 runner 捕获所有
  `Exception`，记录 `starting fresh` 并继续使用同一实验目录；`config_used.yaml` 在恢复检查之前
  已被写入。继续执行完整 epoch 后的保存路径也仍指向原目录。
- **实际复现：**在临时目录中用微型 Linear 模型、模拟数据加载器和首步即停止的训练替身，分别制造
  AMP、slope 配置不匹配。两种情况均记录恢复失败、覆盖临时原配置，并到达训练循环；探针在任何
  参数更新前停止，临时 checkpoint 字节保持不变，未接触生产产物。完整复现命令见下文。
- **影响：**当前保护不能阻止配置不匹配后覆盖原实验。`cfdb5ac` 增加 AMP 一致性检查，`0f4527c`
  提取恢复边界，`09ae78b` 增加 slope 元数据检查，都仍受这一外层行为影响。没有证据说明已记录的
  新 tag pilot 因此受到污染；这是后续运行风险。
- **建议修复验收：**不兼容恢复立即退出，旧配置、history、checkpoint 均不变，训练步调用数为零；
  兼容恢复仍保留原行为。需要 runner 层的回归验证，仅测恢复函数抛异常不足以覆盖此问题。

### 2. 默认 B7 存在可定位的单元素结构盲点，整体影响尚未确定

- **调用链：**`Transformer.forward` 由元素编号构造 `atom_src` → `TransformerEncoderLayer.forward`
  将几何只加入注意力分数 → 注意力对原子 Value 加权 → decoder/输出头。
- **代码推导：**默认 B7 的 `q,k,v` 均取 `src`，没有几何 Value 内容。eval 模式中，同元素原子的
  起始特征相同；对相同 Value 做和为 1 的加权不会引入几何差异，后续逐原子前馈层也不能打破这个
  对称性。这是已有 B7 的表达边界，不能归咎于最新 slope-loss 提交。
- **实际核验：**CPU FP32、冻结 B7 epoch 33、两个 Si 原子、固定原子数，改变晶格与分数坐标，
  eDOS/phDOS 概率谱 TV 分别为 `8.66e-8/1.57e-7`，eta/gamma 最大差 `1.19e-7`。
  使用同样几何变化的 Si/O 对照，eDOS/phDOS TV 为 `0.02645/0.03786`。这是合成输入的结构敏感性
  核验，没有真实谱标签，不能解读成准确率实验。
- **现有结果复核：**对 9 月 26 日 CSV 按 Q1 元素缓存重新分组，并核对划分内索引和材料 ID：

| 划分／分组 | 样本数 | eDOS oracle 中位 R² | 失败数／率 |
|---|---:|---:|---:|
| train 单元素 | 368 | 0.585491 | 23／6.25% |
| train 多元素 | 18,338 | 0.605910 | 129／0.70% |
| valid 单元素 | 37 | 0.467225 | 7／18.92% |
| valid 多元素 | 2,276 | 0.515192 | 138／6.06% |

  上表为冻结 B7 ep33、Q1、全 bin oracle eDOS；只重汇总既有结果，没有重跑全量前向。
  单元素仅占 valid 的 `37/2313=1.60%`，不能解释全体中位数问题。现有 320 个同组成材料对中，
  21 个单元素对的目标／预测 TV 中位数为 `0.27177/0`；299 个多元素对为 `0.26883/0.01738`。
  排除结构匹配对后，258 个多元素对仍为 `0.28229/0.01713`。后者是条件响应偏小的关联线索，
  仍不能排除标签差异、数据支持不足或读出收缩。
- **研究建议：**利用已有 `_g2ctl` 与 `_g2edge` 的相同 epoch 检查点，在 train/valid 上核对几何
  消息是否真正增加结构条件响应，并分别观察 encoder 表征与谱输出。先写明探针、尺度归一化和
  干预对照；若表征有差而输出无差，支持进一步调查读出；若表征仍缺少差异，支持调查输入／encoder。
  这些分支用于缩小假设，不等同于因果证明。不直接重跑已 park 的 G2 或启用新架构。

### 3. 有用的负结果与结论过度外推需要分开

- **合理部分：**R1、G2、R2b、slope 等负结果后保持默认关闭，B7 基准没有被无收益方案替换；AMP、
  checkpoint 边界、CI 与 CIF 入口有独立工程价值。slope pilot 有预先写下的干预、valid 成对判定和
  停损线，高粗糙度组 Δmedian R²=`−0.00167`、机制指标未支持，停止合理。
- **方法问题：**C5、R1、E10、G2、R2 等历史 pilot 在 test 上判断是否晋级；G2 日志明确将 test
  中位 R²作为晋级条件。这些 test 结果已参与研发决策，后续不能再称同一 test 为从未使用的独立
  最终验收集。最新 slope pilot 的 `--skip_test_eval` 与 valid-only 判定纠正了后续使用方式，
  但不能消除历史使用；未来最终泛化确认需要另行讨论评估设计，本轮不改划分。
- **归因过强：**`decisions.md` 对 R1b 使用“证明……query 流形严重退化”的表述，而实现同时改了
  query 生成方式并将可学习 target 置零。现有对照支持否定该组合，无法单独确定 plain MLP 的
  表达能力就是根因；短 pilot 的平局线也不等同于统计等效性检验。建议以后收窄文字，保持 park。
- **目标偏移：**slope pilot 以高粗糙度 205 条为主验收、整体仅为保护项，不能代表全体 eDOS
  准确率目标。9 月 26 日工作区状态已将整体中位 R²／失败率放回主目标，应延续这一修正。

### 4. 近期证据保存和 CI 有小缺口

- 9 月 25 日梯度归因、9 月 26 日谱形支持诊断使用一次性内存脚本，未保存可直接重跑的评估入口。
  CSV 可重汇总，但产生梯度、最近邻和结构匹配结果的完整执行无法直接复现；后续以这些结果决定
  实验前，应将需要复用的计算纳入 `tools/eval/`，复用现有加载与评估代码。
- 梯度归因结果记录的是**组均值梯度的范数** `||mean(g_i)||`，不等同于每样本梯度范数的均值
  `mean(||g_i||)`。日志的“每样本梯度范数”用词容易扩大含义，不能据此证明训练全程不受样本权重影响。
- `tools/ci/check-static.sh` 的测试列表尚未包含最新三个 `test_edos_*` 模块。本轮手动运行它们及
  checkpoint 测试共 **27/27 通过**；这不覆盖上面的 runner 异常处理漏洞。

## 结论与下一步

- **状态：closed（审阅完成，问题尚未修复）。**近期有范围受控的有效探索，也出现了“不断试局部
  模块、再增加相关性诊断，却未缩小结构信息瓶颈”的绕路风险。负结果本身不应被当成无价值工作。
- **建议顺序，待讨论：**先完成恢复失败保护这一小修复；研究上以现有 G2 两臂做一次有明确分支
  结论的结构信息通路核验，再决定唯一的单因素 pilot。总体 eDOS 中位 R²和失败率为主验收，
  blind 与 phDOS 为保护项；不因单元素反例直接启动面向全部材料的改模。
- **建议规则，未写入持久决策：**pilot 显式跳过 test；影响后续决策的诊断保留可重跑入口；
  每项诊断先写明结果将排除哪条假设、触发什么下一步。由马尚酱决定是否固化。
- **记录：**保留开始时已有的工作区变更；`status.md` 仅追加待讨论建议和恢复保护关注点，
  不改 `decisions.md`。本轮无生产符号新增、删除或修改。

## 验证与复现

### 已执行的相关测试

```bash
python3 -m unittest tests.test_edos_slope_loss tests.test_edos_slope_pilot_verdict tests.test_edos_error_attribution tests.test_e5_checkpoint_boundary
```

结果：27 项通过。没有运行完整数据回归或训练；本轮不修改生产代码。

### 冻结 B7 的小型结构输入核验

以下命令在仓库根目录运行，只读取冻结 checkpoint 和 Z0 元素表：

```bash
python3 -B - <<'PY'
import torch
from pymatgen.core import Lattice, Structure
from utils.b7_cif_inference import load_b7_model, structure_to_b7_inputs

torch.set_num_threads(2)
model, metadata = load_b7_model(
    'output/ablation_m1_e9ctl/checkpoint_best.pth', torch.device('cpu'))
structures = []
for species in (['Si', 'Si'], ['Si', 'O']):
    structures.extend([
        Structure(Lattice.cubic(5.4), species, [[0, 0, 0], [.25, .25, .25]]),
        Structure(Lattice.from_parameters(4.1, 6.2, 7.3, 75, 82, 96),
                  species, [[0, 0, 0], [.4, .2, .1]]),
    ])
inputs = [structure_to_b7_inputs(s) for s in structures]
src = torch.stack([x[0] for x in inputs])
pos = torch.stack([x[1] for x in inputs])
with torch.inference_mode():
    outputs = model(src, src.eq(0), pos)
    for name, a, b in [('Si/Si', 0, 1), ('Si/O', 2, 3)]:
        for key in ('edos', 'phdos'):
            p = outputs[key].softmax(-1)
            print(name, key, float(.5 * (p[a] - p[b]).abs().sum()))
        print(name, 'eta/gamma',
              float((outputs['eta'][a] - outputs['eta'][b]).abs().max()))
PY
```

### 恢复不匹配的 runner 调用链核验

以下复现仅在自动清理的临时目录写微型测试文件；训练替身在首步更新之前抛出停止信号：

```bash
python3 -B - <<'PY'
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
import run_ablation_experiments as runner
from utils.experiment_config import ExperimentConfig

class ReachedTrainingLoop(Exception):
    pass

def stop_before_training(*args, **kwargs):
    raise ReachedTrainingLoop()

repo = Path.cwd()
default_yaml = (repo / 'configs/default.yaml').read_text()
for mismatch in ('edos_slope_ratio', 'use_amp'):
    with tempfile.TemporaryDirectory(prefix='uniarpat-review-') as directory:
        root = Path(directory)
        (root / 'configs').mkdir()
        (root / 'configs/default.yaml').write_text(default_yaml)
        run_dir = root / 'output/ablation_m1_review'
        run_dir.mkdir(parents=True)
        original = 'original_config_marker: keep_me\n'
        (run_dir / 'config_used.yaml').write_text(original)
        transformer = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=5e-5)
        checkpoint = dict(model=transformer.state_dict(),
                          optimizer=optimizer.state_dict(), epoch=5,
                          best_val_score=1., use_amp=mismatch == 'use_amp')
        if mismatch == 'edos_slope_ratio':
            checkpoint.update(edos_slope_ratio=.1, edos_slope_lambda=1.)
        path = run_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, path)
        before = path.read_bytes()
        model = SimpleNamespace(model={'transformer': transformer},
                                optimizer={'transformer': optimizer},
                                gscaler=None, edos_slope_ratio=0.,
                                train_one_step=stop_before_training)
        model.to = lambda device: model
        builder = Mock()
        builder.get_model.return_value = model
        builder.get_dataloader.return_value = ['synthetic_batch']
        entered = False
        try:
            os.chdir(root)
            with patch.object(runner, 'ConfigBuilder', return_value=builder), \
                 patch.object(torch.cuda, 'is_available', return_value=False):
                try:
                    runner.train_and_eval(ExperimentConfig(
                        epochs=1, tag='_review', skip_test_eval=True))
                except ReachedTrainingLoop:
                    entered = True
        finally:
            os.chdir(repo)
        print(mismatch, 'training_loop_entered=', entered,
              'config_overwritten=',
              (run_dir / 'config_used.yaml').read_text() != original,
              'checkpoint_unchanged=', path.read_bytes() == before)
PY
```

两种不匹配的实际结果均为 `training_loop_entered=True`、`config_overwritten=True`、
`checkpoint_unchanged=True`。最后一项表示探针及时停止，不表示真实继续训练能保护旧产物。
