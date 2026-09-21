# 设计：E5a 训练 checkpoint 代码边界

> 对应 `docs/status.md` 当前工程关卡。E5 分阶段进行；本阶段只提取 ablation runner 的 checkpoint 责任，不改变训练、评估或模型数学。

## 目标与成功判据

- **目标：**使 checkpoint payload 组装、AMP 设置一致性检查、恢复和原子写入各有唯一实现，避免 runner 在加入新运行时状态时重复或遗漏恢复逻辑。
- **范围：**提取 `run_ablation_experiments.py` 中现有的 `_ckpt`、`_atomic_save` 和 latest-checkpoint 恢复片段到 `utils/ablation_checkpoint.py`。
- **行为合同：**
  1. FP32 payload 键、模型和 optimizer 状态与既有 runner 相同；AMP payload 仍只额外包含 `amp_scaler`。
  2. 旧 checkpoint 缺失 `use_amp` 时按 `False` 解释；不同 AMP 设置恢复仍拒绝；optimizer 状态不可恢复时仍保持现有的宽容回退。
  3. 写入继续采用同目录临时文件后 `os.replace` 的原子替换。
  4. 关闭 AMP 的模型输出、loss、更新与结果文件不改变；不运行训练或测试集评估。

## 改动

- 新增纯运行时模块，不依赖项目模型类：输入 transformer、optimizer、可选 scaler 与标量元数据，返回／恢复标准字典。
- runner 只保留训练控制流、最佳分数与 history；不再拥有 checkpoint 格式细节。

## 测试关卡

- 小型 `nn.Linear + AdamW` 合同：FP32/AMP payload 的键和状态、旧 checkpoint 兼容、AMP 不匹配拒绝、scaler 恢复与原子覆盖。
- 全套既有测试和 `git diff --check`。

## 成本与风险

- 本阶段不移动损失、数据预处理或评估；这些是后续 E5 阶段候选，必须各自另立设计。
- checkpoint 是恢复边界；任何字段变更都必须先由单测固定，再由 runner 使用。
