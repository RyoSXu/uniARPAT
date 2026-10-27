# 日志：2026-10-25 — 固定谱形辅助候选准备

用户同意按[单候选设计](../design/design-2026-10-25-spectral-shape-supervision.md)推进。
代码、相关回归与V100短检完成，正式35轮尚未启动；本轮未提交Git。

## 已落地与冻结条件

新增显式`--spectral_shape_aux`，保持完整A64、新每原子总量、train常数与blind选点。
训练使用`0.5 mean(E_e+S_e)+0.5 mean(E_p+S_p)`，S为历史KL/W1/Huber，辅助权重固定1。
零谱S记0且仍接受E监督；不按valid调权、不改mask、不读取test作评价。
日志分别记录energy和shape分量；选点仅使用E。

主对照为已完成纯E ep32，旧A64 ep31为工程参考；只新增seed42、35轮、batch32、lr5e-5、FP32候选。
准备入口认证旧对照配置、完整history、best及导出，并检查protocol完全相同。
除辅助项和新protocol路径，training record无其他变化；两者checkpoint隔离。

## 验证与成本

- 119项相关测试通过，含7项新增：复用谱形核、单位不变性、零谱无虚假梯度、有限差分梯度、
  loss与blind评估关系、微批单次更新、开关接线/契约隔离/导出拒绝漂移。
  旧checkpoint、选点、局部残差及当前完整晶体输入路径另作回归；不宣称全仓验收。
- 当前model/utils/tools/tests的Ruff通过；收尾检查文档引用、空白及内存语法编译。
- V100-SXM2-32GB、FP32、batch32两次一次性更新2.502秒、0.793秒，
  峰值已分配7505.5 MiB（约7.33 GiB）。扣除辅助项后，训练loss与评估差7.87e−8。
- 候选初值元信息与纯E完全相同，映射后初始state SHA256：
  `924bd18a26ba7fe5d266713df9d463bdb6d1b6547787fce699944327e208eeb6`。未加载训练权重。
- 单批共享Encoder任务梯度范数约1.437/0.509、余弦−0.080，只是短检诊断，
  不据此认定负迁移、梯度平衡、精度或泛化收益。

短检仅两次一次性内存更新和额外lr=0检查，不保存实验checkpoint。
正式35轮沿用约4.43小时成本量级，首轮更新；短检峰值不能代替全程显存。

## 产物与交接

准备：`results/runs/spectral_shape_aux_20261025/report.json`，protocol在同目录。
短检：`results/runs/spectral_shape_aux_preflight_20261025/report.json`，passed=true，绑定52份生产源码。
训练目标：`results/runs/ablation_m1_sbal_shape20261025_shape_a64_s42_e35/`。
准备入口只打印唯一训练命令，报告中的command列表是正式执行入口。

长训练交OpenCode。运行前核对新短检源码与输入身份，训练后比较纯E和旧A64，
报告主指标、失败率、谱形和同组成诊断；不自动换参考、调权、加预算或提交。
旧纯E产物和源码快照保留，不要求已完成实验的源码哈希等于添加辅助实现后的当前源码。
执行提示词在对话中交付，本日志不保存提示词副本。
