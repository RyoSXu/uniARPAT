# 工作日志 2026-09-10：hygiene+工程实施与M1 h1基线

## Hygiene H1–H5 + E1–E4（提交 `3e6df8f` `5b3d280`）
- seed42+set_epoch（ablation/pilot）；评估函数唯一化（`utils/metrics.py`）；死参-6.56M（M1 77.63M→71.08M）+RP上提（117s→111s/轮）；warmup+cosine统一；median+fail口径冻结。
- 附带修：晶格断言误杀（16训练+3验证+1测试合法长轴样本，(0.5,60)→(0.1,1000)）；config落盘；checkpoint全量dict；requirements补齐+lockfile。
- 验证：15/15单测；M1单轮冒烟通过。

## M1 h1基线（100轮，成绩归档 `*_h1.csv`）
- eDOS med 0.519/fail 12.3%（旧0.472/9.3%）；phDOS med 0.678/fail 10.0%（旧0.684/9.0%）；best ep43。
- 解读：eDOS涨（hygiene复合效应，待拆归因），phDOS持平。M2–M5排队中。

## 性能剖析
- 数据0ms/前向113ms（encoder占61%）/反向177ms；7GB/32GB。结论：重跑前不动性能代码；AMP为h1后首个加速臂。

## 文档体系
- 01短名化（Backlog/Decisions/DataSpec/Roadmap）+README索引+archive；04索引；Backlog可执行卡片→瘦身回清单+Design制。
