# Design-C5：解码器MoE

**状态**：待讨论 ｜ **Backlog**：[待开发总清单Backlog.md](./待开发总清单Backlog.md)

## 待定事项
- token级（per能量bin）vs 晶体级（PhysMoE按金属性分诊）先上哪个；
- 专家数/宽度/替换层数（参数预算≤8M）；
- 负载均衡loss权重；
- 与门控注意力的承接关系（M4判死刑后替换）。
