# Design-E：Encoder v2

**状态**：已决断（P0冻结，轮数另定） ｜ **Backlog**：E9 ｜ **关联**：`Backlog.md / Decisions.md / DataSpec.md / Roadmap.md / docs/04_前沿探索/*EncoderV2*.md`
**环境冻结**：`glibc 2.17 / gcc 4.8.5 / 无nvcc,cmake,C++17,Triton,flash-attn / Python 3.12.2 / torch 2.2.1+cu121 / e3nn 0.5.6`
**输入合同**：仅CIF可导信息 ｜ **输出合同**：`H_atom [B,L,512] + mask_atom [B,L]`、`h_global [B,512]`

> 约定：调研定候选，实验定胜负。本文件只写已决断合同与禁令；Q/G/L具体参数与三件套实现进对照矩阵，轮数另定。

---

## 1. 网格与数据合同（已决断 2026-09-10）

* 新增版本化`GridManifest`，字段：`source / split / cell口径 / N_atom定义 / axis_unit / bin_edges / bin语义 / dos_unit / spin策略 / 负频策略 / area_target / provenance`。
* `[-6,6]/128`与`[-10,10]/128`记为两个manifest，不混称同一任务；在统一前不报“任意分辨率”。
* 配套回放：THz→cm⁻¹横轴强度转换保面积一致；原胞/常规胞变换`N_atom`/面积/曲线可回放。

## 2. 网格密度对照（已决断：理论缩范围，对照裁决）

* 理论只定下限：`bin宽 ≤ 典型线宽1/2~1/3`；总数不限2的幂；bin语义为盒平均（`trapz`均值/bin），不按中心点值。
* 最优由Backlog C2b裁决：Stage1 eDOS四臂（E0 anchor/E1窄密/E2非均匀/E3宽密，phDOS固定宽非均匀P1）→ Stage2 phDOS三臂（P0 anchor/P1宽非均匀/P2宽等距，eDOS用Stage1胜者）；判据`median R² + fail率 + 峰区MAE`；胜者永久采用。

## 3. 面积与守恒合同（已决断：先审计禁硬投影）

* 当前禁phDOS `3N`硬投影（只读审计`ratio=area/3N`中位~1.0但约11.5%>3，1/4/8倍级差异来源未定），禁eDOS有限窗口`N_val`硬约束（`NELECT`为全窗口全电子语义≠窗口面积，且`ZVAL`随伪势/磁性/价态变）。
* P0只输出并评估`shape（非负）/ area_pred（同边界求积）/ area_target（同窗口同语义真值求积）/ 相对+绝对误差`；审计稳定后才启用`shape/area(shape)*area_target`。
* E3NN-phDOS记为合同参考（全谱+口径统一前提），不搬权重；M1-M4真值min/max逆归一化只记oracle，blind能力以ScaleHead实测为准（当前M5已崩，不当能力）。

## 4. 负频与离群（已决断）

* 负频保留（虚频=稳定性信号），不静默删；热力学积分处理与谱目标分离。
* 离群按归因表：单bin spike按能量bin `p99.9 winsorize`；真f电子峰保留+训练降权；不可考记缺失，永不填0；尾部用loss加权+median评估。

## 5. 评估与路线（已决断 2026-09-10）

* 双主判：eDOS/phDOS各看`median R² + fail率`，一票否决；`total_MAE`只作参考，不当判据。
* 四格必报：`valid开卷 / valid闭卷 / test开卷 / test闭卷`。开卷指`run_ablation_experiments.py:evaluate_split`用真值min/max逆归一化（训练未喂min/max，`model.py:153`输入仍仅`inp,mask,pos`；`dataset.py:47-49`归一化+评估借统计量）；闭卷指仅CIF+自预测scale。判赢只看闭卷：闭卷相对开卷掉`<0.1`可用，`0.1-0.2`预警，`>0.2`判死（起步线，跑两轮再调）。
* 无外部权重：端到端自研，不装MACE/Orb/UMA权重，不做teacher蒸馏。Design-P0三选一中仅保留(a)自监督遮罩+坐标去噪（15万MP零标签自有）/(b)标量监督，(c)冻结uMLIP删除；预训练放E9-P0之后（地基错了预训练越久偏越远）。Backlog D1 uMLIP伪标签同步暂停（撞本条，见Backlog）。
* 开工门槛：A6切分 + B1 h1 M1-M5 + B2 v2基线完成；h1前不动架构（Decisions#9）。
* 分期（串行晋级，严禁并行）：P0必做=合同§1-4 → Q组§9（Q1→Q2，输停，不进G） → G1§6（精度+显存+五项单测，挂则G2/G3免谈） → L组§10（固定Q+G胜者，L0→L1）；P1选做=Q3/G2/G3/L2，一臂一臂加，赢留输删。P0闭卷达标则P1可缓，仍崩则P1必做。
* 与C线分工（互不重复）：24维特征+旋转/声子增强归C1先做；C2截断mask即G1、C2 SumNorm-KL即L2（等合同修完），E9不另起；C2b网格见§2；C3/C5 MoE在E9-P1之后；D3单标签随Phase1.5后。
* 小模型约束：v2目标12-20M（现状二三十M），11k双谱养不起大等变；先进指小数据鲁棒（闭卷不崩+换网格可用），不指参数大名词多。

---

## 6. 几何G1精确稀疏图（已决断 2026-09-10）

* token不变：`L`槽位+`mask_atom`不动，加1全局token管长程；变的是`邻接mask[B,L,L] bool + 距离/方向[B,L,L]`，`5.5A`外score置`-1e9`，权重≈0。
* 构造：整数`T∈[-2,2]^3`枚举，`r=(frac_j-frac_i+T)@cell`取Cartesian真距离，替代`diff_frac-round()`；`r_cut`单值+五次平滑截断（一二阶导零）；`max_neighbors=48`；纯torch+SDPA math后端，无编译新依赖。
* 判赢：精度不掉+显存降量级+过旋转/平移/置换/图片重编号/晶胞基五项硬单测；挂一项则G2/G3免谈。

## 7. 几何G2低阶等变（已决断 2026-09-10）

* `e3nn 0.5.6`只开`0e+1o（+少量2e）`；`l=1/2`须经等变乘积/收缩成`0e`再用，禁普通Linear直投；`0o`仅受控通道，O(3)目标删`sin`分支。
* 判赢：固定G1，闭卷涨+五项不变性单测过才留；输则退回G1。

## 8. 几何G3 ACE高体阶（已决断 2026-09-10）

* 独立消融：`N=2-3, nu=[2,2,3], Lmax=2, 128x0e+48x1o+24x2e`，目标12-20M；`Lmax=3/nu=3`另起一臂。
* 前置：G2赢才做G2输不做；判赢只看闭卷+分层（元素长尾/晶系/成分重叠），闭卷掉0.2判死。

## 9. 解码Q组（已决断 2026-09-10，固定Encoder）

* Q0现状静态`edos_query[128,512]/phdos_query[64,512]`为锚点。
* Q1坐标MLP：`dataset`返`edos_x/phdos_x+task_id`（GridManifest边界/中心，eDOS以Fermi为零eV，phDOS以nu=0为零cm⁻¹，分开量纲），`Trunk=MLP(x)`，证坐标有用。
* Q2 Fourier：Q1+`[sin,cos](2πBx)`32-64频起步，eDOS高频/phDOS低频，+Softplus非负，推理clip训练域；判held-out query点，背下来不算赢。
* Q3 branch点积去Conv：`b=Branch(h_global), t=Trunk(Fourier(x)), y=Softplus(<b,t>+c)`；bin按盒平均（边界积分/采样）；与cross-attention memory为A/B，赢才有任意分辨率。
* 判赢：尖峰区+带隙/低频声学区+held-out query+闭卷gap缩小；输则停，不进G组。

## 10. 损失L组（已决断 2026-09-10，固定Q+G胜者）

* 顺序：`L0 SmoothL1+非负 → L1 一阶差分/峰加权 → L2 SumNorm+KL（合同修复后） → L3 Wasserstein/CDF消融 → L4 面积投影`。
* TV只记一阶差分损失，不记锐化定理；尖峰报峰位/峰高/积分三项，不只报总MAE。
