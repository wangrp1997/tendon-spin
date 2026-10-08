# 开放式文献调研 v2：真实绳驱灵巧手上的持续指尖在手旋转

**目的：** 弄清现有研究做到什么程度，以及哪些困难仍有充分证据表明尚未解决。  
**本轮不做：** 预设解决方案、为某候选算法凑证据、推荐技术路线、设计新算法、修改控制代码/基线、切换分支、提交或推送。  
**保留：** [`docs/research_survey.md`](research_survey.md)（v1）；下文对 v1 中需纠正之处单独标注。  
**检索日：** 2026-09-29（本轮复核与补检）。  
**项目背景（只作问题语境，不作“缺口证明”）：** `baseline/jiang-full-stack`；五指绳驱、指腹三维力与阵列（阵列原理/剪切可观测性未确认）；目标为重力下无掌面或外部支撑的持续多圈旋转。自编采样 MPC 仅有仿真两圈记录；**外部方法的伯牙适配失败不计入研究缺口。**

---

## A. 核心论文证据表（15 篇）

说明：

- **任务类型：** `持续同向多圈` = 绕轴尽量多转；`多目标重定向` = 连续完成多个目标姿态（可累计多圈，但目标序列不同）。
- **支撑：** `无支撑指尖` / `掌托或掌接触` / `固定中心或铰链` / `桌面` / `承重接触待核`。掌心朝向不能单独证明接触部位。
- **速度：** 分列**物体净转速或累计转角**、**控制频率**、**求解耗时**；未在正文给出的写“正文未报”。
- **代码：** 分别记录官方来源、链接可达性、包的完整性与是否运行过；仅有可达链接不等于可完整复现。未找到官方发布不等于没有发布。

| # | 标题 / 年 | 论文 · 代码 | 任务 | 手与驱动 | 支撑 | 观测 | 真机结果 | 速度类指标 | 主要假设 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **In-Hand Object Rotation via Rapid Motor Adaptation (Hora)** / CoRL 2022 | [arXiv:2210.04887](https://arxiv.org/abs/2210.04887) · [github.com/HaozhiQi/hora](https://github.com/HaozhiQi/hora) · [haozhi.io/hora](https://haozhi.io/hora/) | 持续同向（世界 z）指尖旋转 | Allegro，全驱动位置/力矩栈 | **无掌面支撑**（§1） | 本体+动作历史；无视觉/触觉 | **有**：>30 物体，质量约 5–200 g（§1）；重物组平均累计约 **23.96 rad**，等价约 **0.8 rad/s**，episode 30 s（§5 Real-World Comparisons） | 控制约 **20 Hz**（正文控制图）；净转速见上；求解=策略前向（正文未报 ms） | 仿真预训练+域随机；外参由本体历史**在线推断**（非在线改模型权重）；主轴为 z |
| 2 | **General In-Hand Object Rotation with Vision and Touch (RotateIt)** / 2023 | [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) · [haozhi.io/rotateit](https://haozhi.io/rotateit/)（代码以项目页为准；本轮未核独立完整训练发布包） | **持续多圈**，多轴 | Allegro | **无支撑面**（§1 明确对比有支撑工作） | 深度+图像触觉提取的离散接触位置+本体（§3.2）；指令 **20 Hz**，PD **300 Hz** | **有**；仿真蒸馏后零样本部署（摘要/§1） | 控制 20 Hz；物体净转速正文因布局/图表分散，**本轮未抽出单一总表均值** | 教师用仿真特权信息；学生用噪声视触觉；持续转 vs 单次目标重定向在 §1 区分 |
| 3 | **AnyRotate: Gravity-Invariant In-Hand Object Rotation with Sim-to-Real Touch** / CoRL 2024 | [arXiv:2405.07391](https://arxiv.org/abs/2405.07391) · [项目页](https://maxyang27896.github.io/anyrotate/) · **本轮未在官方项目页找到训练代码链接，发布情况待核** | 多轴、重力不变；持续旋转（episode 内 Rotation Count） | Allegro + DigiTac 类指尖触觉 | **无支撑面精度抓取**（§1–2） | 触觉图像→接触位姿与力；控制 **20 Hz** | **有**：6 种手朝向；Dense Touch 在 palm-down 等表中 Rot/TTT 优于本体/二值触觉（Table 2–3；不将不同物体、手朝向和旋转轴的结果合并成一个转速） | Rot = episode 内转圈计数；TTT≤30 s；非求解器 Hz | 教师特权 RL + 学生蒸馏；触觉观测模型需真实压入标定数据 |
| 4 | **DROP: Dexterous Reorientation via Online Planning** / 2024–2025 | [arXiv:2409.14562](https://arxiv.org/abs/2409.14562) · [caltech-amber.github.io/drop](https://caltech-amber.github.io/drop/) · [github.com/Caltech-AMBER/drop](https://github.com/Caltech-AMBER/drop) | **多目标**立方体重定向；目标误差 <0.4 rad 算达标，新目标与前一个相差至少 90°（任务定义） | LEAP，全驱动；采样 MPC | 掌朝下约 **20°**，可滑落；**有掌几何**（§IV） | 多相机关键点→姿态；估计约 **90 Hz**，规划约 **25–50 Hz**（§IV） | **有**：10 次试验中 CEM 平均达标 **21.3±14.8 个目标**；去掉 80 s 未达标超时条件后为 **33.6±24.9 个**（Table I-B）；不是完整转圈数 | Table I-B 的 CEM Mean Rot/s=0.090±0.081，单位为达标目标数/s，不能直接换算成 rad/s | 依赖物体几何与视觉跟踪；并行 CPU rollout |
| 5 | **Complementarity-Free Multi-Contact Modeling and Optimization for Dexterous Manipulation (FREE)** / 2024–2025 | [arXiv:2408.07855](https://arxiv.org/abs/2408.07855) · [github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation](https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation)（论文给出） | 指尖空中 / TriFinger / **掌上**重定向至目标位姿 | Allegro、TriFinger、指尖系统（仿真） | **混合**：空中无掌；Allegro 例为 on-palm | **仿真状态真值**；正文无触觉闭环贡献 | **本轮未核到作者真机持续旋转实验** | MPC **50–100 Hz**（摘要）；姿态误差约 11°、位置约 7.8 mm、成功率 96.5%（仿真摘要） | 准动态/局部线性化接触几何；优化友好模型 |
| 6 | **Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking (Jiang)** / 2025 | [arXiv:2505.04978](https://arxiv.org/abs/2505.04978) · [github.com/Director-of-G/in_hand_manipulation_2](https://github.com/Director-of-G/in_hand_manipulation_2) · [项目页](https://director-of-g.github.io/in_hand_manipulation_2/)（README：硬件实现未完整开源） | 阀/卡/板/开盒/球等；含接触切换 | Allegro（仅仿真）/ LEAP + Tac3D（真机） | **阀/盒：铰链**；卡/板：**桌面**；球：**中心固定**（§7.1）；圆柱：静抓力跟踪（§7.5.1） | 触觉力 + 实验中简化视觉（RealSense/AprilTag）；高层约 **10 Hz**、低层约 **30 Hz** | **有**：LEAP 五类任务；Allegro 的 Rotate Valve / Rotate Sphere 仅仿真（§7.1、§8） | 频率=控制层频率；**不是**自由物体净转速 | 准动态（§3.1 Assumptions）；几何已知；低层 HFMC **跟踪补偿**模型误差 |
| 7 | **Contact-Implicit Model Predictive Control for Dexterous In-hand Manipulation: A Long-Horizon and Robust Approach (Jiang)** / 2024 | [arXiv:2402.18897](https://arxiv.org/abs/2402.18897) · 同仓库路线 | 长时域在手旋转（仿真） | Allegro | 球固定轴/中心或 **桌面支撑 Free**（正文实验设定） | 模型状态 | 以仿真为主 | 指令转速例：球 **0.29 rad/s**、方块 **0.4 rad/s**；控制器约 **20 Hz** 量级（正文） | CQDC/平滑接触；强调低层必要性 |
| 8 | **Real-Time Robust Finger Gaits Planning under Object Shape and Dynamics Uncertainties (Fan et al.)** / IROS 2017 | [arXiv:1710.10350](https://arxiv.org/abs/1710.10350) · 项目页 [msc.berkeley.edu/…/finger_gaits_planning](https://msc.berkeley.edu/research/finger_gaits_planning.html)；**未核到完整开源代码包** | 抬升+绕 z **连续旋转**+换指（仿真） | 多指仿真手；1D 触觉 | 仿真中物体可抬起旋转（非桌面固定）；**非真机** | 仿真位姿；1D 法向触觉；**不需**3D/6D 触觉与速度测量（摘要） | **无真机**；结论写 future real experiments（§VI） | 期望转速约 **0.2 rad/s**（§V）；双阶段规划在该仿真实验中平均每步计算 **<1 ms**（§V-C）；秒级数值是跟踪误差的稳定时间，不是求解时间 | 质量不确定性至约 40%、惯量约 50%（摘要/§V）；形状未知完整 3D |
| 9 | **OpenAI：Learning Dexterous In-Hand Manipulation** / 2018–2019 | [arXiv:1808.00177](https://arxiv.org/abs/1808.00177) · **本轮未核到官方完整训练+部署包**；不据此断言从未公开 | **多目标**方块/八角体重定向 | **Shadow 腱驱动** | **掌托**（任务在掌上） | PhaseSpace 指尖 + 物体标记或 RGB 视觉 | **有** | 策略约 **12 Hz**，底层更高（§2） | 大规模域随机；腱隙用仿真近似 |
| 10 | **Complex In-Hand Manipulation via Compliance-Enabled Finger Gaiting and Multi-Modal Planning (Morgan et al.)** / RA-L 2022 | [arXiv:2201.07928](https://arxiv.org/abs/2201.07928) · **本轮未核到完整官方代码** | **SO(3) 换指重定向**（非单纯同轴多圈 KPI） | Yale Model Q，**柔顺欠驱动** | **无支撑面**（摘要） | 视觉 6D；手无关节编码/触觉（摘要） | **有** | 低延迟视觉跟踪+在线改规划（摘要）；物体净转速正文未作统一 rpm 表 | 正交安全模态；柔顺接触 |
| 11 | **On the Feasibility of Learning Finger-gaiting In-hand Manipulation with Intrinsic Sensing (Khandate et al.)** / ICRA 2022 | [arXiv:2109.12720](https://arxiv.org/abs/2109.12720) · **本轮未核到广泛使用的官方复现包** | 轴旋转，可达/超过一整圈级换指 | 仿真灵巧手 | 精度抓取；强调相对掌托更难 | 本体±触觉；无显式物体位姿 | **以仿真为主**；相关工作提及真机但本文充分真机多圈表**本轮未核到** | 正文以学习可行性为主 | 初始状态分布工程；RL |
| 12 | **Sampling-Based Model Predictive Control for Dexterous Manipulation on a Biomimetic Tendon-Driven Hand (Hess et al.; Faive)** / 2024–2025 | [arXiv:2411.06183](https://arxiv.org/abs/2411.06183) · 原列仓库 `srl-ethz/faive-mujoco-mpc` 在核验日返回 **404**，本论文官方实现地址待确认；另有 [faive_gym_oss](https://github.com/srl-ethz/faive_gym_oss)，其 README 对应 **2023 年 RL 工作**，不能替作本论文 MPC 代码 | 球 **滚动/翻转/接住** | **Faive 腱驱动**柔顺手 | 手相对水平约 **20°**；滚动/翻转**使用掌面**；翻转时球可离手 | 运动捕捉标记球（接/翻）；MJPC 采样；EKF 腱长→关节 | **有**：滚动完整一圈约 18 s，平均约 **0.35 rad/s**；接球成功率约 **67%**（摘要/硬件节） | 物体转速 0.35 rad/s；VLM 调代价每轮 <2 min（摘要） | 需调任务代价；动态任务依赖外跟踪 |
| 13 | **Learning Robust Dexterous In-Hand Manipulation from Joint Sensors with Proprioceptive Transformer (Yao et al.; ORCA)** / 2026 | [arXiv:2605.21330](https://arxiv.org/abs/2605.21330) · ORCA 硬件/SDK：[github.com/orcahand](https://github.com/orcahand)；**PT 训练代码是否完整开源：本轮未核到论文专用仓库** | **持续**绕 z 转立方体 | **ORCA 腱驱动**；关节磁编码 vs 电机编码对比 | **掌心朝上；是否存在掌面承重待核**。§IV-A 仅写 palm-up，§IV-C 提到拇指从下方支撑，不能由手朝向推定掌托 | **仅关节角/速历史**；策略 **20 Hz** | **有**：相对基线转速约 **3.1×**，图示可达约 **11.8 rpm** 量级与 100% rotation accuracy（Fig.1）；关节传感相对电机编码约 +26.8% 转速 | rpm 与准确率；非 MPC 求解时延 | 教师特权 RL→学生蒸馏；腱传动 mismatch 为动机 |
| 14 | **Rotating without Seeing: Towards In-hand Dexterity through Touch (Touch Dexterity)** / 2023 | [arXiv:2303.10880](https://arxiv.org/abs/2303.10880) · [官方项目页](https://touchdexterity.github.io/)链接到 [in-hand-rotation](https://github.com/YingYuan0414/in-hand-rotation)；仓库网页可达，完整性与运行情况未核 | 绕 x/y/z 掌上旋转 | Allegro + 密布二值力 | **掌托** | 二值触觉（掌+指）；无视觉 | **有**：多物体 | 正文强调相对无传感基线的圈数/卡住；非统一 rpm 表 | 二值接触缩小 sim2real |
| 15 | **Adaptive Contact-Implicit Model Predictive Control with Online Residual Learning (Huang et al.)** / ICRA 2024 | [arXiv:2310.09893](https://arxiv.org/abs/2310.09893) · [作者项目页](https://sites.google.com/view/adaptive-contact-implicit-mpc/home)；完整官方代码包待核 | 臂端滚动未知球/水果，跟踪桌面路径 | Franka Emika Panda | **桌面支撑** | 视觉状态估计；触觉接入列为未来工作（§VIII） | **有**：用粗略几何先验在线适应并完成路径跟踪（§VII-B/C） | 混合接触模型残差**在线更新约 20 Hz**（摘要、§II）；不是物体转速 | 改变模型残差参数，不只是冻结网络推断；不是多指无支撑旋转 |

**核验后的关键澄清：**

1. PT/ORCA 已证明腱驱动真机连续转方块；**掌面是否承重仍待核实**。不能仅依据 palm-up 把它排除在无掌托先例之外，也不能反向认定其完全靠指尖。
2. Faive+MJPC 证明采样 MPC 可在腱驱动真机执行滚球等任务；本次核到的 MPC 仓库地址失效，不能把另一篇工作的 RL 仓库当作该论文实现。
3. Fan 2017 为仿真证据；**平均每步计算 <1 ms** 与秒级跟踪稳定时间是两种指标。
4. DROP 的 Rotations 是**达标目标数**，Hora 的 Rotations 是**累计弧度**，AnyRotate 的 Rot 是**圈数**。同名指标不代表同一单位，不能直接排名。
5. Jiang 由高层规划接触切换、低层用触觉跟踪补偿模型误差；Allegro 球体示例仅仿真。其 100 组静抓测试关闭重力，详见 B6。
6. v1 的推荐方向不属于本轮结论。下面仅比较证据与边界。


---

## B. 五领域综合比较

### B1. 无支撑指尖持续旋转、换指与承重转移

| | 内容 |
|---|---|
| **已有充分证据做到的** | 在 **Allegro 类全驱动手**上，Hora / RotateIt / AnyRotate 以真机展示**无掌面支撑**的指尖持续旋转与换指步态（#1–3）。Morgan 在柔顺欠驱动手上以视觉规划完成无支撑 SO(3) 换指重定向（#10）。Fan（仿真）展示抬升+连续转+换指（#8）。 |
| **成立条件** | 多为仿真大规模训练或柔顺手专用规划；观测为本体 / 视触觉 / 视觉 6D；物体多为中等尺寸日常物或几何体；episode 长度有限（如 30 s）。 |
| **仍缺充分证据** | （i）**绳驱 + 无掌托 + 指尖承重 + 持续同向多圈**同时成立的真机系统：OpenAI Shadow 与 Faive 的已核任务分别使用掌上重定向、掌上滚球；**PT/ORCA 的掌面承重尚待核验**。本轮不能据这些分类宣称该组合尚无人完成。 （ii）**模型基 MPC**在真机上完成与 Hora 同级的无支撑指尖持续多圈：FREE/Jiang/DROP 的真机或设定边界与此不对齐（#4–7）。 （iii）系统报告**承重转移是否满足 wrench 可维持**的统一协议（多数报掉落/TTF/Rot，少报卸载瞬间剩余指载荷裕度）。 |

### B2. 绳驱、欠驱动、柔顺手上的真实在手操作

| | 内容 |
|---|---|
| **已解决（有真机）** | Shadow 腱驱掌上重定向（OpenAI）；Faive 腱驱采样 MPC 滚球/翻转/接球；ORCA 腱驱关节传感连续转方块（掌面承重待核）；Yale 柔顺欠驱动无支撑 SO(3) 换指（Morgan）。 |
| **条件** | 支撑条件按论文逐项区分，不能由 palm-up 推断掌托；部分任务允许球离手，或采用专用柔顺模态；常需外跟踪（Faive 动态）或特权蒸馏（PT）。 |
| **仍缺证据** | 绳弹性、回差、腱鞘摩擦与**无支撑指尖持续旋转**的联合真机量化（谁在何种误差下掉落）——公开论文很少同时给出腱传动误差模型与无支撑多圈 KPI。PT 证明**关节侧传感优于电机侧**对其 ORCA 转方块任务重要（#13），但不能外推到无支撑指尖或未装关节磁编的手。 |

### B3. 触觉控制、接触/滑移估计与视觉遮挡下的操作

| | 内容 |
|---|---|
| **已解决** | Jiang：触觉力跟踪接触参考，补偿简化模型误差（#6）。AnyRotate / RotateIt：密集或图像触觉提升无支撑旋转稳健性；AnyRotate 展示不稳定抓取时的反应式步态（Fig.7）。Touch Dexterity：掌面二值触觉相对无传感改善掌上旋转。Hora：证明**无触觉**也可做无支撑 z 转，但局限写明接触点不准导致失败（Discussion）。 |
| **条件** | 传感器类型已知且经标定（Tac3D、DigiTac、FSR 等）；遮挡问题常用“不用视觉”或“多相机”绕开。 |
| **仍缺证据** | （i）伯牙阵列的原理未明是**项目硬件信息缺失**，不是领域空白；需区分压力分布与可测剪切/纹理运动的传感器。 （ii）视觉完全遮挡下，仅靠指腹力/阵列完成**可计量净转角任务**且控制器不吃外置转角真值的通用方案；对称圆柱尤其困难（外形对轴向角不可观）。 （iii）Jiang 写明实时视觉跟踪超出范围、实验中简化（§9.1）。 |

### B4. 未知动力学、摩擦与传感误差下的鲁棒性与迁移

| | 内容 |
|---|---|
| **已解决** | Hora：本体历史适应质量/尺度等，零样本多物体。Fan：仿真中质量/惯量大幅不确定下跟踪+换指。Huang：约 20 Hz 在线更新混合接触模型残差，在真实桌面滚动中适应未知几何（#15）。Jiang：粗模型+触觉跟踪，扰动下完成多任务；limitation 仍建议系统辨识可再提升。OpenAI/DeXtreme 路线：域随机迁移。DROP：硬件上对规划器/估计消融。 |
| **条件** | 随机化覆盖训练分布；或任务准动态；或掌托降低失稳。 |
| **仍缺证据** | （i）**驱动迟滞、摩擦、传感器位姿误差同时未知**时，先需明确传感和激励条件；本轮未形成其可辨识性的系统证据，不能将未检索到联合辨识当作可解性或创新性证明。 （ii）物体尺寸/质量变化下，**模型基无支撑指尖持续转**的真机迁移（与 RL 域随机不是同一证据）。 （iii）把“伯牙适配失败”当缺口——**证据不足且方法上无效**（任务/模型边界不同）。 |

### B5. 实时规划与高速操作（模型 / 学习 / 混合）

| | 内容 |
|---|---|
| **已解决** | 学习策略真机常用 **12–20 Hz** 指令（OpenAI、Hora、AnyRotate、PT）。DROP 规划 **25–50 Hz**。FREE 仿真 MPC **50–100 Hz**。Jiang 分层 10/30 Hz。Fan 报告平均每步计算 <1 ms（仿真）；Huang 报告约 20 Hz 在线残差更新（真机桌面任务）。Faive：采样 MPC 真机滚球约 **0.35 rad/s**。Hora 重物组约 **0.8 rad/s** 等价转速。 |
| **必须区分的指标** | **物体净转速**（rad/s 或 rpm）≠ **控制频率**（Hz）≠ **单次求解墙钟时间**。AnyRotate 的 Rot/TTT、DROP 的达标目标数、Hora 的 radians、PT 的 rpm **定义不同**。 |
| **仍缺证据** | （i）模型方法在**不可暂停物理**的自由无支撑指尖持续转上的真机实时性。 （ii）本轮尚未建立与人手对齐的转速测试，也未统一物体、支撑、失败处理和任务单位，因此不判断是否接近人手。 （iii）混合方法在绳驱无支撑设定下的系统速度–稳健帕累托曲线。 |

### B6. 不同初始抓取：生成分布与验证范围

“多个目标姿态”“多个物体”与“多个初始抓取”是不同变量。训练中随机初始化也不等于已验证任意真机抓取。

| 工作 | 正文核到的初始化方式 | 可以支持与不能支持的结论 |
|---|---|---|
| Hora | 在标准抓取附近随机物体位置、姿态和手关节位置，筛出稳定抓取（§3.1 Object Initialization and Dynamics Randomization）；仿真评价跨随机参数与初始化（§5） | 明确不是只从一个关节配置训练；未由这些文字建立任意真机初始抓取成功率 |
| RotateIt | 在标准抓取上施加关节偏移 U(−0.25, 0.25) rad，仿真 0.5 s，按接触/高度等条件筛选；**每物体、每尺度预采样 400 个抓取**（补充材料 Stable Precision Grasp Generation） | 使用稳定抓取库；对象已被抓住是任务前提；不等于从任意接触拓扑或不稳定状态起步 |
| AnyRotate | 物体随机姿态、标准手姿态叠加 U(−0.3, 0.3) rad，仿真 **6 s** 并依次改变六个重力方向；按条件保存**每物体 10000 个抓取**（Appendix C）；Fig.7 展示不稳定抓取恢复 | 有抓取分布与恢复示例；单个恢复示例不是未见初始抓取分布上的统计成功率 |
| Jiang | BODex 生成 **100 个不同物体的抓取**，用于低层静态力跟踪；物体用带阻尼自由关节，**关闭重力**（§7.5.1、Fig.12） | 证明多抓取条件下的力跟踪；不能据此证明重力下不同抓取的持续旋转 |
| PT/ORCA | 训练时物体初始姿态进行 SO(3) 随机采样（§III-B）；真机按物体尺寸评价（§IV、Table I） | 随机物体姿态不自动等于独立覆盖不同手指接触配置；本轮未核到真机按初始抓取分组的成功率 |
| DROP / FREE / Fan / Morgan | 各自有任务起始状态、随机目标或跟踪实验；本轮未核齐“初始关节/接触分布 × 真机重复次数”的统一协议 | 保留为待核，不能把未报告写成固定初始化，也不能由连续多目标成功推导任意抓取泛化 |

**当前证据边界：** 不同初始抓取并非无人研究。后续论述必须区分抓取库内随机、未见配置、接触拓扑变化与不稳定抓取恢复，不能统称“随机抓取泛化”。

### B7. 在线适应：参数更新、历史推断与反馈补偿

| 类型 | 代表与实际机制 | 证据边界 |
|---|---|---|
| 历史推断 | Hora 从本体历史推断隐变量；RotateIt/AnyRotate/PT 从历史传感输入推断隐状态（各自方法章节） | 部署时有随观测变化的估计，不等于在线更新模型或网络权重 |
| 在线模型更新 | Huang 更新接触隐式模型的混合残差，约 20 Hz（§II、§VI–VII） | 有桌面真实滚动证据；触觉接入列为未来工作（§VIII），不直接覆盖多指承重换指 |
| 反馈跟踪补偿 | Jiang 的 HFMC 用实测触觉修正运动与力跟踪（§6） | 已能补偿模型误差；不是在线学习全套物理参数 |
| 鲁棒控制 | Fan 用鲁棒控制与力优化处理规定范围的质量/惯量不确定性（§IV–V） | 不能与数据驱动辨识混称；其验证为仿真 |
| 离线校准与随机化 | AnyRotate 离线辨识 16 DoF 的刚度、阻尼、质量、摩擦、armature，共 **80 个参数**（Appendix D），另做训练域随机化（Appendix E） | 真机迁移结果不等于无需标定；应报告校准与训练成本 |

上述类别已有明确先例，因此“在线适应”“使用触觉”或“处理未知动力学”本身均不能作为新贡献。不同机制在相同传感、初始化、支撑与计算预算下的可比证据仍需逐项审查。

---

## C. 有文献依据的未解决问题 vs 看似新颖但已有人解决

### C1. 尚未形成充分证据的验证问题

以下是这次调研的证据边界，不等同于已确认的科学空白，也不构成方法创新声明。

1. **绳驱、无掌面承重、持续同轴多圈的联合真机协议。** 最近的 PT/ORCA 已有腱驱连续旋转，掌面承重仍需核实；Faive、Shadow 的已核任务条件不同。不能在 PT 分类未完成时排除其相关性。
2. **模型方法在重力下持续指尖旋转的迁移与实时证据。** Jiang、DROP、FREE、Fan 分别覆盖不同支撑条件、控制链与仿真/真机；Huang 展示在线接触模型适应，但为桌面任务。需要针对明确任务比较，不能只依据“不是我们的手”认定不足。
3. **未见初始抓取的真机泛化。** Hora、RotateIt、AnyRotate 已使用多抓取初始化；目前汇总尚不能统一回答接触拓扑改变、抓取库外初始化和驱动误差叠加后的真机成功率（B6）。
4. **误差组合下的性能边界。** 已有模型补偿、鲁棒控制、在线残差与域随机化先例（B7）；本轮尚无同一手、相同输入权限下，对驱动、接触、传感误差分别及联合影响的可比汇总。这是当前证据不足，不是断言这些问题无人研究。
5. **统一的效率与承重转移报告。** 目标达标数、净转角、控制 Hz 与计算时延不同；多数论文没有共同的物体、初始抓取和停止条件。当前不能做跨论文性能排名。

### C1a. 项目信息缺失与物理观测限制

- **阵列原理未确认**是伯牙硬件信息缺失。压力阵列、含剪切纹理的光学触觉和三维力传感具有不同可观测量，不能据此宣布领域尚未解决触觉滑移估计。
- 对理想无纹理、轴对称圆柱，**仅凭对旋转不变的外形观测不能唯一确定绝对轴向角**。这是观测对称性造成的歧义，不能把“未找到能恢复它的论文”当作有望由更强算法解决的空白。标记、可跟踪纹理、接触运动假设等会改变观测条件。
- **独立测量评测真值**与**控制器使用物体姿态**要分开。AnyRotate 的标记用于人工计数（§4 Evaluation），其存在不表示标记角度输入策略。
- 驱动迟滞、摩擦和传感器位姿未必能同时唯一辨识；闭环鲁棒操作也未必需要逐项辨识。需要先限定输入、传感、激励与模型，不能预设“全参数联合辨识”是必须解决的问题。

### C2. 看似新颖、但已有工作在所述条件下解决或强覆盖

| 看似新颖的说法 | 实际覆盖 |
|---|---|
| “第一次无支撑指尖持续转” | Hora / RotateIt / AnyRotate 已在 Allegro 真机上做（条件见 #1–3） |
| “第一次用触觉做在手旋转” | Touch Dexterity、RotateIt、AnyRotate、Jiang 等已覆盖不同支撑设定 |
| “第一次换指规划” | Fan 2017、Morgan、Khandate、Jiang CIMPC 等 |
| “第一次处理质量不确定” | Fan（仿真鲁棒控制）；Hora（适应）；域随机 RL |
| “第一次采样 MPC 真机在手” | DROP（LEAP）；Faive（腱驱滚球）——注意任务边界 |
| “第一次腱驱动真机连续转方块” | PT/ORCA 已有连续转证据，掌面承重待核；OpenAI Shadow 另有多目标重定向，二者任务不能混称 |
| “模型误差只能靠重新建模” | Jiang 触觉 HFMC 已做反馈补偿；Huang 已做在线混合残差更新（#6、#15） |
| “没有视觉就不能转” | Hora（本体）；PT（关节）；Touch Dexterity（二值触觉+掌） |

---

## D. 现有开源方法可复用性（相对伯牙类绳驱手）

| 资源 | 能直接用什么 | 需要适配什么 | 缺少的硬件/依赖 |
|---|---|---|---|
| **Hora** | Isaac/训练与适应框架思路；公开代码仓库 | 手模型、动作空间、无特权物体信息时的评测协议 | Allegro 与绳驱动力学不同；无现成伯牙腱模型 |
| **RotateIt / AnyRotate** | 论文中的多模态/触觉观测与蒸馏方法；官方完整代码包仍待核 | 触觉仿真、手模型与真实阵列标定链 | 传感器差异、训练与校准数据；不能把未找到链接写成没有代码 |
| **DROP + MJPC** | CrossEntropyPlanner、并行 rollout、视觉关键点栈 | 手 MJCF、残差、目标序列；执行/规划 MuJoCo 版本 | LEAP 与多相机；CPU 大量线程 |
| **FREE** | `MPCExplicit` / complementarity-free 模型 | 运动学、接触几何、自由物体代价 | 仿真状态；IPOPT；无触觉接口 |
| **Jiang** | 高层 DDP/CQDC、低层 HFMC、ROS2 仿真节点 | 自由物体维数、腱耦合、摩擦；硬件未开源 | Drake/qsim/Crocoddyl、Tac3D、Docker；准动态假设 |
| **Faive–MJPC** | 真机先例与论文中的 MJPC 集成方法；本论文专用代码地址待确认 | 不同手几何与滚球任务 | 原列链接 404；`faive_gym_oss` 为另一篇 RL 工作，不是该 MPC 实现 |
| **ORCA 生态** | 开源腱驱手硬件/SDK/仿真（orcahand） | PT 策略代码完整性与掌面承重条件待核 | 关节磁编；与伯牙传感配置不同 |
| **Fan 2017** | 算法思想（速度级换指 LP + 鲁棒力控） | 几乎需重实现 | 本轮未核到完整官方代码；仅仿真验证 |
| **Touch Dexterity** | 官方项目页链接的 `in-hand-rotation` 仓库网页已核可达 | 手模型、二值触觉布局、任务与输入接口 | 代码内容完整性及硬件链未验证；本轮未运行 |
| **Huang 在线残差 MPC** | 论文中的混合残差更新与 MPC 方法 | 从桌面滚动到多指任务需模型/观测适配 | 官方完整实现待核；论文中的触觉接入是未来工作 |
| **OpenAI / Morgan / Khandate** | 问题定义与基线协议参考 | 训练与硬件适配，代码包完整性需核验 | Shadow/Yale 等不同硬件与观测配置 |

**共性限制：** 开源求解器≠原论文任务已在绳驱无支撑指尖上复现；复用时需适配模型，并明确任务与共同验收协议，且**不能**用适配失败反推原方法无效。

---

## E. 检索过程、来源与待核验项

### E1. 检索日期与策略

- **日期：** 2026-09-29  
- **起点精读 PDF：** `1710.10350`、`2210.04887`、`2309.09979`、`2405.07391`、`2409.14562`、`2408.07855`、`2505.04978`、`2402.18897`、`2411.06183`、`2605.21330`、`1808.00177`、`2201.07928`、`2109.12720`、`2303.10880`、`2310.09893`（`pdftotext` / HTML）。
- **引用追踪：** Semantic Scholar `ARXIV:1710.10350/citations`（返回含 Khandate 2109.12720、Dexterous In-hand Manipulation by Guiding Exploration 2303.03533 等；API 有 429 限流）。  
- **关键词（示例）：** `in-hand rotation fingertip unsupported`；`tendon-driven dexterous in-hand`；`finger gaiting uncertainty`；`tactile in-hand occlusion`；`sampling MPC biomimetic hand`；`proprioceptive transformer ORCA`；以及 Jiang/DROP/FREE/Hora/AnyRotate 专名。  
- **来源：** arXiv、OpenAlex、Semantic Scholar、项目主页、GitHub、已有检索记录（不以聚合站替代官方发布核验）、本地依赖 README（`~/Documents/in_hand_manipulation_2` 等只读）。

### E2. 本轮定点核验记录与仍待核项

核验日期为 **2026-09-29**。上述核心修订依据论文正文与以下官方网页；没有在本轮运行外部代码。

| 核验对象 | 可定位来源与结果 | 尚不能推导的结论 |
|---|---|---|
| Fan 时间单位 | [论文](https://arxiv.org/abs/1710.10350) §V-C：“average computation time … less than one millisecond for each time step”；另文中的 1.1064/2.2138 s 为跟踪稳定时间 | 仿真实验耗时不能直接保证伯牙硬件端到端频率 |
| DROP 指标 | [论文](https://arxiv.org/abs/2409.14562) 任务定义、Table I-B；CEM 21.3±14.8 个达标目标，去超时后 33.6±24.9 个 | 这些次数不能当成完整 360° 圈数 |
| ORCA 支撑分类 | [论文](https://arxiv.org/abs/2605.21330) §IV-A 写 palm-up；§IV-C 描述拇指从下方支撑 | 掌面是否承重、是否全程仅指尖接触需视频/几何或接触证据；本轮未核实，明确保持待核 |
| Touch Dexterity 源码 | [官方项目页](https://touchdexterity.github.io/)链接到 [仓库](https://github.com/YingYuan0414/in-hand-rotation)，两网页可达 | 完整训练/部署链及可运行性未验证；GitHub API 查询受 403 限流，未将网页可达等同于完整复现 |
| Faive 代码 | 原列 `srl-ethz/faive-mujoco-mpc` 返回 404；[faive_gym_oss README](https://github.com/srl-ethz/faive_gym_oss)明确链接 [2308.02453](https://arxiv.org/abs/2308.02453)，为 2023 年 RL 工作 | 不确认该地址从未存在；不将它替作 [2411.06183](https://arxiv.org/abs/2411.06183) 的 MPC 实现；MPC 官方发布地址待核 |
| AnyRotate 代码 | [官方项目页](https://maxyang27896.github.io/anyrotate/)本轮未找到训练代码链接 | 不宣称没有官方代码，后续可追踪作者发布 |
| 初始化 | Hora §3.1；RotateIt 补充材料 Stable Precision Grasp Generation；AnyRotate Appendix C；Jiang §7.5.1；PT §III-B | B6 保留训练随机化与真机初始抓取覆盖范围的区别 |
| 在线适应 | [Huang](https://arxiv.org/abs/2310.09893) §II、§VI–VIII；[AnyRotate](https://arxiv.org/abs/2405.07391) Appendix D–E | 不把在线残差、离线辨识或冻结策略的历史推断混为一类 |

本轮仍未完成：PT 掌面承重核验；未核方法的完整官方代码与硬件链；各方法真机初始抓取分布的统一统计；更近期工作的持续引用追踪。它们不支持“领域首次”或跨论文胜负排名，且无需为保留这些边界先选择一种新算法。

### E3. 证据使用约定（再声明）

- “未找到证据”≠“该方法没有做过”≠“领域首次”。  
- 伯牙适配失败≠文献缺口。  
- 仿真真值 / 在线推断（冻结网络） / 在线学习（更新模型）必须分开写——本表已按此区分。

---

**本轮完成定点事实修订、初始抓取与适应机制补充，止于文献与问题梳理。** 旧报告 [`research_survey.md`](research_survey.md) 保留；控制代码与基线参数未改。
