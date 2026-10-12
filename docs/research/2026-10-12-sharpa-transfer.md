# Sharpa迁移链：公开代码能证明什么

用户询问RL Lab如何迁移、相关论文怎么做，以及尺寸差异是否意味着造假。
2026-10-12只读核对官方固定版本、已归档论文全文；未运行训练、仿真或硬件。

尺寸差异不足以证明造假。可以确认公开默认训练几何与实物推荐不同，
但本次未核实演示所用权重、实际训练配置和物体实测尺寸的一一对应关系。
这限制了完整复现及归因，不能据此推断视频伪造，也不能保证默认配置必然复现演示。

## RL Lab官方代码

固定版本：`95ccda3d948801bb5da4cb7ffea766e03067a63b`。

1. [README](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/README.md)给出抓取缓存、PPO教师训练、`ProprioAdapt`蒸馏，再部署的完整顺序。
2. [默认训练配置](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py)启用质量10–250克、PD增益0.5–2倍、摩擦0.5–2倍、质心±1厘米、外力扰动及重力课程。教师使用特权信息，普通观测包含关节、控制目标和五指接触力；默认接触位置通道关闭。
3. [蒸馏代码](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/algo/padapt/padapt.py)冻结其它网络，只训练`adapt_tconv`，用30帧历史拟合教师特权编码的潜变量，损失为潜变量MSE。该历史也包含启用的触觉力，名称中的Proprio不意味着仅有关节。
4. [部署入口](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/scripts/deploy.py)明确选择`ProprioAdapt`，推理只传归一化普通观测和历史，不传仿真物体真值。
5. [部署配置](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env_cfg.py)要求手部校准、默认预热，目标控制20Hz；配置注释说明sleep控制不保证精确频率。实物触觉力乘1/1.5、低于0.2置零，接触位置默认关闭。[部署环境](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env.py)有触觉校准接口、预存初始化轨迹和目标增量限幅。这里只阅读代码，未调用驱动。

**尺寸开关确实存在，但默认范围为`[0.5,0.5,1]`，即固定尺寸。**
不能把其它动力学随机化或可配置尺寸接口描述成这个默认实验已经训练多尺寸。
默认仿真圆柱40×32毫米，README实物推荐48×60毫米；高径比0.8→1.25，
也不是相同物体的均匀放大。详见[之前的尺寸核对](2026-10-12-sharpa-object-size.md)。
触觉反馈和历史适应可能支持未见尺寸泛化，但本次源码核对不证明该演示实际依赖了哪种机制。

## 相关论文应分别理解

- [Tacmap，2602.21625v2](https://arxiv.org/abs/2602.21625v2)：III-C通过精密运动台采集真实触觉图像及几何深度标签，训练ResNet编码/解码器，把真实图像转换到与仿真一致的形变深度图。V-D在仿真训练PPO，实物部署时使用同一触觉表示；论文报告无实物策略微调，旋转展示操作球体。真实触觉翻译模型本身仍使用真实标注数据。它不解释RL Lab圆柱尺寸差异。
- [TacBPM，2609.18174v1](https://arxiv.org/abs/2609.18174v1)：III-B先训练八个球体专家，直径为8厘米的0.3–1.0倍，即24–80毫米，再蒸馏触觉条件行为先验；下游冻结先验并训练残差潜变量策略、适应解码器。IV-D实物使用标定接触力/位置和20Hz反馈，并按测得的1Hz正弦响应匹配关节增益；手内动作环不使用物体姿态反馈。这是多阶段、多尺度训练，不能把其论文结果直接归给RL Lab默认圆柱代码。

论文全文版本、下载日期和哈希沿用[已归档清单](2026-10-08-rotation-metrics/data/paper_manifest.json)。
本次直接下载并读取官方deploy.py、padapt.py及deploy_env.py；URL、时间及哈希
见[本次来源清单](2026-10-12-sharpa-transfer-sources.json)，原始文件位于忽略目录
`outputs/references/sharpa_transfer_2026-10-12/`。未进行新的论文数据库查询。
使用项目[既有paper-lookup流程及工具归属记录](../../references/README.md)。

## 与当前Boya结果的关系

当前B40M是固定几何、名义动力学的特权教师，策略输入没有触觉。
48×60毫米试验直接沿用40×32毫米原抓取、中心/朝向、操作区和观测常量，
没有完成上述历史/触觉学生蒸馏、硬件标定或尺寸覆盖训练。
这次9.15秒失稳只说明这个冻结Boya教师的直接尺寸迁移失败；
既不能否定Sharpa实物结果，也不构成其方法在Boya上的复现结论。
