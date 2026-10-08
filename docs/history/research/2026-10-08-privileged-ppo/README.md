# 特权 PPO 首轮：净转94.62°，稳定续转仍未解决

2026-10-08。用户提出学习路线后，另立`privileged_rotation_ppo_v1`，
保留原40×32mm/50g/抓取44、13路原kp10位置输入、腕/LF固定、
补丁MuJoCo3.13 native/.5ms。此前的无学习方法与结果继续作为基线。
学习策略输入真实关节/对象状态及载荷，20Hz直接协调全部13输入；
没有预设换指序列、模仿旧MPC、切换控制器或改变引擎。

**正式首轮已完成131072动作步/13022768实际物理步/64次PPO更新，
训练1805个episode，总墙钟325.23s，按预定更新预算结束。**
初始与第16/32/64更新策略各冻结后从原始抓握独立评测，声明120s
窗，无reset/切换/训练中途接管；结果如下，不汇总训练episode角度。

|冻结checkpoint更新|有效执行s|终点净角°|峰值净角°|实际停止原因|
|---|---:|---:|---:|---|
|0（未训练对照）|3.092|5.344970|5.393079|真实支持不足|
|16|4.8795|59.526901|59.526901|漂移超过5mm|
|32|4.1465|79.582029|79.582029|轴倾超过15°|
|64|2.506|94.622083|94.622083|漂移超过5mm|

**最终checkpoint在有效前缀净角上超过旧补丁native v6的75.003724°，
但2.506s有限推进不等于持续旋转。** 首违规帧2.5065s漂移5.000800mm，
轴倾14.621699°、TH/FF/RF仍真承托，MF载荷0；只位置门槛触发，
没有观察到该帧已经掉落。有效段最少3带载指、max组对象力5.040980N、
对象/自穿透.410242/.327237mm、无外撑/数值违规。
最终累计正94.622083°/反0°；其余checkpoint反转保留在原净角。
所有checkpoint的每物理步指令/qpos/qvel/电机/物理指标/标记/完整
末态原生独立重放0，存储观测上的冻结策略推理重放0；源码/引擎/
checkpoint及原初態身份保留。没有一个checkpoint完成完整120s窗口。

当前训练改进了短时推进，没有学出稳定反复换指和后继。第32到64
更新净角增加15.04°，有效长度却4.1465→2.506s；最近100训练episode
平均有效时长2.86578s，76漂移/24轴倾停止，平均净角87.39854°。
最终失败episode训练return仍+10.83755（已含−5失败罚），所以该
奖励允许快速有限推进后越界获得正回报。这是已观察到的目标不足，
尚未证明唯一原因，也没有证明单独加大失败罚就能学会持续换指。
第一轮仅一个seed、原抓握/名义动力学/短训练，不能当RL全局上限、
原论文SOTA、受限观测控制或现实误差鲁棒证明。

[首轮小结果与完整来源](data/pilot.json) · [逐checkpoint小表](data/evaluations.csv) ·
[更新实际比较](../../experiments/2026-10-07-rotation-demo-benchmark/FOLLOWUP_COMPARISON.md) ·
[最终冻结策略实际视频，原速](../../../outputs/research/learning/privileged_ppo_v1/pilot_01/demo_update_0064.mp4)。
视频只显示原始有效物理帧，末帧停留1s没有新增执行，蓝条仅显示标记；
[视频来源核验](data/video.json)。原始训练/全物理数组/权重留ignored outputs。

此前已完成GPU learner与原cp311物理worker连接，以及一次独立
基础设施smoke：1环境×32步×2次更新，共64动作/6167实际物理步，
6次训练episode结束，墙钟5.05s。初始和第2更新checkpoint分别从
原初态评测.5s，均到时限，物理全数组独立重放0、存储观测上的
冻结策略推理重放0。**这是链路验证，不是120s旋转能力成绩。**
三项PPO测试确认时间截断bootstrap、reset隔断GAE及饱和动作的有限
概率/梯度。未据该短测试宣称持续旋转、超过基线或真机鲁棒。

首轮遵守seed44、8环境×256步×64更新和900s阶段预算，不据第32更新
超过75°提前报持续成功或改控制器。继续学习路线的下一验证要求是
长程承托/位置/姿态与换指接续的收益，而非仅追求更快单段旋转；
后续训练协议尚未启动。不追加本轮事后奖励扫描，不直接进行真机
或观察受限学生部署；先取得稳定名义学习策略的独立长程证据。

[执行前协议](PROTOCOL.md) · [smoke小结果](data/smoke.json) ·
[环境](../../../research/learning/environment.py) ·
[PPO训练](../../../research/learning/train.py) ·
[冻结策略评测](../../../research/learning/evaluate.py)。

运行（复用现有Torch/CUDA环境，不修改全局Python环境）：

```bash
/home/rw/miniconda3/envs/env_isaaclab/bin/python -u -m research.learning.train \
  --out outputs/research/learning/privileged_ppo_v1/NEW_RUN
```

独立重评测已保存checkpoint（不能使用既有输出目录）：

```bash
/home/rw/miniconda3/envs/env_isaaclab/bin/python -u -m research.learning.evaluate \
  --checkpoint outputs/research/learning/privileged_ppo_v1/NEW_RUN/checkpoints/update_0064.pt \
  --out outputs/research/learning/privileged_ppo_v1/NEW_EVALUATION
```

参考Hora的特权PPO→本体历史适应与AnyRotate的触觉学生/误差处理。
文献/官方README有在Allegro旋转的依据；本实现是伯牙原生PPO起点，
不是完整复现两篇工作，也不是它们能解决本手任务的保证。
先检验名义特权策略，再分别验证域随机化和受限观测学生。
