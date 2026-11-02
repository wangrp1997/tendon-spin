# TendonSpin C1260M 学生蒸馏

本项目自己的30帧本体/触觉历史学生。教师固定为已选定的C1260M；与旧20帧部署策略、botyard的RGB-D学生使用不同的检查点接口。

## 输入与网络

学生只接收传感历史，每帧57维：16路主动关节角、16路已发目标、4路独立读取的被动J1，以及五指接触、力幅值、触觉有效性、时效和真实/填充帧标记。角度按已知限位处理，目标保留弧度。三轴力在SDK边界取范数；使用固定量程和`log1p`保留力大小，不做逐帧单位向量归一化。第一版不依赖相机，也不输入物体位置、质量或摩擦真值。

教师原96维短观测、8维latent和16维动作头保持原尺寸。57维学生历史通过独立TCN预测latent，再进入冻结的原动作头。教师观测归一化器随学生保存，训练与部署都保持冻结。被动J1增加观测，不增加独立动作通道；仿真沿用当前1∶1耦合，不安排专项比例测试。

仿真触觉来自各指**distal刚体**与物体接触的法向力加摩擦力合力。这是理想刚体代理，尚未限制到实物传感器的局部覆盖区域，也不模拟压力阵列。当前力阈值、饱和与滤波参数属于显式工程默认值，不是实测传感器标定结果；均保存在`observation_contract.json`和检查点内。

## 训练与独立验证

- 同时训练`touch`与`proprio`两个学生；后者屏蔽全部触觉、触觉有效性和时效，保留相同的关节输入和网络规模。
- 开始采集教师控制的轨迹，随后逐步转为学生控制；两个学生访问的状态都进入共享训练回放。教师只提供动作和latent监督，不进行PPO更新。
- 监督目标为动作MSE加latent MSE；只优化历史编码器。回合结束立即清空该环境历史，失败前缀保留，不拼接到下一回合。
- 九种尺寸均分训练和验证环境。验证数据来自独立的教师回合，只用于选择检查点，不用于梯度更新；默认采集前400个控制步的完整固定验证集，收集齐后才选best。此前的验证分数只作诊断，避免随验证集增长而比较不同口径。使用原训练抓取缓存，因此不声称验证了未见抓取或未见物体形状。
- 训练结束后自动用新测试种子，配对评测教师、触觉学生和本体学生。每例最多30秒；失败保留有效前缀，不接管、不重置。对照实际初态和物理参数，核验外力/关节噪声刺激一致。
- 仿真接入门槛预先固定为教师平均净旋转的90%，且30秒保持率较教师下降不超过0.1。报告所有对照结果；达到门槛也不等于实物验收通过。

## 运行

使用已安装Isaac Lab的本项目虚拟环境。先通过`CUDA_VISIBLE_DEVICES`显式绑定可用的物理GPU，再运行：

```bash
PYTHONPATH=. .venv/bin/python -u scripts/run_boya_student_distillation.py \
  --out outputs/boya_touch_student_20261102_v1 --num-envs 1008
```

所有输出目录必须是新目录，禁止覆盖旧实验。入口先训练，再执行独立闭环评估；基础设施错误保留日志并停止，不自动重试。

结果位于输出目录下：

- `status.json`：整个队列状态；`training/status.json`：采集、更新、验证及触觉覆盖情况。
- `training/touch_best.pth`、`training/proprio_best.pth`：验证集选择的学生。
- `training/*_last.pth`、`training/learner_latest.pth`：最新学生与优化器状态。当前入口尚不接受自动续跑；恢复时需明确缓存重置和回放重建，不宣称精确物理续接。
- `training/losses.jsonl`、`validation.jsonl`、`episodes.jsonl`：训练及回合结果。
- `training/data/`：逐控制步的实际传感帧、教师动作/latent、执行动作、回合ID和驱动来源；`control_complete.npy`标记实际完成的控制，未完成的预分配尾部不算数据。来源值`-1/0/1`依次表示教师/触觉学生/本体学生。`validation_samples.npz`保存固定验证集。
- `evaluation/comparison.json`：最终配对比较；`fixed_cases.npz`、`*_rollout.npz`：实际案例、控制与旋转记录。
- `sources/`：本轮实现快照；训练子目录另外保存原教师物理和任务依赖快照。

需要停止整个队列时，在输出根目录创建`STOP`；当前完整物理控制结束后退出并保留已有检查点。直接运行训练入口时也支持其输出目录内的`STOP`。

## 传感推理接口

```python
import numpy as np
from tendonspin.students.runtime import TouchStudentRuntime

runtime = TouchStudentRuntime('outputs/.../training/touch_best.pth')
action = runtime.step(
    joint_pos_rad=joint_angles_20,         # 顺序见检查点observation_contract
    issued_targets_rad=last_targets_16,   # 真正发给驱动的目标，单位rad
    force_magnitude_n=np.linalg.norm(fingertip_forces_5x3, axis=-1),
    tactile_valid=valid_5,
    tactile_age_s=ages_5,
    timestamp_s=acquisition_timestamp,
)
```

此接口只返回16维`[-1,1]`动作，不读取教师、不做在线学习、不发送实物命令。真实SDK仍需按已知通道顺序、单位和控制语义连接；动作到目标的增量、限幅、插值必须对应本项目原0.35rad/s、50ms控制周期。重抓、重新开始或驱动重连时调用`runtime.reset()`。压力阵列接触位置等特征留待SDK语义明确后另设版本，不在v1中伪造。
