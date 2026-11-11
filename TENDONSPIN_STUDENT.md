# TendonSpin C1260M 学生蒸馏

本项目自己的30帧本体/触觉历史学生。教师固定为已选定的C1260M；与旧20帧部署策略、botyard的RGB-D学生使用不同的检查点接口。

## 2026-11-03首轮结果

本项目首轮训练及独立闭环评测已于北京时间 **2026-11-03 18:34:59** 全部完成。训练2000控制步、2,016,000次仿真交互（含验证环境），两个学生各8896次更新。固定14,400条验证样本选择best；测试使用独立种子2026110203，27组新ID物理条件，三个冻结控制器各运行完整600控制步，每组最多30秒。

| 策略 | 平均净旋转 | P10净旋转 | 保持30秒 | 教师平均旋转比例 |
| --- | ---: | ---: | ---: | ---: |
| C1260M教师 | 1026.46° | 494.84° | 24/27 | 100% |
| 触觉学生 | 986.73° | 405.20° | 22/27 | 96.13% |
| 纯本体感觉学生 | 1016.39° | 511.11° | 22/27 | 99.02% |

两个学生均达到预先固定的仿真门槛（平均净旋转≥教师90%，保持率下降≤10个百分点）。本轮纯本体学生的平均旋转比触觉学生高3.01%，两者保持率均为22/27；这组有限案例未显示触觉增益，也不足以证明总体或实物性能优劣。按此轮结果，后续接入以`proprio_best.pth`为优先基线，`touch_best.pth`保留作对照；测试未用于重新拟合或挑选新的训练检查点。

- 正式输出：`outputs/boya_touch_student_20261102_v1/`；最终成绩：`evaluation/comparison.json`。
- 纯本体权重：`training/proprio_best.pth`，SHA-256 `d9cb5593e383c4b0f34986d0288d647a599390d96ea91b33681794efd17cdbe2`。
- 触觉权重：`training/touch_best.pth`，SHA-256 `4e1c9bfbecd11857153a7c4a51275996036ce062679b94844099e1de6259c490`。
- 复核：`final_verification.json`；两份权重、固定案例、三份轨迹哈希均一致，从物理步旋转增量重算成绩通过，外力和编码器噪声逐项配对一致，10个队列源及36个训练依赖保持原哈希。
- 主监督3709537、训练3709647、评测1128790均已退出；宿主机进程及GPU列表已核对，本轮显存已释放，其他项目任务保持运行。无需重启或重复评测本轮。
- 实物尚未验证；使用原抓取缓存和名义1∶1耦合，不增加J1/J2专项测试。SDK采集/驱动仍需按现有传感推理接口连接，本轮未发送实物命令。权重及原始轨迹存于服务器输出目录，Git记录代码、说明和成绩摘要。

## 输入与网络

学生只接收传感历史，每帧57维：16路主动关节角、16路已发目标、4路独立读取的被动J1，以及五指接触、力幅值、触觉有效性、时效和真实/填充帧标记。角度按已知限位处理，目标保留弧度。三轴力在SDK边界取范数；使用固定量程和`log1p`保留力大小，不做逐帧单位向量归一化。第一版不依赖相机，也不输入物体位置、质量或摩擦真值。

教师原96维短观测、8维latent和16维动作头保持原尺寸。57维学生历史通过独立TCN预测latent，再进入冻结的原动作头。教师观测归一化器随学生保存，训练与部署都保持冻结。被动J1增加观测，不增加独立动作通道；仿真沿用当前1∶1耦合，不安排专项比例测试。

仿真触觉来自各指**distal刚体**与物体接触的法向力加摩擦力合力。这是理想刚体代理，尚未限制到实物传感器的局部覆盖区域，也不模拟压力阵列。当前力阈值、饱和与滤波参数属于显式工程默认值，不是实测传感器标定结果；均保存在`observation_contract.json`和检查点内。

## 左手迁移

环境通过`hand_side`选择`HandProfile`，任务、奖励、DR、学生网络和评测逻辑共用。右手继续使用原USD；左手默认读取从当前右手V3.2逐项镜像生成的`assets/hand/mirror_right_v3_2/usd/left_dexhand/left_dexhand.usda`。生成器同时镜像24个STL、所有刚体/关节局部位姿、关节轴和惯量，并保留右手质量、阻尼、增益和力矩范围；来源与逐文件哈希见`assets/hand/mirror_right_v3_2/manifest.json`。厂商V3.2左手保留为参考，但它不是严格镜像，因此不作为默认运行资产。也可用`TENDONSPIN_LEFT_HAND_USD`或`--left-asset`覆盖路径。

镜像STL和约46MB的Isaac USD属于本机生成资产，不进入Git；仓库保留生成器、小体积URDF/MJCF与哈希清单。首次使用时执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_mirrored_right_hand.py
PYTHONPATH=. .venv/bin/python -u scripts/import_mirrored_left_hand.py
```

左手profile显式处理`left_`关节/刚体前缀、镜像世界Y轴和镜像手掌根姿态；训练和评测记录实际资产、USD结构、profiles及source SHA-256。左手直接继承右手TendonSpin契约的全部控制参数。严格镜像保持右手关节坐标和初始q/commands，因此脚本不再重新拟合抓取姿态，只把物体和手掌放入镜像场景。原C1260M右手没有在左手重训；结果属于冻结策略跨手评测，仍不构成实物成功声明。

迁移前运行静态预检。脚本核对USD主动关节/限位、远端刚体、接触排除和J1-J2联动。严格镜像分支原样保留右手164个初始抓取状态，避免把策略推出训练分布；这不替代Isaac动力学或硬件验收。

```bash
PYTHONPATH=. .venv/bin/python -u scripts/prepare_boya_left_transfer.py \
  --out outputs/boya_left_transfer_prep_20261110_v5
```

严格镜像资产预检通过；24个刚体、22个转动关节、右手控制契约和28项接触排除一致，164个初始抓取及九个尺寸组全部保留。正式输出在`outputs/boya_left_transfer_prep_20261110_v5/`。旧的厂商左手、`v1/v2`重新拟合缓存和错误根变换输出均不能用于当前目标。

```bash
PYTHONPATH=. .venv/bin/python -u scripts/evaluate_boya_touch_student.py \
  --training outputs/boya_touch_student_20261102_v1/training \
  --out outputs/boya_touch_student_left_transfer_strict_mirror_20261110_v1 \
  --hand-side left --num-envs 27 \
  --scene-config outputs/boya_left_transfer_prep_20261110_v5/scene_configuration_left_candidate.json
```

严格镜像教师与纯本体学生配对评测已于2026-11-10完成，正式输出在`outputs/boya_touch_student_left_transfer_strict_mirror_20261110_v1/`。27个案例使用相同初态和物理条件：

| 控制器 | 平均净旋转 | P10净旋转 | 平均有效时长 | 完整保持 |
| --- | ---: | ---: | ---: | ---: |
| C1260M教师 | -933.55° | -1252.80° | 28.01 s | 23/27 |
| 纯本体学生 | -974.04° | -1248.50° | 27.06 s | 22/27 |

镜像会翻转旋转赝矢量的符号，因此左手有符号净旋转为负是正确结果。按手侧归一化后，教师幅值为`933.55°`，纯本体学生为`974.04°`，学生/教师比为`1.043`；学生保持率22/27，仅比教师下降3.70个百分点。按预先声明的`≥90%`旋转和`≤10`个百分点保持率门槛，纯本体学生通过。

原始右手教师同一协议的结果为幅值`1026.46°`、保持24/27。严格镜像左手教师达到右手幅值的`90.95%`、保持23/27。学生初态最大误差为`1.19e-7`，配对刺激核验通过，评测进程正常完成。该结果支持冻结策略在严格镜像仿真中复用；厂商左手和实物仍未验证。

纯本体学生部署时不再要求触觉占位输入：

```python
action = proprio_runtime.step(
    joint_pos_rad=joint_angles_20,
    issued_targets_rad=last_targets_16,
    timestamp_s=acquisition_timestamp,
)
```

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
