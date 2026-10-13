# TendonSpin

伯牙腱驱灵巧手的手内物体旋转项目。使用 [Hora](https://github.com/HaozhiQi/hora) PPO 训练教师策略，
研究触觉与动作历史驱动的学生策略；参考 [Hora](https://haozhi.io/hora)、
[AnyRotate](https://maxyang27896.github.io/anyrotate/) 和
[Sharpa RL Lab](https://github.com/sharpa-robotics/sharpa-rl-lab)。

代码和模型资源独立保存在本仓库。仿真使用现有 Isaac Sim/Lab 环境，
同时保留原始 MuJoCo 模型与工具。

## 目录

| 目录 | 内容 |
| --- | --- |
| `tendonspin/` | 手模型接口、仿真环境、策略和评价指标 |
| `scripts/` | 训练、冻结策略评估和视频渲染入口 |
| `assets/` | CAD、原始抓取状态和可选物体配置 |
| `third_party/` | 参考算法源码及许可证 |
| `assets/config/` | 公开的模型运行配置 |
| `outputs/` | 本地权重、TensorBoard、轨迹和视频，不提交到 Git |

## 使用

Isaac 路径使用现有 Isaac Sim 6.1 / Isaac Lab 3 环境。先准备本地模型配置，
再将 URDF 转换成该环境的 USD 资源；生成的运行记录和 USD 留在本地：

```bash
python scripts/prepare_local_runtime.py
PYTHONPATH=. python scripts/probe_boya_import.py --out docs/data/isaac_boya_import_phases.json
```

训练和评估入口的参数：

```bash
PYTHONPATH=. python scripts/train_boya_hora.py --help
PYTHONPATH=. python scripts/evaluate_boya_hora.py --help
PYTHONPATH=. python scripts/evaluate_boya_hora_initial_state.py --help
```

物体默认使用原始 40×32 毫米圆柱，也可选择命名配置或指定圆柱、球、长方体的尺寸。
训练可通过 `--scene-config` 选择物体与名义抓取配置、`--cache` 指定对应初态缓存。
五指构型配置位于
[isaac_learning_scene.json](assets/grasp/fivefinger_uniform48p8_side2deg_v1/isaac_learning_scene.json)。
抓取配置保持双腕，开放16路手指动作，四个远端mimic被动耦合。

物理随机化教师使用独立入口 `run_boya_hora_randomized_milestones.py`，参数配置见
[boya_hora_randomized_b.json](assets/config/boya_hora_randomized_b.json)。该入口按Hora正式教师脚本
的尺寸、质量、摩擦、质心、PD和外力机制运行，使用各尺寸对应的抓取缓存；B奖励保留项目的姿态差分和掉落成本适配。

冻结评估通过 `--checkpoint` 和匹配的 `--source-record` 载入策略及运行配置，
配套 `--cache`；`evaluate_boya_hora_initial_state.py` 还支持 `--cache-index`。
权重与缓存需自行提供。训练的 `--protocol` 接收项目内的运行配置文件，
用于记录来源身份；历史实验专用的启动脚本需要对应的本地实验资料。

参考代码版本与许可证见[THIRD_PARTY.md](THIRD_PARTY.md)及
[参考项目](references/README.md)。当前训练接口是特权教师；历史适应和触觉学生仍在开发。
