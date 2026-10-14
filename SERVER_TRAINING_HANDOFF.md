# 服务器交接：五指构型 C 组教师训练

用户已授权：在服务器空闲 GPU 上测试合适的并行环境数，接入下面的 C 奖励，从零训练 **50M 环境动作**，每 10M 自动冻结评估。本次不在本地启动新训练。此文件是明确要求公开的交接说明；`docs/`、`AGENTS.md`、权重和训练日志仍不发布。

## 环境与训练配置

| 项目 | 当前本地配置，服务器应优先保持 |
| --- | --- |
| 仿真版本 | Isaac Sim **6.1.0.0**；Isaac Lab **3.0.0rc1** |
| Python / PyTorch | **3.12.14 / 2.11.0** |
| 后端 / 求解器 | `isaaclab_physx` / GPU PhysX，`original_tgs16_4`；不是 `isaaclab_ov` |
| 碰撞 | 现有 CAD convex、自碰撞和 28 项排除；contact offset 0.1mm、CCD 关闭；无支撑地面 |
| 时间 | physics dt 0.0005s；100 物理步/控制；20Hz |
| 执行器 | 五指全部开放：TH4 + FF3 + MF3 + RF3 + LF3 = **16 动作**；双腕固定、四个 distal mimic 被动；现有外部 PD |
| 观测 / 算法 | 特权教师：96 维观测、9 维真实物理属性；Hora PPO、horizon 8、mini-epochs 5、minibatch 512、seed 43 |
| 物体 / 构型 | 名义白色圆柱 **48.77355 × 32mm、50g**；五指构型 `fivefinger_uniform48p8_side2deg_v1`，长指侧摆约 ±2°；原位置与方向不变 |
| 任务终止 | `boya_workspace`：既有工作区高度/侧向边界及确认；数值检查保持；接触少仅作诊断 |

使用现有兼容运行环境，不改共享环境、不自行切换引擎。记录服务器实际 GPU/驱动/CUDA、软件版本及代码 SHA；版本不兼容时先汇报具体问题。

## B 与 C 的区别及已完成工作

代码交接基点为 `a9f2d00`。**B** = `hora_pose_delta_drop16`：姿态差分速度投影到固定世界轴，旋转项双向截断 ±0.5rad/s，再加工作区失败成本 16。

**C** = `boya_object_spin_positive_cap_drop16`：每个物理步从 XYZW 姿态差分计算旋转增量，投影到相邻物体自身 Z 轴的负向平均轴，累加后除以控制周期；正向仍封顶 **0.5rad/s**，反向不截断。保留原线速度/手关节姿态/力矩/功耗系数 **−0.3/−0.3/−0.1/−2**，保留 terminal codes **9/10** 的成本 **16**，普通 timeout 不罚。关节姿态项不是物体倾斜惩罚。C 是 Boya 新奖励分支，不称原版 Hora 奖励。

- 实现：[spin_progress_reward.py](tendonspin/rl/spin_progress_reward.py)；配置：[boya_spin_progress_reward.json](assets/config/boya_spin_progress_reward.json)。
- 已通过 6 项 CPU 解析检查和 4 条历史冻结轨迹重算，未做新训练或新物理执行。30M 同一完整有效 control 窗口的平均旋转分由 B 的 +0.226640 改为 C 的 −0.142165，与该窗口净反转 −3.821618° 一致；仅验证计算，尚未证明训练改善。
- **C 尚未接入训练/评估入口。**服务器先复用现有随机化训练链建立独立 C 入口：控制开始时 `begin()`，每个物理步 `advance()`，控制结束调用 `compute_reward()`。它已经包含失败成本，勿重复扣 16。保留原 reset/timeout/数值与物理终止逻辑。
- 将 C 模块及配置加入 source hash、checkpoint/resume contract 和冻结评估；所有计划/日志/权重明确写 `arm=C` 与完整 reward profile。旧 B 脚本有硬编码 B 和 1024，不能直接运行它并声称训练了 C。

## 已筛选抓取与域随机化：复用，不重新搜索

沿用已完成的五指构型、九尺寸 **164 个** Isaac 筛选缓存和从第一轮 reset 开始的域随机化；不回退原四指抓取。五指参与操作并允许接触变化。现有准入要求稳定和至少两指组接触，**不是每个缓存同时五指接触的认证**，不得把它描述成 164 个全五指接触姿态；本交接不新增这种筛选门槛。

缓存需另行从本地复制到服务器同一仓库相对位置，Git 不包含它：

```text
outputs/boya_hora_scale_caches_v1/grasp_cache.npz
SHA256: 6cb72539fc3c40d1e75a49983019c9b4be679a37bd5d8dcef406c06696aaf993
各尺寸状态数: [3, 1, 5, 10, 12, 30, 20, 41, 42]
```

配置来源：[boya_hora_randomized_b.json](assets/config/boya_hora_randomized_b.json)、[boya_hora_randomized_scene.json](assets/config/boya_hora_randomized_scene.json)。新建 C 配置副本，仅替换奖励/组名/选定环境数；保持缓存 SHA、名义姿态和其余参数。

随机化已实现：尺寸中心 0.70:0.02:0.86、每环境固定 ±0.025 尺寸抖动；启动时摩擦 0.3–3、局部 COM 每轴 ±0.01m；reset 质量 0.01–0.25kg、P 比率 2.9/3–3.1/3、D 比率 0.9–1.1；关节观测噪声 ±0.02rad；每控制步概率 0.25 的 Gaussian×mass×2 外力及原衰减。`replicate_physics=False`；使用对应尺寸组缓存；质量变化时保留各尺寸名义惯量，与已有分支一致。来源是 [Hora v0.0.1 正式教师脚本](https://github.com/HaozhiQi/hora/blob/v0.0.1/scripts/train_s1.sh)，它实际启用外力，不能照 YAML 的 force=0。

新 checkout 先运行 `python scripts/prepare_local_runtime.py`，再用 `PYTHONPATH=. python scripts/probe_boya_import.py --out docs/data/isaac_boya_import_phases.json` 在本机生成 USD/导入身份。旧入口依赖本地 `docs` 协议；C 入口使用本交接或新本地协议作 `--protocol`，不能依赖未提交的历史实验文档。

## 空闲 GPU 上选择环境数，再正式训练

1. 用 `nvidia-smi` 确认空闲卡，选择可用且最快的一张，以 `CUDA_VISIBLE_DEVICES` 绑定；Isaac 和 PPO 使用同一卡。当前链是单 GPU；不启动重复 50M 任务充数，不干扰其他任务。
2. **1024 是本地选择，不是算法上限。**在同一 C 配置下，从 1024 起倍增环境数；最多 6 档、总计最多 20 分钟。每档独立短试，预热 2 updates，再测 8 updates；记录完整物理采样+PPO 的 actions/s、更新时间、显存峰值和主机内存，选吞吐最高且留有保存/评估余量的档，而不是单选环境最多或 GPU 利用率最高的档。
3. 容量测试的 OOM 标为失败，该档及更大档停止，选已完成的最佳档；首档都失败或出现非容量执行/源/数值错误则汇报，不擅改 dt、碰撞、任务或算法。测试权重和动作不计正式 50M；测试不加资源 watchdog/cgroup/自动内存停机机制。
4. 将选定 `--num-envs` 贯穿 C trainer、stage、milestones、配置与 resume 身份；固定 horizon 8/minibatch 512/mini-epochs 5，不为提高吞吐修改奖励/控制频率/PPO 系数。按实际 `batch = num_envs × 8` 向上取整各 10M 节点，报告实际总动作与 update 数，不能继续硬写 1024 时的 6104 updates。
5. 确认 C 在原生训练/评估中生效、初态/cache/property 与配置一致、首更新有限且模型实际更新后，启动后台 **fresh seed43 C50M**：网络、normalizers、Adam、adaptive LR 和动作计数从零；不初始化自 B20M/B40M 或容量测试权重。用户已授权本交接所述训练，无需重复问启动许可。

## 保存、评估和结束

正式预算是所有环境累计 **50,000,000 控制动作**，不是每个环境 50M，也不是 update 数。每 10/20/30/40/50M 做一次冻结名义初态 **30s 或物理失败即停**的原状态连续评估，沿用现有离线分析。只在段间恢复自己的完整 C learner；仿真从匹配缓存重置，记录此事实。环境数正式启动后固定。

训练使用 headless、summary logs、TensorBoard、周期 policy/完整 learner 成对可恢复 checkpoint；仅评估保存逐物理步轨迹。保留所有旧 B/四指/五指权重与历史，不覆盖。评估报告有效时长、signed net/peak/backward、末 10s 净角、最大漂移/倾斜及真实停止原因；完整 30s 且净角和末段为正只是名义进展信号，不证明泛化、无限旋转或实机成功。

普通评估掉落按既定预算继续；正式训练遇执行/源/resume/数值错误则停并汇报，不自动重试或改路线。50M 完成后停，不自动续训。启动后交付选定环境数、实测吞吐、进程/输出位置、实际预算和 ETA，结束后交付节点评估表；代码/配置及本交接可 push，缓存/权重/完整日志留在 ignored 目录。
