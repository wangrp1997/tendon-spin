# 服务器交接：五指构型 C 组教师训练

## 最新状态：2026-10-15，210M 评估初始化错误后停止

**当前没有训练进程运行。** 2026-10-15 10:04，210M 训练正常完成并保存（实际210,108,416动作、1603updates，14.01%），随后冻结评估在策略加载前失败，物理步数0；监督按既定规则停止，未自动重试。原主PID2227265已退出，以下“运行中”描述均是历史记录。

错误：PhysX `setCMassLocalPose pose is not valid`，随后名义50g质量读回误差约0.05kg；根因待独立诊断，不推定为策略NaN或显存OOM。完整learner/policy哈希、有限性、配对、Adam步数2051840及52个冻结源/配置联合身份均已CPU核验通过。最后有效评估仍是200M（30秒、1238.48°），210M无有效评估。

证据：`outputs/server_c50m_deployment/c1p5b_210m_failure_audit.json` / `.md`；恢复源是 `outputs/boya_c_fresh1p5b_server_v1/actions_210000000/training/teacher_final.pth`。保留全部权重和失败记录；不得重新运行原fresh命令覆盖或重新训练。遵守执行错误停止汇报、不自动重试约束；人工诊断/恢复另行记录授权和验证结果。原引擎、抓取、算法、奖励、DR和冻结协议均未修改。


用户已授权：在服务器空闲 GPU 上测试合适的并行环境数，接入下面的 C 奖励，从零训练 **1.5B（1,500,000,000）累计环境动作**，每 10M 自动冻结评估，共 150 个节点。原 50M 是早期测试预算，用户已在 2026-10-14 明确改为参考原 Hora 正式上限。本次仅在服务器选定 GPU 1 上运行。此文件是明确要求公开的交接说明；`docs/`、`AGENTS.md`、权重和训练日志仍不发布。

### 2026-10-15 最新授权：先 push，再从210M恢复

用户已明确要求“先push再恢复”。**不再等待新的启动确认**；先确认本次恢复代码已经推送，再启动独立续训目录 `outputs/boya_c_fresh1p5b_server_recovery_sync_v1/`。本文件以下停机/待审阅描述是推送前历史；实际启动状态以新目录 `status.json`、当前训练 `result.json` 和PID为准。不得重跑旧fresh入口。

正式修复不改安装文件：新增 `tendonspin/physics/physx_property_sync.py` 和 `scripts/run_boya_with_property_sync.py`，对精确SHA固定的安装版PhysX，仅在质量/COM/环境ID的现有D2H copy完成后让CPU消费者继续；其他GPU/求解器操作保持异步。旧52个冻结源码/配置及完整learner合同原样保留，兼容入口3个新文件另存哈希与运行审计，不隐瞒运行时差异。

新增 `scripts/resume_boya_spin_progress_milestones.py`：核验旧节点和自己的210M完整learner，保存原失败记录，用新目录引用已验证的210M诊断评估并记录显式接纳，再继续220M～1500M共129段。每10M原名义30s评估、16384环境、minibatch512、seed43、C奖励、DR、抓取和原PhysX不变；执行/源/恢复/数值错误仍停报，不自动重试。

推送前验证：相关CPU测试11项通过；新生产入口在独立64环境配置副本上完成2updates/1024动作且模型更新、属性回读正确。完整只读历史/恢复预检 `outputs/boya_c_recovery_preflight_sync_v1/` 为prepared，21个节点校验完成；`--prepare-only` 不启动正式训练。独立小验证不计入正式1.5B。诊断候选三行安装文件patch继续保留但不应用，正式路线采用上述有来源审计的进程内同步。

### 2026-10-15 人工诊断结果：正式训练仍暂停

用户已明确允许开始独立诊断。已在隔离小场景复现安装版 PhysX 适配层的异步 GPU→pinned CPU 属性回读问题：原 `set_masses_index` / `set_coms_index` / 环境ID回读在 CPU 消费前缺少等待；冷缓冲实验产生0质量及与正式日志一致的无效COM姿态报错。原错误未保存实际缓冲内容，不将实验零质量当作当时直接观测。完整报告：`outputs/server_c50m_deployment/c1p5b_210m_diagnosis_report.md` / `.json`。

已生成未应用的三行候选 `outputs/server_c50m_deployment/isaaclab_physx_property_host_readback_sync_candidate.patch`，仅在现有D2H copy后等待stream。临时进程验证通过：单环境80次属性设置及冷缓冲、64环境80次属性设置和200次原DR reset均正确。52个冻结源/配置及安装文件原始哈希复核一致；**未修改安装环境、冻结代码、奖励、抓取、PPO或物理参数**。

使用进程内临时等待的独立210M冻结评估已完成：30秒未掉落，net **1345.4828°（3.7375圈）**，比200M约+8.64%，仍低于70M最佳1686.1693°。名义50g/摩擦0.5/COM0读回正确，与200M初态各项误差0。输出 `outputs/server_c50m_deployment/physx_sync_eval210m_20261015_v1/`，含同步wrapper来源；**尚未计入正式210M节点，原失败评估未覆盖**。

原64环境未同步的100次实际DR reset小样本也通过；历史训练并无逐次真实属性回读，因此不能排除或量化过去偶发错误，也不能将后期名义速度回退直接归因于此。200M有84.17%控制步达到0.5rad/s正向奖励上限；奖励分解提示速度与优化目标可能存在取舍，但未证明因果。

恢复草案 `outputs/server_c50m_deployment/c1p5b_210m_recovery_plan_draft.json` 已保存，未执行。安装适配层兼容补丁需明确审阅记录；随后只允许从自己的210M完整learner续训，使用新的continuation监督输出，先明确接纳210M独立评估，再从220M（实际220069888）继续剩余129个节点。原主入口fresh-only，不得对旧目录重跑。当前没有正式训练进程。

## 服务器接手进度：2026-10-14，已在 tmux 中恢复

用户已明确要求恢复工作，取消六档/20 分钟限制。最新偏好是**全程总耗时最短，计入每段初始化和评估；显存只须留余量，不要求接近满载**。用户已明确选择正式 **C1.5B** 上限，并在预算问题中确认其余设置保持，包括 minibatch512 和每10M自动评估；准备与验证完成后直接从零启动，无须重复确认。容量测试已结束，最终选择 **16384 环境**；正式配置与独立冻结协议已冻结。**正式任务已后台启动并完成健康首更新**，主监督 PID2227265、阶段PID2227276、训练PID2227277，启动北京时间2026-10-14 16:33:13。不得重复启动。输出 `outputs/boya_c_fresh1p5b_server_v1/`，启动记录 `outputs/server_c50m_deployment/c1p5b_formal_launch.json`；恢复时先读正式status/stage/training result并核对PID，不改冻结源码/配置/协议。首更新及policy/learner配对验证通过：零动作/updates、空Adam，131072动作/update1/Adam1280步；模型有限且变化。健康证据 `outputs/server_c50m_deployment/c1p5b_first_update_verified.json`。本次观测累计1179648动作/9updates，预热后早期完整速度5082动作/s；启动报告 `outputs/server_c50m_deployment/c1p5b_startup_report.json`，ETA约4.4–5.3天。正式训练与150节点评估仍在运行，并未完成1.5B。

### 已完成部署与验证

- 仓库 `/home/wangrenpeng/tendon-spin` 已按用户要求 pull，HEAD `e652db81449245588273470b4d9f725fb4e7e169`。服务器 C 代码已整理为独立 Git 版本；提交与推送状态以 Git 和 `outputs/server_c50m_deployment/` 的发布记录为准。保留工作区与全部训练输出，勿 reset/clean。旧 B 配置、抓取、物理引擎和 Hora PPO 没有改写。
- 缓存已取得并校验：SHA256 `6cb72539fc3c40d1e75a49983019c9b4be679a37bd5d8dcef406c06696aaf993`，164 状态，九组 `[3,1,5,10,12,30,20,41,42]`。无需再寻找或筛选缓存。
- GPU 1 为 RTX PRO 6000 Blackwell Server Edition，97,887 MiB，UUID `GPU-a3f9cc58-fe6f-96d3-ab1a-e1dc33b09e7a`；驱动 580.159.03、CUDA 13.0。绑定 `CUDA_VISIBLE_DEVICES=1`，Isaac/PPO 逻辑 cuda:0 实际均用 GPU 1。GPU 0 有他人任务，不干扰。
- 独立 `.venv` 已部署：Python 3.12.14、torch 2.11.0+cu130、Isaac Sim 6.1.0.0、Isaac Lab 3.0.0rc1、Warp 1.16.0、NumPy 2.3.1、SciPy 1.17.0、packaging 26.0、newton-usd-schemas 0.4.1。完整库存 `outputs/server_c50m_deployment/installed_environment.json`。未改共享 Sim 或系统 Python。
- Isaac Lab 官方安装 overrides 固定 Newton 1.5.2，与 Sim core 的 Newton 1.5.0 元数据存在一项已报告且按官方覆盖解决的依赖差异；**实际仿真后端经运行时类验证为 isaaclab_physx**，未选择 Newton。官方 wheel 与 overrides 保存在 `runtime/` 和 deployment 目录。
- 本机 import probe 完成，结构检查通过：22 joints 和 mimic 保留。USD `outputs/isaac_boya_import/boya/boya.usda`，SHA256 `4e4914a1aade2661cf33b13ed75bcbd6e77f1ff4651b2c61d1ca974f49f7cfa7`；导入身份 `docs/data/isaac_boya_import_phases.json`。
- C 原生 trainer/eval 已接入同一 `SpinProgressBoyaEnv.control_reward()`。控制开始 begin、每物理步 advance、结束 compute_reward；C 正向封顶、反向不截断、失败扣 16 一次。B 共享边界有行为一致性测试，原物理/DR/reset/PPO 不变。
- CPU 奖励/原生控制/初始化/resume/预算测试通过。真实 GPU 短训 64 环境、4 updates/2048 动作，首更新有限且参数变化；缓存、名义初态、随机化属性读回一致。独立姿态重算与原生 C 旋转项最大误差 `4.94e-8`。短冻结评估在 0.25s 掉落，属于未训练策略验证，不代表效果。自身完整 learner 恢复到第 5 update 成功，成对 policy/full learner 权重一致。
- 验证汇总 `outputs/server_c50m_deployment/short_validation_passed.json`；短训/评估/resume 在 `outputs/boya_c_short_server_v1/`，动作不计正式 1.5B。

### 容量与全程效率测试

原测试 `outputs/boya_c_capacity_server_v1/` 和扩展测试 `outputs/boya_c_capacity_extended_server_v1/` 都保留：

| 环境数 | 完整训练吞吐，动作/s | 峰值显存，MiB | 每段初始化及退出，秒 |
| --- | ---: | ---: | ---: |
| 1024 | 741 | 4001 | 36 |
| 2048 | 1370 | 4421 | 67 |
| 4096 | 2309 | 5173 | 128 |
| 8192 | 3843 | 6841 | 257 |
| 16384 | 5169 | 10021 | 532 |
| 24576 | 无效：材料数量上限 | 11487 | 未完成训练 |
| 32768 | 无效：材料数量上限 | — | 未完成训练 |

32768 档在初始化时多次报 `PhysX error: PxPhysics::createMaterial: limit of 64K materials reached`，随后 SIGSEGV（退出 -11），未完成 PPO update。**不是显存 OOM；已先向用户汇报，原测试及更大队列已停止。** 不重试该档，不修改引擎/材料/抓取/DR/PPO 去绕过。所有较小成功档日志无此错误。

补测低于失败档的 **24576 已结束**，同样触发材料上限并退出 -11，未完成 PPO 更新。补充测试已选择 **16384**，复用成功五档的全程效率测量：

- 输出 `outputs/boya_c_capacity_total_time_server_v1/benchmark.json`，启动记录 `outputs/server_c50m_deployment/capacity_total_time_launch.json`，日志 `capacity_total_time.log`；phase=completed，旧 PID 2200288 已退出；不恢复容量测试。
- 每档 fresh 2 updates 预热+8 updates测量，minibatch512/horizon8/mini-epochs5保持。记录采样、PPO、日志/保存开销及初始化/退出；没有资源 watchdog/cgroup/自动内存停机。
- 固定测试协议 `docs/experiments/2026-10-14-server-c50m/CAPACITY_PROTOCOL.md` 保留不改；其中早期显存/50M选择偏好已由本节最新用户授权覆盖。
- 补测命令的 `--target-actions 50000000` 是预算确认前的比较预览，**不能当正式预算**。已以1.5B/150次初始化重算并记录 `outputs/server_c50m_deployment/final_c1p5b_selection.json`；16384训练+初始化约102.8小时，另加评估，初步 ETA4.3–5.3天。
- 最终按 `实际向上取整动作/实测吞吐 + 150×实测每段初始化退出` 比较。冻结单环境30s评估是各环境数共有的额外开销，加入 ETA；不能只选最高稳态 FPS。预留4894MiB显存。
- `benchmark_boya_spin_progress.py` 已增加同运行时/源/DR/cache身份验证后的测量复用、动态档数/时限、实时资源记录、总用时选择及64K材料错误检测。该错误会判为容量失败并停止更大档，不计有效吞吐。

### 正式训练运行与后续接手

1. 容量测试已完成，原测试和临时渲染进程已退出；不要重复启动或恢复测试权重。
2. C配置已固定16384/1500000000，独立冻结 `docs/experiments/2026-10-14-server-c1p5b/PROTOCOL.md` 已生成；正式训练运行中不改源码、C配置或该协议；旧B保留。
3. 已后台启动 fresh seed43 `scripts/run_boya_spin_progress_milestones.py --num-envs 16384 --out outputs/boya_c_fresh1p5b_server_v1 --scene-config assets/config/boya_hora_randomized_c_scene.json --randomization-config assets/config/boya_hora_randomized_c.json --protocol docs/experiments/2026-10-14-server-c1p5b/PROTOCOL.md`，绑定 GPU 1、可靠后台运行。禁止用测试/历史权重初始化。
4. 首更新有限/模型变化/从零身份与完整 learner/policy 成对保存已确认；启动报告记录环境数、实测吞吐/显存、PID/输出、按 batch=num_envs×8 向上取整的实际 1.5B 动作与 update 数、ETA。每 10M 自动冻结名义原状态 30s 或失败即停评估；普通掉落继续，执行/源/resume/数值错误停止，1.5B 后停止。

新增 C 文件为 `isaac_spin_progress.py`、C config/scene、`train_boya_spin_progress.py`、`evaluate_boya_spin_progress.py`、`run_boya_spin_progress_stage.py`、`run_boya_spin_progress_milestones.py`、`spin_progress_milestone_checks.py`、`benchmark_boya_spin_progress.py` 及两份必要测试。所有权重/完整日志仍在 ignored `outputs/`，保留原历史。工具普通沙箱目前报 `bwrap: Creating new namespace failed`，经自动审批使用沙箱外命令；这是执行环境问题，不是更换仿真路线的理由。

### 部署审计与最新节点评估

用户要求发布服务器代码版本，范围为C代码、配置、必要测试及本交接；164缓存此前已在Git。`.venv/`、`runtime/`、本地协议、权重、完整日志、审计输出和demo视频不纳入本次提交。

对照交接与原B物理/DR/PPO审计未发现额外部署遗漏：52个冻结源/配置哈希未变，实际生成USD保留28碰撞排除；30M的30100物理样本全部有限，四个distal独立耦合误差最大1.51e-5rad，请求/转发力矩差0。C奖励独立重算最大误差3.48e-7，失败-16一次。10/20/30M完整learner与policy及normalizers一致，Adam计数/学习率和自身恢复链验证通过。证据 `outputs/server_c50m_deployment/c1p5b_deployment_audit.md`。

原有假设继续保留：电机armature尚未实测，采用A=4*D_effective*dt；质量reset不同比重算名义50g转动惯量；冻结名义评估关闭物理DR，但控制后仍有固定seed43的±0.02rad关节观测噪声。不要将它描述为零输入噪声评估或实机参数已标定。

30M名义冻结评估保持15.05秒，signed net+455.37°，随后工作区掉落；未达到完整30秒。正式任务已进入30→40M阶段，主监督PID2227265不变。接手时先读正式status及current_run的training/result，不根据本交接的历史动作数判断当前进度，不重复启动。

### 临时部署视频

已生成 `outputs/boya_c_deployment_demo_tmp/rendered/isaac_native.mp4`，4秒/20fps，斜侧+俯视，RTX原生渲染短验证真实轨迹；2048动作策略在0.25s掉落，用慢放和定格查看部署构型，不代表正式训练效果。`render.json` 记录完成状态、来源哈希及零新增物理/训练动作；已向用户提供文件链接。看完可删整个临时目录，保留原短验证记录。

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
- 服务器接手前的历史验证：6项CPU解析检查和4条冻结轨迹重算；当时未做新训练或新物理执行。30M 同一完整有效 control 窗口的平均旋转分由 B 的 +0.226640 改为 C 的 −0.142165，与该窗口净反转 −3.821618° 一致；仅验证计算，尚未证明训练改善。
- **C 已在服务器接入并完成短验证。**原接入要求如下：控制开始时 `begin()`，每个物理步 `advance()`，控制结束调用 `compute_reward()`。它已经包含失败成本，勿重复扣 16。保留原 reset/timeout/数值与物理终止逻辑。
- 将 C 模块及配置加入 source hash、checkpoint/resume contract 和冻结评估；所有计划/日志/权重明确写 `arm=C` 与完整 reward profile。旧 B 脚本有硬编码 B 和 1024，不能直接运行它并声称训练了 C。

## 已筛选抓取与域随机化：复用，不重新搜索

沿用已完成的五指构型、九尺寸 **164 个** Isaac 筛选缓存和从第一轮 reset 开始的域随机化；不回退原四指抓取。五指参与操作并允许接触变化。现有准入要求稳定和至少两指组接触，**不是每个缓存同时五指接触的认证**，不得把它描述成 164 个全五指接触姿态；本交接不新增这种筛选门槛。

这份约 30KB 的抓取缓存已按用户授权加入 Git，服务器 `git pull` 后即可使用，无需另行复制。其他训练输出和权重仍忽略。缓存路径和校验值：

```text
outputs/boya_hora_scale_caches_v1/grasp_cache.npz
SHA256: 6cb72539fc3c40d1e75a49983019c9b4be679a37bd5d8dcef406c06696aaf993
各尺寸状态数: [3, 1, 5, 10, 12, 30, 20, 41, 42]
```

配置来源：[boya_hora_randomized_b.json](assets/config/boya_hora_randomized_b.json)、[boya_hora_randomized_scene.json](assets/config/boya_hora_randomized_scene.json)。新建 C 配置副本，仅替换奖励/组名/选定环境数；保持缓存 SHA、名义姿态和其余参数。

随机化已实现：尺寸中心 0.70:0.02:0.86、每环境固定 ±0.025 尺寸抖动；启动时摩擦 0.3–3、局部 COM 每轴 ±0.01m；reset 质量 0.01–0.25kg、P 比率 2.9/3–3.1/3、D 比率 0.9–1.1；关节观测噪声 ±0.02rad；每控制步概率 0.25 的 Gaussian×mass×2 外力及原衰减。`replicate_physics=False`；使用对应尺寸组缓存；质量变化时保留各尺寸名义惯量，与已有分支一致。来源是 [Hora v0.0.1 正式教师脚本](https://github.com/HaozhiQi/hora/blob/v0.0.1/scripts/train_s1.sh)，它实际启用外力，不能照 YAML 的 force=0。

新 checkout 先运行 `python scripts/prepare_local_runtime.py`，再用 `PYTHONPATH=. python scripts/probe_boya_import.py --out docs/data/isaac_boya_import_phases.json` 在本机生成 USD/导入身份。旧入口依赖本地 `docs` 协议；C 入口使用本交接或新本地协议作 `--protocol`，不能依赖未提交的历史实验文档。

## 空闲 GPU 上选择环境数，再正式训练

1. 用 `nvidia-smi` 确认空闲卡，选择可用且最快的一张，以 `CUDA_VISIBLE_DEVICES` 绑定；Isaac 和 PPO 使用同一卡。当前链是单 GPU；不启动重复正式任务充数，不干扰其他任务。
2. **1024 是本地选择，不是算法上限。**在同一 C 配置下，从 1024 起倍增环境数；原最多 6 档/20 分钟限制已按用户 2026-10-14 授权取消，可扩大档数和延长测试。每档独立短试，预热 2 updates，再测 8 updates；记录完整物理采样+PPO 的 actions/s、更新时间、显存峰值和主机内存，按用户最新偏好，选完整预算预计总耗时最短且留有保存/评估余量的档，计入每10M重建/评估。
3. 容量测试的 OOM 标为失败，该档及更大档停止，选已完成的最佳档；首档都失败或出现非容量执行/源/数值错误则汇报，不擅改 dt、碰撞、任务或算法。测试权重和动作不计正式 1.5B；测试不加资源 watchdog/cgroup/自动内存停机机制。
4. 将选定 `--num-envs` 贯穿 C trainer、stage、milestones、配置与 resume 身份；固定 horizon 8/minibatch 512/mini-epochs 5，不为提高吞吐修改奖励/控制频率/PPO 系数。按实际 `batch = num_envs × 8` 向上取整各 10M 节点，报告实际总动作与 update 数，不能继续硬写旧50M/1024时的6104 updates。
5. 确认 C 在原生训练/评估中生效、初态/cache/property 与配置一致、首更新有限且模型实际更新后，启动后台 **fresh seed43 C1.5B**：网络、normalizers、Adam、adaptive LR 和动作计数从零；不初始化自 B20M/B40M 或容量测试权重。用户已授权本交接所述训练，无需重复问启动许可。

## 保存、评估和结束

正式预算是所有环境累计 **1,500,000,000 控制动作**，不是每个环境1.5B，也不是update数。原Hora `third_party/hora/configs/train/AllegroHandHora.yaml` 的 `max_agent_steps` 为1.5B；原minibatch32768与本交接512不同，当前仅参考动作预算，不称完全复现。每10M（10/20/.../1500M，共150个节点）做一次冻结名义初态 **30s 或物理失败即停**的原状态连续评估，沿用现有离线分析。只在段间恢复自己的完整 C learner；仿真从匹配缓存重置，记录此事实。环境数正式启动后固定。

训练使用 headless、summary logs、TensorBoard、周期 policy/完整 learner 成对可恢复 checkpoint；仅评估保存逐物理步轨迹。保留所有旧 B/四指/五指权重与历史，不覆盖。评估报告有效时长、signed net/peak/backward、末 10s 净角、最大漂移/倾斜及真实停止原因；完整 30s 且净角和末段为正只是名义进展信号，不证明泛化、无限旋转或实机成功。

普通评估掉落按既定预算继续；正式训练遇执行/源/resume/数值错误则停并汇报，不自动重试或改路线。1.5B完成后停，不自动续训。启动后交付选定环境数、实测吞吐、进程/输出位置、实际预算和 ETA，结束后交付节点评估表；代码/配置、本交接及上述小型抓取缓存可 push，权重/完整日志和其他输出留在 ignored 目录。
