# 五指B40M冻结教师初态与时窗验证

授权：用户2026-10-13要求把后续计划写成逐项勾选文档，并先启动第1项。
复用2026-10-12新抓取B组的冻结策略、物理、评分和离线分析；新增评估入口只支持明确指定缓存初态，原训练/名义评估代码和权重不改。

## 冻结身份

- 权重：outputs/boya_uniform48p8_b50m_v1/actions_040000000/training/teacher_final.pth。
- SHA256：c400562ff3d04497b451aa6594c6fa09c4268f8bc2c2203d312ce281ac0400e6；累计40,001,536动作/4883更新。
- 圆柱：48.7735468437074mm直径×32mm高度、50g；full gravity；fivefinger_uniform48p8_side2deg_v1场景。原四指B40M保留，不在本阶段重新评估。
- 缓存：outputs/boya_uniform48p8_isaac_transfer_v1/grasp_cache.npz，24状态，SHA25625f790e6080a55db19654128bebf97a78545b3bea189ea9bbdf889d1b93d1c52。
- 从训练记录恢复同一固定canonical中心/轴、尺寸和姿态；缓存编号仅替换本次episode的初始q、object pose和commands，关节/物体速度按训练reset规则置零。原名义初态120s要求与原40M评估initial_state.npz精确匹配。
- 16路TH4+FF3+MF3+RF3+LF3位置目标增量，双腕保持，四个远端mimic被动；不锁小指。clipped外部PD/CAD凸碰撞、自碰撞/28排除项不变。
- 特权教师：96本体输入+9privileged属性/状态输入，8维latent；自己的观测normalizer，eval模式、无优化/归一化更新；不是实机可部署学生。
- IsaacSim6.1.0.0/Lab3.0.0rc1/GPUPhysX original_tgs16_4，摩擦0.5/contact offset0.1mm/CCD off；dt0.0005s、100physics/control、20Hz；headless单环境，无视频额外渲染。
- B奖励hora_pose_delta_drop16身份保留，固定canonical轴、不加倾斜成本；workspace高度/侧向边界及两次20Hz确认、数值检查不变。接触数与旧strict规则只作诊断，不设五指同时接触门槛。

## 固定执行预算

顺序为cache0、cache7、cache15、cache23各一次30s-or-first-physical-failure，然后原名义初态一次120s-or-failure；最多5个新episode/240模拟秒/480,000physics步/4,800控制动作，0训练动作。每个episode独立进程，内部不reset、不换策略，不把角度相加为单次成绩。

缓存编号按24项列表等距选取，执行前固定，不按旋转结果选样。它们是训练缓存样本，检验训练初态之间的稳定性，不证明未见抓取泛化、统计优势或无限旋转。

沿用评估wall cap：30s节点1500wall秒，120s节点按比例6000wall秒；actual stop必须如实报告，不把wall stop当完整物理时窗。无资源watchdog/cgroup/自动内存停止；仅固定评估wall预算，不启动训练。

普通物理失败继续余下固定节点，执行/source/checkpoint/初态错误或manual stop停止整条链；不自动重试、扩预算或改配置。每个episode保存full physics state/action/contact/valid-prefix gates、初态和控制输入，复用离线分析，保留raw provenance。

## 结果与验收

每项报告actual/valid时长、signed moving-cylinder-axis valid-prefix净角、峰值、回退、完整时窗最后10s（30s）或30s（120s）净角、最大漂移/倾斜/法向力及真实停止原因。最大漂移/倾斜沿用训练canonical参考，不重置参考以掩盖初态偏差。

建议进入后续鲁棒性分支的初步信号：四个30s节点和一个120s节点均完整执行，净角和完整时窗末段净角均为正。不新增固定圈数或姿态误差硬门槛；同时交付漂移/倾斜给用户判断操作质量。完成固定评估清单和满足旋转信号分别记录，失败不掩盖。

结束后自动更新docs/next_steps.md的第1步勾选和结果备注，提交小结果/代码/协议并push；权重和full traces留在ignored outputs。第2–4步不自动启动。
