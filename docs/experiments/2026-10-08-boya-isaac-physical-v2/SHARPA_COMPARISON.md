# 与本机 Sharpa RL Lab 的针对性对照

用户要求解释固定目标下的异常运动，并参考已有 Sharpa 项目；不追加额外检查。
本次仅读取两边执行器代码和组合 USD，无引擎启动、物理步或训练。
参考仓库：/home/rw/Projects/sharpa-rl-lab，HEAD 5accf024d376685eaa17da7aa4614498217eab4d。

|项目|本机 Sharpa RL Lab|当前伯牙 physical-v2|
|---|---|---|
|关节驱动|22 个主动关节，组合 USD 未见 mimic|18 个源电机，另有 4 个被动联动关节；导入文件声明 NewtonMimicAPI|
|电机折算惯量|组合 USD 的 22 个关节均配置 physxJoint:armature，范围 0.00012–0.0032 kg·m²|当前 USD 和执行器配置没有显式设置 armature；v2 未保存实际运行时 armature 读回|
|速度阻尼|外部计算 Kp(q_target−q)−Kd·qdot，再经 IdealPDActuator 的执行器限幅路径|源位置弹簧力先限幅；被动阻尼改作 PhysX viscous_friction，执行器转发器增益为 0|
|碰撞排除|组合 USD 有 15 个 filteredPairs 关系目标|源 XML 的 28 对具名排除未显式移植，PhysX 自动相邻过滤另计|
|初始夹持|在该仿真环境生成 grasp cache；训练从缓存采样|直接恢复 MuJoCo 中保存的 grasp44，需要正确转接动力学|

直接源码位置：
- Sharpa rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env.py:97–109、194–201；
  读取 USD 的增益，外部 PD 力矩控制时将转发执行器增益清零。
- Sharpa assets/SharpaWave/right_sharpa_wave.usda 引用的组合物理资产；
  armature 来自 configuration/right_sharpa_wave_physics.usd 等组合层。
- TendonSpin tendonspin/physics/isaac_boya.py:make_scene、SourcePositionAdapter.apply。

Sharpa 当前旋转配置还使用初始弱重力课程（−0.05），抓取配置使用全重力；
不能直接复制弱重力而称为相同伯牙任务的修复。两边都使用 IdealPDActuator，
因此不能把区别描述为“Sharpa 隐式 PD、伯牙显式 PD”。原始 USD 的 drive 增益
存在角度单位转换，不能将文件数值直接当运行时 SI 增益比较。

解释：固定位置目标仍会产生保持力矩。电机折算惯量、阻尼路径和联动约束会决定
关节受到这些力矩后的动态响应。Sharpa 的可运行资产包含伯牙本次没有完整移植
的执行器动力学配置，尤其此前漏看的 armature。当前记录支持数值发散，
不支持认定 Isaac 引擎本身出错，也尚未隔离证明 armature、阻尼或联动中的
某一项是唯一原因。下一步修复应参考其完整执行器配置；伯牙的折算惯量需要
来自本机传动参数或明确标为仿真假设，不能把 Sharpa 数值照搬为伯牙真实参数。

视频说明：pose_replay_v2.mp4 的运动来自已保存 Isaac/PhysX 世界刚体姿态，
画面由 MuJoCo CAD 渲染器绘制；它不是 Isaac 原生录像。之前用户回答未突出
渲染器来源，现明确更正。此轮不为更换画面重跑物理或制作额外视频。
