# Hora传参与本地有效迭代核对

2026-10-10，只读源码核对；单回合执行配置和结果另见本目录协议/README。

Hora vendored v0.0.1的默认配置确实把TGS、位置8/速度0传到了原生
`gym.create_sim(..., self.sim_params)`。源码路径如下：

1. [任务YAML](../../../third_party/hora/configs/task/AllegroHandHora.yaml:92)
   声明 `num_position_iterations: 8`、`num_velocity_iterations: 0`；
   [总配置](../../../third_party/hora/configs/config.yaml:23)默认TGS。
2. [train.py](../../../third_party/hora/train.py:56)将`config.task`转换为字典后
   传给任务；[转换函数](../../../third_party/hora/hora/utils/reformat.py:43)
   读取OmegaConf插值值。
3. [AllegroHandHora](../../../third_party/hora/hora/tasks/allegro_hand_hora.py:46)
   调用基类；[VecTask](../../../third_party/hora/hora/tasks/base/vec_task.py:141)
   调用`_parse_sim_params`。
4. [解析函数](../../../third_party/hora/hora/tasks/base/vec_task.py:379)
   逐项`setattr(sim_params.physx, opt, config_sim['physx'][opt])`，因此0不会
   被布尔判定跳过。[创建函数](../../../third_party/hora/hora/tasks/base/vec_task.py:224)
   只设置时间步/重力轴，然后直接把同一对象传入`gym.create_sim`。

在vendored任务及启动脚本中没有发现后续位置/速度迭代覆盖。Hydra允许用户从命令行
主动覆盖配置；这里核对的是参考源码默认路径，没有运行原IsaacGym Preview3，
不能据此声称执行并验证了原Hora基线或其内部实际求解次数。

本地已完成4次回合的实际USD显示：场景速度下限4/上限255；手部articulation请求4；
手部各刚体和圆柱请求1。场景位置下限16、articulation及刚体请求16。
只设场景速度下限0仍会留下这些正请求，所以不足以得到有效0次。

本地安装的`isaaclab_physx/physics/physx_manager_cfg.py:77`起明确规定：取actor请求
的最大值，再夹到场景min/max范围。`physx_manager.py:807`起把这些配置逐项写入
`physxScene:*`；PhysxSchema110.3.2的schema声明速度下限默认0、上限默认255，
刚体/articulation速度请求默认1。实际声明、代码片段及完整文件SHA256见
[source_check.json](source_check.json)，其中保留安装包绝对路径和版本身份。

本次独立诊断保持位置16，将场景速度min/max设为0/0，所有刚体/articulation请求
设为0。使用已有`scene_setup`钩子，在spawn后、`sim.reset()`初始化求解器前写入
运行场景的覆盖层，不修改源USD或已训练环境代码。初始化后再次读取所有迭代属性，
并与已有4次回合比较，只允许声明的速度迭代字段差异。

“有效0次”的证据是初始化前后的完整USD请求、0/0场景上下限及安装包聚合规则，
不是测量内部kernel的循环计数。已检查的原生tensor API和PhysX Python声明没有
直接求解迭代计数读取接口；运行结果另记录实际view上可见的solver/iteration方法。
本次位置16/速度0也不是原Hora位置8/速度0的完整复现。
