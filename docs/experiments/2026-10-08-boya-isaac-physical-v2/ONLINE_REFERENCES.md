# 在线官方依据：缩小Isaac物理转接问题

按用户“有些问题可以检索下网络”的要求，进行了有界官方文档/issue搜索和静态核查。
0新引擎启动、0训练、0新物理步。URL、UTC、响应状态、原网页SHA256、可用范围与
短引文见[来源清单](data/online_source_manifest.json)；原文留在ignored outputs。

[PhysX5.6.1官方Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#articulation-drive-stability)
讨论手指抵住物体时drive、limit、contact互相覆盖，以及mimic与stiff drive共同存在时
的求解困难。约束顺序影响最终速度，增加迭代有时没有定性改变。Mimic Joints一节
明确limits/contacts在mimic之后处理，可能牺牲mimic关系；mimic冲量不受源电机
maxForce直接限制。所以电机限幅正确仍可能有巨大约束力。
这是官方机制，不是已经复现某个Isaac6.1 bug。v2外部计算PD、转发drive增益0，
与该页implicit-drive例子不同，不能直接把全部原因归为gain太大。

[MuJoCo官方solver参数](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
解释正值solref的timeconst/dampratio柔顺，以及solimp的阻抗。
[本机源参数](data/source_constraint_audit.json)四处active mimic均为
solref=[.002,1]、solimp=[.95,.995,.001,.5,2]，不是无限刚性的角度等式。
USD只显式复制mimic角度比例/offset/enabled/leader关系；没有移植和验证上述柔顺
响应。这是具体的转接缺项，比例相同不代表耦合动力学等价。
安装版NewtonMimicAPI是当前官方模式；不能仅因没有旧PhysxMimicJointAPI就认定
mimic未生效，也不能未经响应测试就断言默认硬约束是本次唯一原因。

[PhysX摩擦API](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxJointFrictionParams.html)
定义粘性项−coefficient×speed；Articulations还说明TGS/PGS摩擦冲量累计不同，
TGS可能显得更弱。[USD轴参数](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/joints/physx_joint_schema.html#physxjointaxisapi)
的角单位是degrees，而本机Lab actuator接口声明rad。因此要核对层间转换与耗散
响应，不能盲目将.05乘/除57.3。v2系数读回正确不代表实际衰减与MuJoCo
implicitfast积分等价。所取在线Tensor页未包含setter正文，不用它证明单位。

[Isaac Lab issue4129](https://github.com/isaac-sim/IsaacLab/issues/4129)
涉及Sim5.0/Lab2.2.1的friction命名/模式问题，提出缺少轴API时setter可能无效。
它提醒做实际响应测试，不能当成本机Sim6.1已经确认的bug。其他Robotiq等issue
版本/机器人不同，未核实到可直接照搬到伯牙的完整修复。评论API403；Google
跳转页和Bing无关MRT结果均保留并排除，不伪称全网已经查全。

|问题|已有证据|有依据的下一步|
|---|---|---|
|坐标/电机传递|v2初态FK通过、action0/目标不变、请求/转发/读回力矩一致|保留已通过项，不改抓取或放宽漂移|
|联动/柔顺|源软约束没有对应移植，3ms联动误差达4.16rad|最小无接触诊断验证mimic动态，再确定柔顺映射|
|碰撞过滤|28对源exclude未显式转入；v2没有手内冲量轨迹|按原具名排除恢复映射、记录实际手内接触|
|阻尼/惯性/积分|数值仍异常，系数读回过不代表衰减一致|测零电机自由衰减、核对实际惯性/时序，不随机扫gain|

建议下一次限定<=5分钟，只隔离验证物理接口，等用户选择后再启动。
若无接触仍异常，优先定位joint/阻尼/惯性；若仅接触后异常，再查源碰撞排除与
约束柔顺。无接触诊断不计抓取/旋转成绩；正式原初态任务必须另行完整执行。
此处没有修复成功、基线训练完成或SDK唯一根因的结论。
