# PhysX官方资料追查：速度与姿态增量的语义

2026-10-10，用户要求网络检索，区分引擎问题与API问题。
本次只查公开资料和对应源配置，新增物理步、训练动作均为0。

**官方明确说明：接触/约束下，报告的刚体速度不保证等于相邻状态的位姿差分。
所以此前把本地现象笼统称为“速度失真/数值矛盾”不够准确。
现有证据更支持求解器速度语义与本任务旋转评分之间存在适配问题；尚不能判定引擎bug。**

## 官方直接解释

[PhysX5.4.1手册，Solver Iterations](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#solver-iterations)：

> In general, and in particular when the bodies are subject to contact or other constraints, one cannot expect that the reported body velocity will match the position difference between two simulation steps.

文档解释其split-impulse策略：位置迭代包含几何误差修正；速度迭代主要求解不带该几何bias的速度约束。
返回并延续到下一帧的body velocity，不能直接当作这一帧位姿实际变化/时间。
文档的[TGS专节](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#projected-gauss-seidel-and-temporal-gauss-seidel)
还说明位置迭代对应内部子步，速度迭代求解最后子步的无bias约束，通常可在几何误差不大时使用0次速度迭代。

这是对现象的直接官方解释线索，不是对本地已安装SDK内部每个中间量的验证。
引用版本明确为5.4.1；本次未将它假写为本地SDK的精确版本手册。
原始getter/Lab最大差为0的既有结果，仍支持“未发现Lab读取层错误”。
本地接触段差异为何达到观测幅度、几何/碰撞/执行器/求解设置各贡献多少，尚未确认。
“接口正常返回其定义的速度”和“这个速度适合直接评价净转角”是两个不同的问题。

## 找到的参考配置差异

Hora vendored v0.0.1源码中，
[任务配置](../../../third_party/hora/configs/task/AllegroHandHora.yaml)写的是
`num_position_iterations: 8`、`num_velocity_iterations: 0`；
[总配置](../../../third_party/hora/configs/config.yaml)默认`solver_type: 1`（TGS）。
本地[BoyaParallel](../../../tendonspin/physics/isaac_parallel.py)是场景下限16/4，手部articulation请求16/4；
刚完成的16次诊断只将场景速度迭代下限改成16。

这证明参考配置与适配配置不同，不是已经执行/验证了原Hora基线，也没有证明0次必然修复本地现象。
在Lab中仅把scene minimum设为0，仍可能被actor/articulation自身的速度迭代请求抬高，
不能把这一操作声称为“有效0次”或“已经对齐原版”。任何后续对照都要明确并核对有效配置。
本次没有修改这些参数或开始新的回合。

## 诊断日志中的“超过4次”警告

官方[固定提交517a007的CHANGELOG，v5.3.0-105.1](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/CHANGELOG.md)
说明：过去TGS会将超过4次的速度迭代静默转换为位置迭代；新行为按请求分别计算两类迭代。
这与本地16次日志的警告直接对应，说明增加速度迭代并不等于沿旧行为单纯增加计算精度。
它没有证明本地4次配置存在某个已知bug。

还检索到与Isaac Sim6.1.0/omni.physx110.3.2有关的公开报告：
[GPU TGS速度迭代冲量丢失 #543](https://github.com/NVIDIA-Omniverse/PhysX/issues/543)、
[contact-last读取旧link速度 #549](https://github.com/NVIDIA-Omniverse/PhysX/issues/549)。
这些是各自带特定触发条件的报告，不能仅因版本相同就认定命中。
#549要求contact-last开启且external-forces-every-iteration关闭；本地冻结回合记录分别为false/true。
#543涉及分区/动态刚体数量等条件，本次未验证这些内部触发条件，不能将其定为本地根因。

## 后续方向

优先核对参考与适配的有效求解器配置及旋转输入语义，保留原Hora公式作为参照。
若开展有效0次等配置对照，应作为新的、单独声明的引擎配置；
若改用姿态差分输入，也应作为单独声明的适配变体。
在这些验证前，不能声称原Hora奖励有缺陷、已经证明引擎bug、或已经修好训练。
已有4/16两回合的原成绩和原始轨迹保持其历史身份。

[检索来源、原文、版本与hash](physx_web_sources.json)。下载的完整页面保存在忽略目录
`outputs/boya_physx_web_20261010`。部分搜索/API入口受JS、403或限流阻挡，记录中明确标注；
上述官方手册与固定提交CHANGELOG均已成功读取全文，不依赖搜索摘要。
