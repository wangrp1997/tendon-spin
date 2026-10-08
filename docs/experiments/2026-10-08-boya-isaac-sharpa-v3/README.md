# 伯牙 Isaac 执行器适配：完成原抓取 5 秒保持

用户批准“开始适配快速交付”。直接参考本机 Sharpa RL Lab 的外部 PD→限幅执行器
组织方式完成适配，按[预先声明的协议](PROTOCOL.md)执行一次原初态episode。
controller=boya_sharpa_pd_hold_v3；0控制器切换、0episode reset、0策略训练。
实际10000步/5秒，进程84.90秒，正常到达预定时限。

|指标|v2历史失败|本次v3|
|---|---:|---:|
|实际执行时间|0.003 s|5.000 s|
|最大圆柱位置偏移|5.856909 mm|0.125847 mm|
|最大关节速度|3620.986 rad/s|0.904000 rad/s|
|最大单link法向合力|668.873 N|4.025412 N|
|最大联动误差|4.163788 rad|0.000002988 rad|
|停止原因|位置偏移超过5mm|预定5秒时限|

保留原40×32mm/50g/grasp44和全重力。新版本增加声明的电机惯量，
将主动电机位置项与速度阻尼统一限幅；被动关节阻尼折算到联动主电机，
没有提供被动关节独立动作。清零solver joint friction，显式恢复28对原碰撞排除，
保留NewtonMimicAPI并显式写出1:1关系。原v2代码/数据快照继续留存。

本次通过的是预先声明的短保持标准，已消除这次运行中“3ms异常发散”的症状。
这是多项配置一起修复后的执行结果，不能单独归因于armature或某一项阻尼。
armature按A=4·D_effective·dt得到，未测量真机电机折算惯量；普通指关节.0001、
联动主关节.0002、腕.001 kg·m²。这会改变动力学，仍需后续标定/随机化，
不能把短保持结果当作真机鲁棒性、完整动力学等价或持续旋转成绩。
MuJoCo的solref/solimp没有严格映射；实际联动误差满足本次标准。
完整穿透/外部支撑等物理认证未在此轮新增，benchmark_validated=false。

[Isaac原生录像](../../../outputs/isaac_boya_sharpa_v3/isaac_native.mp4)：
同次Isaac Sim RTX相机直接拍摄，20Hz、含初态和终态；画面显示真实仿真时间。
末尾多停留1秒；没有MuJoCo绘图、姿态回放或重新运行物理来拍摄。
[小型执行记录](result.json)包含模型、配置、源码SHA256及逐步NPZ路径。
原始数据、源码快照、视频位于ignored outputs/isaac_boya_sharpa_v3/；
启动日志outputs/isaac_boya_sharpa_v3_launch.log。

实现入口：
- tendonspin/physics/isaac_boya_sharpa.py：参数、约束与Sharpa式PD适配。
- tendonspin/physics/isaac_boya.py：保留v2默认行为，增加执行器与场景构建扩展入口。
- scripts/run_boya_sharpa_hold.py：同次物理执行、逐步记录和原生录像。

按用户要求不追加额外检查或扫描。阶段代码本地提交，不push。
