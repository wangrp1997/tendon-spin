# 官方无渲染入口启动/步进通过

已读当前状态、后端决策和teacher-v2协议/结果。用户明确同意本次进程许可，
并在150s直接SimulationApp启动超时后同意一次最多5分钟的官方headless诊断。

使用既有env_isaaclab、官方AppLauncher + isaaclab.python.headless.kit，
CUDA:0/默认PhysX/.005s/全重力/无相机；没有手、物体、控制器、训练或模型替换。
**约3.95s进程结束，return_code0；构造返回、reset返回、3步均返回。**
这是空场景可启动的证据，不是GPU多环境性能、伯牙接触/耦合、训练或旋转成功。
此前完整SimulationApp路径150s超时单独保留，不推断驱动或引擎一定有缺陷。

最后落盘阶段为before close。安装版SimulationApp.close源码明确默认fastShutdown
直接终止进程，因此没有观察到协议预期的close-return标记。这个标记要求未满足
须保留；构造/reset/3步/正常退出事实不受影响。没有追加第二次启动来美化记录。
完整日志在outputs，launcher/experience/API/脚本/协议哈希和父进程结果见
[小证据](../../data/isaac_lab_headless_diagnostic.json)；
[固定协议](PROTOCOL.md)；[旧直接入口超时](../../data/isaac_direct_startup_timeout.json)。

启动问题已有可用官方入口。后续独立校验原伯牙URDF/CAD、自碰撞、四处被动mimic、
13实际输入/力矩/位置界及原无支撑抓取；通过前不启动大型PPO。没有安装/重建环境。
