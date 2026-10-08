# 正式训练从零启动，续训测试单独保留

用户最终要求：先验证续训，再从零后台训练约1000万次。已完成续训验证并停止
原短续训进程；正式训练是独立新运行，**没有 --resume，没有加载旧策略**。
实际读取初始权重确认与seed43新初始化一致、与已训练预览策略不同；归一化
也重置，首检查点update1/actions8192/Adamstep80。测试数据没有计入正式训练。

当前状态：**训练预算与独立评估已完成**，结果见文末。保留启动快照：14次更新/
114,688次动作，约849.2动作/秒；该数值
会随运行推进，不表示训练完成或旋转成功。1024环境、无头/无相机、nice+10、
4线程，未启用资源看门狗、内存阈值停止或cgroup限额。
目标1000万按完整rollout对齐为10,002,432次动作/1221次PPO更新，全部从零累计。
物理任务、PPO、28-state抓取缓存与原协议一致。当前仍是名义特权教师，
DR/student、完整基线对比、实际部署与持续旋转验证尚未完成。

TensorBoard：**http://127.0.0.1:6006/**，选择 **formal_fresh/.**。
已实际从HTTP读取到该运行的曲线标签：总回合奖励、回合长度、actor/critic损失、
旋转奖励、有效净转角/峰值/倒转/有效时长、各终止原因比例、学习率/KL/熵和速度。
每个更新flush，横轴是本运行累计动作数。preview、resume_test为独立历史曲线。
曲线是探索训练回合统计，不是原始状态固定策略评测成绩。

- 事件文件：`outputs/boya_hora1024_fresh10m_v1/training/training/stage1_tb/`
- 训练进度：`outputs/boya_hora1024_fresh10m_v1/training/result.json`
- 训练日志：`outputs/boya_hora1024_fresh10m_v1/training.log`
- 检查点索引：`outputs/boya_hora1024_fresh10m_v1/training/latest_checkpoint.json`
- 后台阶段状态：`outputs/boya_hora1024_fresh10m_v1/stage.json`

首更新、每16个更新及正常结束保存配对检查点（模型、观测/价值归一化、Adam、
LR/RNG与步数），未来可从本次正式训练的检查点继续。每16更新约131,072次动作。
只有轻量统计和模型写盘；不再保存全部训练物理轨迹。

续训验证：原模型/两套归一化恢复完全一致，实际Isaac/CUDA训练首恢复更新的
Adam640→720、动作65,536→73,728，参数改变且loss有限。[证据](resume_verification.json)。
原续训被用户要求停止，退出143，记录的是最后完整更新21/172,032累计，
最近周期安全检查点16/131,072；没有假称正常结束或保存了所有中断状态。
Isaac覆盖了初始化前的Python退出回调，入口已改为场景初始化后安装保存退出
回调，本次启动记录确认安装。没有额外运行GPU退出测试；周期检查点持续保存。

正常达到预算后已执行原先排队的120秒窗口原始状态冻结策略评测，
在首次触发已声明的终止门槛时结束；逐步证据及结果见文末。
后台进程不依赖聊天保持开启。手动请求停止时不启动该评测。

[协议](PROTOCOL.md)、[启动快照](launch_snapshot.json)、[启动时训练记录](training_launch_snapshot.json)、
[进程命令](process_launch.json)。启动时仅本地提交；用户之后创建远端并授权阶段性push，见仓库AGENTS.md。


## 10M完成与独立评估（2026-10-08）

训练正常exit0，10,002,432动作、1221更新，stop=update budget，没有自动续训。
预定final检查点评估exit0，只表示程序正常结束；同一原始grasp44、单一冻结教师、
0reset/0switch、请求120s、实际2.1485s、有效2.148s，净角/峰角+63.626313°、
倒转1.042972°，stop=tilt>15deg。30s/120s预算均只有这条短前缀，不外推后续运动。

末个有效帧：倾斜14.996597°、位移2.555635mm、最大法向力3.674483N。
首次违规帧：倾斜15.005651°、位移2.559715mm、最大法向力3.683823N，
其他已记录数值界未触发。这说明自定义倾斜阈值终止，不证明已经掉落。

用户指出的复现偏差已核实：旧伯牙5mm/15°/12N验收界被用于RL训练done，
与Hora原版高度/时限终止不同，不得称完整复现，+63.63°不是原版Hora的能力上限。
详见[源码核对](../../research/2026-10-08-training-stop/README.md)。
本轮证据保留，没有通过事后改阈值重写成绩或启动替代实验。

小证据：[训练摘要](training_final_summary.json)、[独立评估](evaluation_result.json)、
[末两帧](terminal_evidence.json)、[后台完成状态](completed_stage.json)。
完整physics/control轨迹、权重与TensorBoard保留在原ignored outputs目录。
