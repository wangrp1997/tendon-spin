#128环境/28抓取的Hora旋转训练已启动

用户要求适当增加显存使用并保持电脑响应。当前128个Isaac无头环境，
相机关闭，CPU nice+10，Torch/OMP/MKL线程4；整卡显存采样约
3835MiB/16303MiB。显存>12GiB时训练在更新边界停止，
这是采样监控不是硬隔离，GPU利用率会随时变化。

从seed43随机网络重新开始，均匀采样28个已筛选抓取，PPO/任务/物理保持
[协议](PROTOCOL.md)所述设置。128×horizon8×64updates=65536动作预算，
minibatch512、每次5epoch，每16updates存一次checkpoint与optimizer/RNG。
训练上限1800秒，完成后独立原初态冻结策略评测最多120秒仿真时间。

本页是启动阶段记录，**尚未完成训练或评测**。抓取缓存、训练复位与
原初态评测分开。捕获时完成20次更新、20480次动作；
不得将这份启动快照当作后续实时状态。
[启动记录](launch.json)、[实际配置](config.yaml)。
实时进度：outputs/boya_hora_cache28_env128_stage1/training/result.json；
阶段状态：outputs/boya_hora_cache28_env128_stage1/stage.json。
结果目录持续写入，自动评测尚未执行，不提前给旋转成绩。

此前64env扩展阶段在用户要求调整并行数后中止，旧数据单列；当前训练
样本计数从0重新计算，不叠加旧阶段数据。完整属性随机化/学生/真机
还未完成，这次为固定名义物理条件下的旋转学习。
