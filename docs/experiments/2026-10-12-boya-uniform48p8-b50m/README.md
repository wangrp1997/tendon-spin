# 新抓取B组训练

2026-10-12 18:30:24（上海）正式启动，外层PID3215548、训练PID3215550。
24个Isaac迁移通过的姿态，48.77355×32mm/50g圆柱；从零seed43，
1024环境、16动作，原B奖励hora_pose_delta_drop16、原workspace任务与引擎。
24个姿态来自五指手势方案，准入不要求同时五指接触。

总预算50,003,968动作，每10M一次固定新名义姿态30s-or-failure冻结评估。
每段恢复自己的完整learner，物理从同24-cache重置；旧抓取和全部旧权重保留。
输出：outputs/boya_uniform48p8_b50m_v1；TensorBoard位于各actions节点的
training/training，评估曲线位于evaluation_tb。首轮8192动作/update1/
Adam80已验证，初始两套normalizer为原始fresh值，模型已更新。
启动记录时49,152动作/update6，约715actions/s，无执行错误；首次评估约4小时，
全程约20–22小时，依据实际速度更新。此处仅确认启动，尚无旋转成绩。

完整配置见[固定协议](PROTOCOL.md)，实际源/状态/缓存身份见[启动证据](LAUNCH.json)。
