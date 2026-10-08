# 第一轮与第三轮：训练结束自动评估

用户已授权。两套都选训练约1000万动作后的teacher_final.pth，各自使用自己的观测
归一化参数；最后一份不等于最好一份，本次固定相同训练预算进行对照。

复用第三轮已排定的最终评估，再补跑一次第一轮旧权重。统一原始grasp44、当前
boya_workspace终止规则、120秒窗口，报告净转角/维持时长/倒转/终止原因。
额外评估不会在训练期间占用GPU。此前第二轮比较队列维持停止。

[协议](PROTOCOL.md)。输出outputs/boya_hora_matched_workspace10m_v1。
当前准备中，未生成同条件成绩。活动训练/评估器及其已锁定源码不修改。

## 自动队列已开启

后台等待进程PID2650612，实际状态waiting_for_training_and_existing_evaluation，
额外GPU回合0。11项必要CPU比较测试通过，实际manifest的源码/观测/预算兼容准备通过。
运行时仅允许已声明的接触诊断字段增量，其余物理代码仍要求一致。
新训练及其最终评估正常完成后，自动运行第一轮最终权重，再生成JSON/CSV/Markdown对照。
遇手动停止、评估错误或身份不一致会记录原因并停止，不自动重试。

[队列状态](queue_status.json)、[队列命令](queue_launch.json)、[固定计划](prepared_plan.json)。
