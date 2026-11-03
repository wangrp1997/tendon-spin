## 2026-11-03 TendonSpin本项目学生蒸馏及独立评测已完成

本项目首轮训练及独立闭环评测已于北京时间 **2026-11-03 18:34:59** 全部完成。训练2000控制步、2,016,000次仿真交互（含验证环境），两个学生各8896次更新。固定14,400条验证样本选择best；测试使用独立种子2026110203，27组新ID物理条件，三个冻结控制器各运行完整600控制步，每组最多30秒。

| 策略 | 平均净旋转 | P10净旋转 | 保持30秒 | 教师平均旋转比例 |
| --- | ---: | ---: | ---: | ---: |
| C1260M教师 | 1026.46° | 494.84° | 24/27 | 100% |
| 触觉学生 | 986.73° | 405.20° | 22/27 | 96.13% |
| 纯本体感觉学生 | 1016.39° | 511.11° | 22/27 | 99.02% |

两个学生均达到预先固定的仿真门槛（平均净旋转≥教师90%，保持率下降≤10个百分点）。本轮纯本体学生的平均旋转比触觉学生高3.01%，两者保持率均为22/27；这组有限案例未显示触觉增益，也不足以证明总体或实物性能优劣。按此轮结果，后续接入以`proprio_best.pth`为优先基线，`touch_best.pth`保留作对照；测试未用于重新拟合或挑选新的训练检查点。

- 正式输出：`outputs/boya_touch_student_20261102_v1/`；最终成绩：`evaluation/comparison.json`。
- 纯本体权重：`training/proprio_best.pth`，SHA-256 `d9cb5593e383c4b0f34986d0288d647a599390d96ea91b33681794efd17cdbe2`。
- 触觉权重：`training/touch_best.pth`，SHA-256 `4e1c9bfbecd11857153a7c4a51275996036ce062679b94844099e1de6259c490`。
- 复核：`final_verification.json`；两份权重、固定案例、三份轨迹哈希均一致，从物理步旋转增量重算成绩通过，外力和编码器噪声逐项配对一致，10个队列源及36个训练依赖保持原哈希。
- 主监督3709537、训练3709647、评测1128790均已退出；宿主机进程及GPU列表已核对，本轮显存已释放，其他项目任务保持运行。无需重启或重复评测本轮。
- 实物尚未验证；使用原抓取缓存和名义1∶1耦合，不增加J1/J2专项测试。SDK采集/驱动仍需按现有传感推理接口连接，本轮未发送实物命令。权重及原始轨迹存于服务器输出目录，Git记录代码、说明和成绩摘要。

下方2026-11-02启动记录保留作历史，实时最终结果以上述完成记录为准。

## 2026-11-02 TendonSpin 本项目历史触觉学生：实现完成，正式蒸馏已启动

用户明确要求开始本项目自己的学生蒸馏。不是botyard任务，也不是续训C/D教师；J1/J2保留现有1∶1名义模型，不安排比例专项测试。新增实现独立于冻结教师源及旧20帧部署接口。

- 正式输出：`outputs/boya_touch_student_20261102_v1/`；监督PID **3709537**，物理GPU1 / `GPU-a3f9cc58-fe6f-96d3-ab1a-e1dc33b09e7a`。启动时GPU1约33GiB空闲；GPU0未启动本任务。
- 实时总状态：`status.json`；学习状态：`training/status.json`；日志`training.log`/`evaluation.log`。启动状态不等于学习或验收已完成，后续以实际状态为准。不得重复启动本轮。
- 教师固定C1260M，SHA `a1a4f3c20c62ed766bcce014eb15704d80b962cee0ffe4edff95ba1e2ef4b5a2`。已核验其原物理/任务依赖文件均保持原哈希；奖励、终止、机械耦合不变。
- 1008环境，864训练/144独立验证，九尺度均衡。每帧57维，30帧历史；20路真实可测关节角（含四被动J1）、16路已发目标及接触/力幅值/有效性/时效。无相机必需输入。仿真触觉为distal刚体法向加摩擦合力代理，不是压力阵列或已校准的局部传感器。
- 同时训练touch和proprio两学生；后者屏蔽全部触觉及其元数据。原96维观测归一化与16维动作头冻结，独立TCN预测8维latent。使用动作MSE+latent MSE，只更新TCN。
- 32步教师采集、1024次预热更新，然后1968步逐步转学生控制，每16步各64更新；计划每学生8896次更新，总2,016,000仿真交互（含验证环境）。两个学生自己访问的状态均进入训练回放，验证环境始终由教师控制。
- 固定验证集来自前400控制步，每4步采样，收齐之前不选best，仅显示临时验证分数。数据不进入梯度更新。最终测试使用独立seed2026110203，训练seed2026110201；27新ID物理条件、三控制器、每例30s，全部有效前缀保留，不接管/不重置。
- 全队列自动先训练再测试，阈值提前固定：学生DR平均净转≥教师90%，保持率下降≤0.1。报告本体对照与失败，不提前声称触觉有效或实物可用。仍使用旧抓取缓存，不能宣称未见形状/耦合验证。
- 已通过11项CPU检查。完整GPU短流程`outputs/boya_touch_student_20261102_pipeline_smoke_02/`已完成：两学生各3次真实更新，冻结权重与教师动作一致性通过，独立学生加载及三控制器配对初态/刺激核验通过；该0.1s短流程仅验证实现，不作为策略效果成绩。
- 首次调试采集`outputs/boya_touch_student_20261102_smoke_01/`暴露训练前验证best占位值不可JSON序列化，已修复；原记录保留，不计作有效训练。其进程3659178已退出。正式版本改为有限值/空值日志，并增加逐步原始传感帧、监督、执行动作、回合与完成掩码归档。
- 新代码：`tendonspin/students/`、`scripts/train_boya_touch_student.py`、`scripts/evaluate_boya_touch_student.py`、`scripts/run_boya_student_distillation.py`；使用说明`TENDONSPIN_STUDENT.md`。检查点schema `tendonspin-c-touch-history30-v1`，不接受旧策略或botyard检查点。
- 此次未发送实物命令。`TouchStudentRuntime`仅返回动作，实际SDK驱动由后续明确通道/单位/控制语义后连接。权重不做实物在线更新，不要求专项标定。
- 当前源码和协议已快照；队列按哈希校验。后续勿修改本轮绑定的新训练/评测Python文件，否则后续来源验证会停止。创建输出根`STOP`可在控制边界停止整队列；如中断，保留learner/日志，由新恢复方案明确缓存重置与回放重建，不自动宣称精确续跑。

## 2026-11-01 botyard C1260M 蒸馏与独立闭环验收已完成，可交接

用户委托本代理的 botyard 蒸馏任务已完成；不是仅启动训练。正式队列于 2026-11-01T20:48:37.019430+08:00 完成，约22.60分钟（不含开发/先行教师转移验证）。40条训练/验证采集、10000监督updates、30条独立闭环评估全部完成，全部精确原生回放一致。训练17775帧、独立验证6000帧，按验证集选7000步，测试不参与拟合/选择。

| 策略 | 九个新DR平均净旋转 | DR保持30s | 名义净旋转 | 名义保持30s |
| --- | ---: | ---: | ---: | ---: |
| C1260教师 | 1082.00° | 9/9 | 1368.64° | 是 |
| 旧40M名义学生 | 169.06° | 5/9 | 1508.79° | 是 |
| 新C1260学生 | 1048.68° | 9/9 | 1355.42° | 是 |

新学生达到教师DR净旋转的96.92%；名义旋转相对旧学生下降10.17%，必须保留此取舍说明。本轮有限面板满足预声明的名义闭环与DR新旧比较标准，`recommended_as_DR_default=true`，仅为仿真接入验收，不是总体/实物鲁棒保证。

- 可用学生：`/home/wangrenpeng/botyard-inhand/outputs/research/c1260_distillation_20261101/trial_01/handoff/student_best.pth`。
- 学生 SHA-256：`e46490f802adf656e4f58e96c7447d822ff597c4e145bb12ee6dcd4fa51fa986`；教师 SHA-256：`a1a4f3c20c62ed766bcce014eb15704d80b962cee0ffe4edff95ba1e2ef4b5a2`。
- 交接说明：`/home/wangrenpeng/botyard-inhand/outputs/research/c1260_distillation_20261101/trial_01/handoff/README.md`；清单 `handoff/manifest.json`；完整小型结果 `comparison.json`。
- 原 botyard `SensorStudent` 严格加载通过；student schema仍为sensor-student-v1，新教师动作头/normalizer已内嵌，部署不加载教师、不用物体真值。旧学生未覆盖。
- 接入新底座时不能默认复用旧A的残差/特征校准/critic/回放，应由目标代理另建绑定并验证。
- 代码 `research/c1260_distillation/` 已就地交付到botyard。独立本地分支 `research/c1260-sensor-distillation-20261101`，提交 `90e0f6e8f6508d5896a1f092b31c43e6f9e257a9`；通过临时索引提交，目标当前分支/HEAD/原索引/未提交改动均保持。
- Git远端git.nrs-lab.com当前无可用认证（Coder auth provider不匹配、无非交互用户名），因此未推送；本地代码、模型和交接文件均可直接使用。未向其他代理发送消息，由用户告知。

详细记录：`docs/experiments/2026-11-01-teacher-selection/BOTYARD_DISTILLATION_RESULT.md`、`outputs/server_c50m_deployment/botyard_distillation_result_20261101.json`。下面进行中记录已成为历史，禁止自动重跑已经完成的蒸馏队列。

## 2026-11-01 用户委托本代理在 botyard 完成 C1260M 蒸馏

用户明确要求本代理按 botyard 代码/流程完成蒸馏，之后由用户自己通知那边代理。已在目标仓库新建独立 `research/c1260_distillation/`，未替换旧学生/适应实验，无其他代理消息。结果目录 `/home/wangrenpeng/botyard-inhand/outputs/research/c1260_distillation_20261101/trial_01`；监督 PID 952303，实时 `pipeline_status.json`，最终预期 `handoff/`。

新教师 CPU 推理与 Isaac 存档60组输入最大误差4.17e-7；DR特权归一化误差0。官方 MuJoCo3.16.0 名义原始初态完整30s净转1364.1534°、末10s468.0555°，原检查通过、精确回放一致。名义与一条DR的实际传感采集各完整30s，回放通过。三个必要检查通过（reset窗口边界、条件划分独立、实际采集传感流与原 SensorInference 一致）。

固定30训练回合、10独立验证回合；原 SensorStudent CNN/TConv、原action+latent loss、C1260冻结动作头/normalizer，10000updates。最终10独立条件（九尺度DR+名义）分别评测教师/旧学生/新学生，保留提前失败前缀并核验三者初态与实际属性配对。四个原生进程在GPU1渲染，Torch拟合GPU1/15%显存上限。当前采集队列已实际启动；训练尚待采集完成，不要把准备当作训练完成。

源码与 protocol/binding 已钉住；后续不要修改这些 Python 文件或 `distillation_protocol.json`。数值/实现/回放/身份错误停止，不自动重试；普通预定物理失败保留且继续其他固定案例。旧学生保持原 SHA 43c564...9f3ba。最终能否建议换底座由完整独立闭环评估决定，不能提前说“已经可以用了”。

交接最终结果以 `comparison.json` 与 `handoff/manifest.json` 为准。相关记录 `outputs/server_c50m_deployment/botyard_distillation_delegation_20261101.json`。

## 2026-11-01 教师评估已全部完成：best = C1260M

完成于 2026-11-01T19:26:37.297057+08:00，总耗时 125.24 分钟。四个权重各 492 个新条件案例、共 1968 例全部完成，推理冻结、实际初始状态/物理参数配对和最终 checkpoint/policy/source-record 哈希复核通过，无基础设施未完成。输出 `outputs/boya_c_teacher_selection_20261101_v1/status.json` 的 `phase=completed`，自动最终清单 `best_teacher.json` 已生成。

| 权重 | 平均净旋转（度） | P10（度） | 保持30s |
| --- | ---: | ---: | ---: |
| 1240M | 990.98 | 468.93 | 444/492 (90.24%) |
| 1260M | 1020.31 | 525.01 | 436/492 (88.62%) |
| 1460M | 968.74 | 603.30 | 461/492 (93.70%) |
| 1500M | 949.71 | 190.66 | 436/492 (88.62%) |

按事先固定的平均有效前缀净旋转主目标选 C1260M；保持率及 P10 更高的 C1460M 保留为稳定性备选。1260M 相比最后 1500M 的平均净旋转提高 7.43%，保持率同为 88.62%。

- 主教师完整 learner：`outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/actions_1260000000/training/teacher_final.pth`；SHA-256 `a1a4f3c20c62ed766bcce014eb15704d80b962cee0ffe4edff95ba1e2ef4b5a2`。
- 配套 policy：`outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/actions_1260000000/training/teacher_final_policy.pth`；必须使用同 checkpoint 的观测归一化器。
- 结果记录：`docs/experiments/2026-11-01-teacher-selection/RESULTS.md`；完整比较 `outputs/boya_c_teacher_selection_20261101_v1/comparison.json`。
- 监督 PID 577552 与最后 worker PID 772753 已退出，本轮显存已自动释放。未改其他任务，未启动学生训练。下面的启动/进行中记录均为历史。

## 2026-11-01 教师四候选新随机条件配对评估（已授权运行）

用户确认比较 1240M、1260M、1460M、1500M 并换新随机条件后，明确要求“好的你评测下？”。已于 2026-11-01T17:21:22.613471+08:00 启动独立评估队列，监督 PID 577552，物理 GPU1 / GPU-a3f9cc58-fe6f-96d3-ab1a-e1dc33b09e7a；不继续 C/D 训练。此节是启动记录，后续实时状态以 status.json 为准。

- 输出：`outputs/boya_c_teacher_selection_20261101_v1/`，执行入口：`outputs/server_c50m_deployment/run_teacher_selection_20261101.py`。
- 实时状态：`status.json`；详细日志：`supervisor.log` 与各权重 `.log`；已完成 9env/九尺寸/1s 实际属性及初始状态验证，当前首个完整模型为 1260M。
- 新测试种子：2026110101、2026110102、2026110103，与旧种子不交叠；同一套 164 个缓存抓取 × 3 新条件共 492 案例/模型，每例最多 30s。四模型依次 1260M、1460M、1240M、1500M，全程串行，共 1968 案例。
- 四个 learner/policy/normalizer 的 CPU 审计通过，原配对评估 6 项 CPU 测试通过；独立临时目录用历史结果验证最终报告/选择流程通过（不是真实新评估结果），见 `offline_pipeline_validation.json`。
- 固定任务、物理、分布、缓存、终止与评分，使用各自匹配且冻结的 normalizer。所有实际案例元数据、初始状态及物理参数必须配对通过。
- 全部完整评估和核验通过后，自动保存 `comparison.json`、`comparison.md` 和 `best_teacher.json`。预声明排序以全部案例平均有效前缀净旋转为主，保持率和 P10 同时报告，另列保持率最高备选。未全部完成前不能声称已选定新 best。
- 协议：`docs/experiments/2026-11-01-teacher-selection/PROTOCOL.md`。每模型墙钟预算 14400s；基础设施错误停止且无自动重试，保留部分结果；不把未完成评估当作策略失败。共享 GPU 负载可能影响速度。
- 原始权重/历史 fixture/冻结训练代码未覆盖。不改变其他运行进程，不启动学生训练。不要修改已钉住的本次协议或监督脚本，否则运行时来源核验会停止。

# 服务器交接：五指构型 C 组教师训练

## 2026-10-29：D评估超时后决定提前结束，资源已释放，C继续训练

用户随后明确确认“先停止D并做好实验记录”。正式归档 `docs/experiments/2026-10-29-d-termination/EXPERIMENT_RECORD.md` 已完成，包含实际配置、最近六个完整评估、同预算C/D对照、超时无效结果边界及权重/证据哈希；原冻结协议和原始记录均未改。

用户询问D是否报错、是否值得继续，并要求不继续时释放资源。核实D已完成1150025728动作/8774更新，随后1150M随机化评估达到7200秒运行上限，实际7206.74秒仅推进14.9/30仿真秒，491/492例已物理终止，评估以error和wall_budget_exhausted=true保存。监督27017于北京时间21:43:54附近停止，评估worker1801278已退出；GPU进程列表均无两者，资源在本次决定前已释放。本次未发信号或启动重试，未将不完整1150M评估当作有效成绩。

决定**D本轮提前结束，不补跑1150M评估、不继续至原1.5B预算；后续若要恢复须有新的明确指令**。最近1090M至1140M六个完整492例评估，平均净角约133–135°、有效时长约3.8–3.9秒、30秒保持均为0/492，停止原因为物体低于Boya操作区。1140M同预算C为871.65°、保持415/492；D这一单次零掉落成本消融尚未学会稳定持物旋转，继续计算的预期收益有限，不据此推断所有随机种子的因果结论。

D1150M完整learner/policy及此前历史全部保留；CPU审计确认成对、模型/两类normalizer/Adam有限及72个固定源未变。learner SHA256 `cb201ce968be7e96b2514affa30760823d5c2de6ab44c7fa1e3997a918daf58c`。原status、错误、部分评估及checkpoint哈希未改，停训决定另存当前D输出的 `termination_decision_20261029.json` 和 `outputs/server_c50m_deployment/d_termination_decision_20261029.json`。本次未修改训练源码、协议、奖励或其他任务，也未重启网页；原网页保留故障记录。21:51快照C约1304.56M/86.97%，监督27016/worker1779177正常训练，无当前错误；WM23652在110M例行冻结评估中，无当前错误，后续以实时状态为准。

## 2026-10-28：服务器中断后 C/D 续训首更新验收通过，三组网页监控恢复

用户明确要求继续训练并恢复 C、D、WM 网页监控，随后澄清故障是**显存耗尽**，系统内存充足、不需要额外管理。系统本次启动时间为北京时间17:50:57；旧C/D心跳最后在17:43:58/59附近，无新的执行错误记录。显存耗尽是用户报告的故障原因，本次未独立证明死机根因。保留全部旧目录、日志、权重和状态，不把旧status中的training视为仍然存活。

复用原 `scripts/resume_boya_interrupted.py`，在新目录准备并启动：C输出 `outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/`，监督 **27016**、worker **27371**；D输出 `outputs/boya_d_fresh1p5b_server_interrupted_recovery_20261028_v1/`，监督 **27017**、worker **27374**。C恢复自己的1231028224动作/9392更新，重算10次未保存更新；D恢复1061158912动作/8096更新，重算11次未保存更新。原GPU1 UUID、16384环境、1.5B累计预算、每10M同492例/30秒DR评估、学习和物理合同及既有显存准入保持。C下一目标1.24B，D下一目标1.07B；不要重复启动。

**北京时间18:20首更新验收均通过**：C9392→9393更新、1231028224→1231159296动作，Adam12021760→12023040；D8096→8097更新、1061158912→1061289984动作，Adam10362880→10364160。恢复初始模型和两类normalizer与各自断点逐张量相同，合同相同，首更新参数确实变化、模型/normalizer/Adam有限、完整learner和policy配对。C78/D72个冻结源及原中断证据哈希不变。原恢复CPU测试4项/9子项通过；首更新证据 `outputs/server_c50m_deployment/cd_interrupted_recovery_startup_20261028.json`，各新目录另有 `recovery_preflight.json` 和 `startup_verified_20261028.json`。

网页 **PID43202**，地址仍为 `http://127.0.0.1:8765/`。已跟随本次C/D目录，训练历史截到各自实际保存断点，保留C86/D106个完整评估节点。WM只读接入其负责方发布的 `evidence/wm_training_resume_20261028.json`，核验新格式的源码哈希、任务/学习配置未变声明、自己的100007936动作/6104轮checkpoint哈希、完整恢复证明和当前进程身份；当前WM **PID23652**、输出 `server_shared_1024_eval_resume_20261028_01`，已实际继续更新，保留20条名义/DR评估记录。未修改或启动WM训练。网页22项/42子项测试及真实HTML/API检查通过，证据 `outputs/boya_training_dashboard_v1/interrupted_recovery_integration_20261028.json`；服务日志 `server_interrupted_recovery_20261028.log`。页面每15秒刷新，服务不占GPU。

用户另询问任务迁移。现场两卡算力利用率均约98–100%，GPU0显存仍有余量；空闲显存不能证明迁移会提高吞吐。本次未迁移任务或操作其他项目进程。诊断快照 `outputs/server_c50m_deployment/gpu_assignment_review_20261028.json`；当前占用和进度以后续实时查询为准。

## 2026-10-25：训练继续，修复WM网页进度不随原事件文件更新

用户询问训练和网页是否正常。2026-10-25T18:09:47.381090+08:00核实C监督41956正常续训，D监督41957处于850M例行随机化评估，WM33671实际训练至5265轮/86261760动作。三组进程均存活，C/D心跳正常且未报告错误。当前实时进度以API为准。

发现网页WM卡在80019456动作/4884轮并误显示“等待新训练日志”，原始训练TensorBoard和日志实际一直推进。原因是80M评估创建较新的 `physical_metrics_v2_80003072` 辅助事件文件后，目录级EventAccumulator跟踪移到新文件，忽略仍写入旧事件文件的训练writer。网页改为逐事件文件独立增量读取，只合并既有训练tag，同step按wall_time取最新；恢复点和名义/DR评估校验保持。真实事件文件回归覆盖“原writer开始→出现辅助writer→原writer继续追加”，22项网页CPU检查通过。

只重启网页47283→2027536，未重启或更改C/D/WM训练。8765的HTML与API均200，修复后WM显示训练中并跟上真实轮数，保留C70/D84个完整评估节点与WM16行名义/随机化评估。C78/D72个冻结源哈希均未变。证据 `outputs/boya_training_dashboard_v1/wm_event_reader_fix_20261025.json`；日志 `server_wm_event_reader_fix_20261025.log`。浏览器刷新原页面即可取得新数据。

## 2026-10-24：C/D 断点续训首更新验收通过，三组监控已恢复

用户在确认两组进程退出后明确要求恢复训练，并要求恢复 C、D、WM 三组网页监控。旧 C/D 日志与监督心跳最后均在10月23日17:28附近，未记录新的 traceback/OOM；当前环境10月24日12:12启动，具体中断原因未知。旧 status 虽仍写 training，原 PID 已退出，不能据此认为原训练存活。

新增 `scripts/resume_boya_interrupted.py`，只从自己的成对周期完整 learner 恢复，用新目录保留全部旧状态、日志及 checkpoint。C 为990248960动作/7555更新，D 为738197504动作/5632更新；分别重算未保存的12/2次更新。CPU验证模型、两类normalizer、Adam、调度器/随机状态字段、计数、原learner合同及来源哈希通过。22项恢复/原训练CPU检查通过，监督训练/评估循环与既有 OOM 恢复入口按 AST 验证一致，仅恢复引导改为周期断点。

北京时间12:43:40已启动 C 监督 **41956** / worker **42361**，新目录 `outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261024_v1/`；D 监督 **41957** / worker **42360**，新目录 `outputs/boya_d_fresh1p5b_server_interrupted_recovery_20261024_v1/`。原GPU1 UUID、16384环境、1.5B累计目标、每10M同492例/30秒评估、抓取/奖励/PPO/DR/引擎均保留；C评估wall21600秒、D7200秒沿用各自原值。首目标C1B/D740M。每个worker复用15141MiB连续3次显存准入，worker执行错误停止且不自动重试。78个C/72个D固定源运行中不得修改；本交接及网页不在冻结源中。**北京时间12:55首更新验收通过**：C7555→7556更新、990248960→990380032动作；D5632→5633更新、738197504→738328576动作。两组恢复初始模型/两类normalizer与各自断点逐张量相同，训练合同相同，首更新参数确实变化且数值有限，成对learner/policy匹配，Adam各推进1280步（C9670400→9671680，D7208960→7210240）。原记录和全部冻结源码哈希保持不变。启动验收 `outputs/server_c50m_deployment/cd_interrupted_recovery_startup_20261024.json`；实际worker命令逐项一致验收 `cd_interrupted_worker_contract_20261024.json`。两组均已实际续训，不要重复启动。预检证据为各新输出 `recovery_preflight.json`；说明 `docs/experiments/2026-10-24-interrupted/RECOVERY.md`。

8765网页已恢复，服务 **47283**，地址 `http://127.0.0.1:8765/`。已跟随上述新C/D目录，训练历史只保留到各自恢复断点，避免将未保存更新当作当前进度；旧完整C62/D73个DR节点保留。WM只读接入其项目新交接 `evidence/wm_training_resume_20261024.json`，校验其新输出 resume_verified、checkpoint哈希/身份/计数与已声明学习状态，当前WM进程 **33671** / 输出 `server_shared_1024_eval_resume_20261024_01`、恢复76152832动作/4648轮，实际已继续更新；保留7组名义/随机化共14行完整评估及分页。此处未修改或重启WM训练。

网页21项CPU检查、真实HTTP/HTML/API与历史数量验证通过；验收 `outputs/boya_training_dashboard_v1/interrupted_recovery_integration_20261024.json`。网页不导入GPU训练运行时、不占GPU。12:55真实API确认C/D均为训练中，WM也在继续更新，最新动作及节点数以上述新目录status/result与API为准，旧启动记录仅作历史。

## 2026-10-22 最新人工恢复：已从920M完整状态续训，首更新验证通过

用户本轮询问“C现在还能重启吗，显存是不是不够？”，按继续C训练的上下文执行一次人工恢复。09:26查询GPU1空闲 **17288MiB（16.88GiB）**；原成功16384环境容量测试记录全设备峰值 **10021MiB（9.79GiB）**，预计其他分配不变时启动后约余7267MiB。之前17GiB空闲并不等同无法运行；凌晨OOM是故障当时的分配失败，不应混为当前资源不足。

新入口 `scripts/resume_boya_oom.py`；新输出 `outputs/boya_c_fresh1p5b_server_oom_recovery_v1/`。监督PID **1516134**，930M训练worker **1516494**，物理GPU1/原UUID、16384环境。启动前连续3次、间隔5秒检查空闲≥15141MiB（历史10021MiB峰值+5120MiB余量），已通过；每个后续worker启动前同样检查，不足时保持带心跳的 `waiting_for_gpu_memory`，实际worker出错仍停止，不自动重试、不换卡、不改变环境数、不停止其他任务。显存准入不等于预留显存，其他任务后来增加分配仍可能导致OOM。

恢复源仍是原920M完整learner，SHA256 `76664c4a38cf698453fa653d1573aa8587648afcb0cd0e3b6c134dd3f43b0b2a`，920125440动作/7020updates。**复用已经完整验证的920M评估**，用symlink保留该完成阶段，不重复这次近3小时评估；直接执行下一930M目标（实际930086912动作），最终1.5B请求预算、每10M完整保存与492例/30秒DR评估、21600秒评估wall上限都不变。原失败目录/status/log/hash保持不变；原72个固定源不改，新恢复入口/CPU测试/说明新增3个，合计 **75个pin**，运行中不得修改。

32项CPU检查通过，包括仅接受零更新的既定OOM、权重来源、显存连续准入/等待可停止、监督循环除准入和复用完成阶段外AST一致、原训练循环、网页初始化历史保留。实际worker CLI逐项对比故障前相同，只有输出和审计路径变化。启动验收 `outputs/server_c50m_deployment/c930m_oom_recovery_startup_20261022.json`；预检在新输出 `recovery_preflight.json`；冻结说明 `docs/experiments/2026-10-22-c930m-oom/RECOVERY.md`。**09:47已完成首个新更新，7020→7021、920125440→920256512动作，完整learner/policy配对、数值有限、参数确实更新。恢复初始模型/两类normalizer与920M逐张量完全相同，合同完全一致，Adam8985600→8986880，未重置训练状态。** 首轮实测751.23动作/秒，按该临时速率930M训练剩余约13086秒（3.64小时），后续评估另计；首轮速度不代表长期稳定吞吐。09:47 GPU1总已用72779MiB、空闲24472MiB（23.90GiB），本C进程9700MiB（9.47GiB）。共享负载已发生变化，不沿用启动前17GiB的空闲快照判断现在资源。后续以status/验收报告为准。

网页8765已接新C目录，恢复验收时PID1518043；本日WM监控更新后当前网页PID **1946597**（仅网页服务重启，训练PID不变），C55个完整DR历史节点、D64个节点、WM及其分页均保留。另修复网页在恢复场景初始化期间误用临时新lineage、导致历史成绩消失的问题：从已知同组的resume权重记录取原lineage，未读任意外部路径或触碰训练。09:47网页C动作920256512与实际新更新相符，最新完整C评估仍920M，D约645.27M正常运行。网页、训练恢复和失败归档不可重复启动。

## 2026-10-22 最新故障：C920M已完成，930M场景初始化因CUDA显存不足停止

北京时间2026-10-22 **01:25:40**，C监督3405172在启动930M训练worker894766后记录 `training exited1; stop without retry` 并退出；两个进程均不再运行。最后完整动作 **920125440**、完整更新 **7020**；930M失败worker的动作、session动作、更新均为0，尚未完成场景构建/加载learner，更没有产生930M模型。当前错误目录 `outputs/boya_c_fresh1p5b_server_dr_eval_timeout_recovery_v1/actions_930000000/`，不要当作训练仍在运行，也不要直接重跑旧恢复命令覆盖已有目录。

直接原因已由 `training.log` 确认：PhysX先报 `CUDA error: out of memory`，无法分配 `mJacobianDataDev`（264241152字节）等缓冲，随后Warp也报OOM，最终异常 `Failed to allocate 88 bytes on device 'cuda:0'`。进程的cuda:0通过原GPU UUID绑定到物理GPU1，不表示使用了物理GPU0。这是共享GPU内存分配失败，不是上次840M评估的7200秒超时；具体占用来源没有证据，不归责于某个其他任务。09:05现场GPU1总已用79963MiB、空闲17288MiB，GPU0已用93880MiB；这是事后快照，不能代替故障瞬间的占用记录。

920M完整learner保存在 `actions_920000000/training/teacher_final.pth`，SHA256 `76664c4a38cf698453fa653d1573aa8587648afcb0cd0e3b6c134dd3f43b0b2a`。CPU审计验证910/920M完整checkpoint哈希、模型/两类normalizer/Adam有限、计数及自己的910→920恢复链全部通过，原72个固定源和安装后端哈希均未变。900/910/920M的492例30秒冻结DR评估均完整完成，墙钟分别3050.52/6115.76/10706.87秒；920M约2小时58分，在先前放宽的6小时上限内，说明该超时修复没有回退。审计 `outputs/server_c50m_deployment/c930m_oom_audit_20261022.json`。

本次按用户“C又出错了”执行CPU/日志/资源诊断并记录现场，未启动GPU重试，未改环境数、引擎、奖励、PPO、DR、缓存或冻结源，也未停止D、WM或其他任务。D在09:05约643.17M继续运行。恢复边界明确为自己的920M完整learner，下一目标930M；应先解决共享GPU可用显存与启动余量，保留旧失败现场并使用独立恢复输出，不能复用只接受840M评估超时的旧prepare入口。继续遵守执行错误停止报告、不自动重试。

## 2026-10-21最新授权与恢复：C840M补评估已启动，通过后自动从完整learner续训

用户在故障报告后明确要求“那你看下怎么解决？”，已按本次修复授权准备、验证并启动独立恢复监督；下方“未恢复/等待指令”是先前故障报告时状态。本次没有从零训练，没有停止D或其他用户任务，也没有修改原69个固定源、引擎、抓取、奖励、PPO或领域随机化配置。

恢复入口 `scripts/resume_boya_randomized_timeout.py`；新输出 `outputs/boya_c_fresh1p5b_server_dr_eval_timeout_recovery_v1/`，监督PID **3405172**，840M补评估worker **3405215**，绑定原GPU1 UUID。旧错误目录及其部分结果/权重全部原样保留，原错误status和失败result哈希已核验未变。新目录先以symlink引用原840040448动作/6409updates的完整训练记录，**从头补跑同一492例、30秒仿真窗口的固定配对评估，不拼接失败前缀**。评估校验及真实初态/属性配对通过后，自动恢复自己的840M完整learner，从850M（实际850001920动作）继续原1.5B预算/每10M保存及DR评估。16384环境、seed43、mini512/horizon8/epochs5及全部旧训练合同保持；段间缓存reset与原流程相同。

唯一运行限制变化：评估wall上限7200→21600秒（2→6小时）；这不是增加仿真窗口或修改算法。新监督run循环与旧冻结run逐AST比较，除读取该评估wall参数外完全一致。新增恢复入口/必要测试/本地运行说明3个源，合计72个pin；原69个源及安装后端不变。仍无自动重试；新的执行/源码/恢复/数值错误会停止汇报，不重复运行旧fresh或本目录命令。明确拒绝原监督/worker仍活着、非wall超时故障、异模型权重及已有恢复输出。

26项CPU测试通过。北京时间01:11验收：补评估推进到2.0/30秒，约93.65环境动作/秒，早期预计剩余2942秒（约49分钟，随共享负载变化）；初态误差全0、属性读回质量误差1.49e-8kg/其余0、normalizer更新0，无新错误。D约529.14M仍运行。此时尚未完成840M新评估，也尚未进入850M训练，不能将“已启动恢复监督”表述为新评估已通过或训练已恢复更新。后续以新目录status及result为准，不重复启动。

网页8765已切换到新C目录，PID **3408792**（旧网页3391054已退出）；C此前46个完整DR节点和原训练历史全部保留，失败840M不作为完整成绩。监控从同lineage各历史目录合并有效评估，仅接纳对应训练checkpoint身份相符的完整结果；分页和WM功能保留。网页重启未触碰训练进程。启动验收报告 `outputs/server_c50m_deployment/c840m_timeout_recovery_startup_20261021.json`；预检 `outputs/boya_c_fresh1p5b_server_dr_eval_timeout_recovery_v1/recovery_preflight.json`；运行说明 `docs/experiments/2026-10-21-c840m-timeout/RECOVERY.md`。本地运行说明和72个pin文件在运行中不得改写。

## 2026-10-21：C840M训练已完成，配对随机化评估超时后监督停止

北京时间2026-10-21 00:22:49，C监督3265832因840M评估进程3239548退出1而停止；两进程均已退出，不要把旧启动记录当作正在运行，也不要自动重试。当前目录 `outputs/boya_c_fresh1p5b_server_dr_eval_v1/actions_840000000/`。训练正常完成840040448动作、6409updates，`training/teacher_final.pth` 与 `teacher_final_policy.pth` 均完整保存；未丢失840M训练进度。

直接错误是 `Evaluation incomplete due to wall budget; preserve partial results`：评估达到7200秒墙钟上限，实际7221.91秒时仅执行17.1/30仿真秒。本次评估约23.37环境动作/秒，830M完整评估曾用3681.06秒；吞吐下降的具体原因尚未定位，不能直接认定为他人占卡、显存不足或引擎错误。840M的部分cases/summary/result保留，但status=error、wall_budget_exhausted=true，不作为完整成绩；最后有效自动评估仍为830078976动作。D监督仍运行，未停止其他项目任务。

已复用现有CPU校验器验证840M learner/policy配对、模型及normalizers/Adam有限、物理/奖励合同和自身830M恢复链全部通过；69个固定源及安装后端哈希未变。评估初态误差0、属性读回最大质量误差1.49e-8kg、normalizer更新0。完整learner SHA256 `b9940c244a97ec3e8c5b401b7a49b1f75a882da59df9a9877c6dd62882a8fdad`；policy SHA256 `b55ce0f07ee46c0aeb5ff1c9b02c5f91fa37f43caac4c9a438ee4a7203ad20e4`。诊断证据 `outputs/server_c50m_deployment/c840m_timeout_audit_20261021.json`。

本次仅按用户“好像C出错了”进行只读诊断和现场记录，没有启动恢复、补评估或修改时间上限。继续遵守执行错误先汇报、不自动重试。恢复边界是自己的840M完整learner；须保留原失败目录，在独立输出中完成同一492例/30秒配对评估并验证，再继续850M以后的训练，不重新从零训练、不重复计算已完成动作。恢复实施和评估墙钟预算调整待用户后续指令，不改抓取、物理引擎、奖励、PPO或DR范围。

## 2026-10-16：只读在线监控网页，C/D同训练量及WM定期冻结评估已接入

已更新独立监控 `scripts/serve_boya_training_dashboard.py`，当前PID **3391054**（此前网页PID3297001/3310650已退出），本机 `http://127.0.0.1:8765/`，后台日志和启动信息 `outputs/boya_training_dashboard_v1/`。在VS Code“端口”面板转发8765即可从浏览器访问，原页面刷新即可。启动命令用本仓库 `.venv/bin/python`（需已有CPU TensorBoard读取器）；`CUDA_VISIBLE_DEVICES` 为空。该服务仅GET读取既定任务的JSON/TensorBoard日志，不导入Torch/Isaac/TensorFlow、不占GPU、不启动评估或训练、不浏览任意文件。状态每15秒自动刷新；C/D及WM的完整节点评估自动显示。重启前核对PID和监听端口，不能重复占用。

网页包含C/D实际进度、最近5次完整更新的吞吐、GPU1共享状态；按lineage筛选并去重的连续训练回报/吞吐。**评估默认按组分开，训练曲线默认显示0至两组共同已完成训练量，使用C早期历史与D当前过程。** 可选各自完整走势、全量总览；“仅相同训练量对照”只保留C/D实际累计动作完全相同的随机化评估节点，当前无共同节点，不能将C410M与刚起步的D作为同训练预算对照，也不将C410M续训点移为零。C70M/230M/300M是独立补测，410M起每10M自动DR；D从零且首10M前等待。固定相同492例的净角/P10/中位/九尺寸等权、30s保持率和尺寸分组保留；拒绝名义、未完成、不同fixture或其他lineage结果。两组回报包含不同掉落项，能力对照还须看相同预算下的物理指标。

**评估节点记录已拆成C、D各自独立的全宽表格**，明确各组掉落扣分与从零累计口径；选中节点和九尺寸详情互不覆盖。“全部节点总览”仅合并上方曲线，节点表仍分组；“仅相同训练量对照”才筛选两组实际动作完全相同的节点，无共同节点时两表明确显示暂无、清空过期尺寸详情。此次仅修改网页HTML及本交接，服务每次读取HTML，原页面刷新即可，无须重启监督或训练。真实API快照C447.087M/D51.380M、完整评估C7/D5节点、共同节点0；实际内联JS运行及独立点击、刷新保留选择、各指标、空节点/合成同预算筛选、WM显示验证通过，无浏览器视觉验收。证据 `outputs/boya_training_dashboard_v1/cd_separate_records_verification.json`；快照不是实时进度。

**C/D节点表新增独立分页**：默认各10条、最新节点在前，可选20/50条；各有最新/上一页/下一页、页码及记录范围。自动刷新保留各自页码和当前页内选中节点，切换比较口径或页大小回到第一页；尺寸详情只对应本组当前页的节点。真实API及测试内25/23条多页记录的翻页边界、独立选择、刷新、新增节点、空筛选与WM显示验证通过，HTTP确认生效，无浏览器视觉验收。仅改网页与本交接，未重启训练或服务；证据 `outputs/boya_training_dashboard_v1/cd_pagination_verification.json`。

**2026-10-22：WM网页接入对方agent的70M超时恢复。** 用户报告已由WM agent恢复训练，网页仍红色。核实旧WM3382697已在70M/4273轮的DR seed1702评估达到7200秒上限后退出；原网页仍只跟随10月16日attachment，因此显示旧错误。对方新交接 `/home/wangrenpeng/boya-inhand-wm/evidence/wm_evaluation_timeout_recovery_20261022.json` 已发布，新进程 **1895909**，新输出 `outputs/server_shared_1024_eval_resume_20261022_01/`，从70008832动作/4273轮完整状态恢复。网页现校验predecessor、GPU1/1024环境/原329318400预算、resume_verified的确切权重SHA与模型身份/完整状态证明、checkpoint文件实际哈希，并合并旧训练历史到恢复点及之后的新日志。最新网页已显示 **训练中、70.09M、无当前错误**，下一80M自动评估。

WM完整评估历史按各次恢复边界及当时协议验证后合并；仍保留7个完整节点（名义/DR共14行）、各自分页及逐物体详情，失败70M不冒充有效成绩。旧70M超时在评估状态显示为1个历史未完成评估，不再覆盖新进程状态。新节点只跟随对方声明的运行超时21600秒，其余20秒窗口、固定种子、重力与科学指标保持严格检查；未泛化放宽协议或混入不同任务。19项CPU测试、真实读取、HTTP/HTML及C75/D69固定源审计通过；服务未导入Torch/TensorFlow、不占GPU。当前网页PID1946597（旧1518043已退出），日志 `outputs/boya_training_dashboard_v1/server_wm_recovery_20261022.log`，验收 `wm_recovery_integration_20261022.json`。**本次未修改WM源码、配置、权重、输出证据或进程，也未重启WM/C/D训练；只改本仓库监控和交接。** 无浏览器视觉验收，当前数据以API为准。

**2026-10-21：WM节点表拆分并独立分页。** 按用户要求，将原混排的WM评估节点拆为“名义评估”和“领域随机化评估”两张全宽表：名义为基准物理条件31例，随机化为3个固定种子共93例；两者沿用相同指标和20秒窗口。各表默认最新10条，可选20/50条，独立最新/上一页/下一页、页码、当前节点、逐30资产/31构型详情及分组选择；自动刷新保留各自页码和页内选择，清空结果时移除过期详情。加权方式继续由原全局选择器控制，曲线读取全部完整节点。此次仅改 `assets/dashboard/boya_training.html` 与本交接，原页面刷新即可生效，未重启网页、训练或评估。真实API两类各7节点及合成25/23节点的翻页、边界、选择、刷新、新增记录、单类/空结果、加权切换和C/D状态独立性验证通过，HTTP确认返回新HTML；无浏览器视觉验收。C恢复72个及D69个固定源哈希均未变。证据 `outputs/boya_training_dashboard_v1/wm_split_pagination_verification.json`；7节点为验证快照，当前数量以网页为准。

**BOTYARD WM已只读接入同一网页，并跟随其定期评估交接**，独立卡片、训练及评估曲线，不并入C/D奖励或评估。原启动来源 `/home/wangrenpeng/boya-inhand-wm/evidence/wm_server_1024_launch_20261016.json`；新增读取该项目 `evidence/wm_periodic_evaluation_attachment_20261016.json`。WM自身监督完成100轮/1638400动作完整学习状态恢复后已关闭旧进程3287213，新worker **3382697**，当前输出 `/home/wangrenpeng/boya-inhand-wm/outputs/server_shared_1024_eval_20261016_01/`。网页只有在resume_verified与旧身份/完整checkpoint哈希/epoch/frame一致且exact_learning_state_comparison=true后才选择新目录；合并旧历史至恢复轮数及之后新日志。保留GPU1、1024环境、horizon16、20100轮/329318400交互、31构型、物理DR。原生日志performance/info使用本轮开始frame，完成动作按epoch×1024×16；reward/WM loss按完成epoch关联，不显示未完成下一轮损失。旧TB最后刷新99轮，而验证checkpoint为100轮；总进度下限取已验证100轮，曲线不编造缺失日志点。这里未修改WM项目源码、配置或进程。

**WM定期冻结评估已在网页启用**：读取 `periodic_evaluation.json`、`evaluations/*/snapshot.json` / `summary.json` 和子任务result/progress/cases；每10M，名义seed1700×31构型，加随机化seed1701/1702/1703×31，共124例，单例最多20s，训练预算不计评估动作。显示四子任务进度、名义/DR各自净角和保持率曲线、持续正转率（完整20s且每1s净正转≥0.05rad）、实际有效时长及逐30资产/31构型结果。默认先构型内等权再30资产等权，可切31构型等权；弧度转为度。必须全部124例完成、checkpoint/训练身份/协议一致、模型及normalizer冻结、优化步0、保留早失败、后续节点配对初态验证通过后才显示成绩；拒绝未完成或不符记录，不混入接口测试。WM20s与C/D30s分开展示。北京时间12:48仍在1.6384M接入基线评估、下一常规10M，尚无完整WM成绩；后续自动跟随结果。

CPU13项口径测试、JS实际待完成状态及合成完整结果夹具的全部视图执行、脚本语法、真实HTTP/只读边界通过；合成夹具仅用于测试，不发布到页面，无浏览器视觉验收。运行时未导入Torch/TensorFlow；C/D各69个固定源哈希未变，两组均继续训练。最新证据 `outputs/boya_training_dashboard_v1/wm_periodic_evaluation_verification.json`；此前WM训练监控接入证据 `wm_integration_verification.json` 保留。北京时间12:48快照：C425.198M、D20.709M、WM1.6384M冻结评估中，未报告执行错误；当前状态以网页/API为准。

自动查询Coder外部直达链接的认证方式被自动审批拒绝：它会读取另一进程的认证令牌，超出网页请求的授权。未执行该读取，不重试或绕过；使用VS Code端口转发。需要远程地址时沿用用户现有工作区访问方式，不从其他进程取凭据。

## 2026-10-16 最新授权：C/D每10M改为领域随机化评估，D从零且取消掉落扣分，两组均用GPU1

用户明确新增 **D = C仅将显式掉落成本16改为0**，从零训练、16384环境、seed43/minibatch512/horizon8/epochs5、1.5B累计动作、原164缓存与完整DR；不是历史A。随后用户说明GPU0别人要用，故**C、D训练及评估全部绑定物理GPU1**，本次没有启动任何新GPU0任务。共享算力会降低各自速度，不操作其他项目进程。

**C已经在410,124,288动作的完整learner边界切换监督。** 原监督2701917按本次授权正常退出（旧status=stopped、Stop requested before worker launch是主动切换，不是训练异常）；410M原名义评估已自然完成，权重及历史保留。新监督 **3265832**，新输出 `outputs/boya_c_fresh1p5b_server_dr_eval_v1/`，410M的492例随机化评估已完成并自动续训420M段，随后继续至1.5B。不要恢复旧监督或重复启动。切换审计 `outputs/server_c50m_deployment/dr_eval_transition_20261016.json`。

D独立64环境2updates/1024动作短训练已完成；CPU共11项测试通过，确认同轨迹C/D仅在原失败9/10控制相差16、其他奖励逐项相同，原PPO训练循环AST一致，完整恢复链与不完整评估拒绝检查通过。D九尺寸9环境1s随机化入口验证已完成、属性与参考完全一致，初态最大差4.77e-7（容差1e-6）；正式D监督 **3267674** 已启动，worker3267685在GPU1以16384环境训练。输出 `outputs/boya_d_fresh1p5b_server_dr_eval_v1/`；不重复启动，不将短测计入正式动作，首更新健康状态以该目录status/result及fresh_first_update_verified.json为准。

**D正式首更新已验证健康**：从零动作/updates/空Adam，131072动作/update1/Adam1280步，模型有限、变化且完整learner/policy配对；初始网络及两种normalizers与原C fresh seed43逐张量完全相同。C410M新492例DR评估已完成：平均532.01°、30s保持417/492=84.76%，九尺寸等权466.83°，初态及属性与70M参考差0；现已自动进入410→420M，worker **3275940** 正常更新，完整合同与410M逐项相同、恢复初始网络/normalizers完全相同、Adam4005120→4006400，首更新有限、变化且配对。生产检查与后续状态记录 `docs/experiments/2026-10-16-d-no-drop/RUN_STATUS.md`。

**北京时间09:54:54 双训练测量**：C410.780M（3134updates），D4.063M（31updates）；最近3次实际同卡更新C2621动作/s、D2492动作/s，合计5113动作/s。GPU1显存used37913MiB、free59338MiB，GPU0无本次任务。启动报告 `outputs/server_c50m_deployment/cd_randomized_startup_report_20261016.json` / `.md`。以这段早期吞吐、后续每次约9min初始化和16.7～35min的492例评估估计，C剩余6.7～8.1天、D9.6～11.5天；没有完整双任务周期测量，共享负载及C完成后释放算力会改变ETA，不当精确承诺。最新动作数和错误只读各新目录status及current_run，不从本文件的历史快照判断。

新入口 `scripts/run_boya_randomized_milestones.py`；C原trainer/物理/奖励及原55冻结源不改，D独立trainer通过原PPO、继承C环境及旋转/其他奖励，只用零终止成本信号。新C/D随机化评估器复用昨晚 `outputs/boya_c_paired_dr_eval_20261016_gpu0_v3/` 的 **cases.json和stimuli.npz原字节**：164训练抓取×3新ID参数/噪声/外力条件，492环境，每例最长30s，首失败后保留有效前缀，不外推、不重置。每10M保存成对完整learner/policy→随机化评估→下一训练段，替代后续名义评估；同时报告缓存加权、九尺寸等权和对70M配对比较，初态与实际属性回读必须一致。当前69个联合源/配置身份冻结；协议 `docs/experiments/2026-10-16-d-no-drop/PROTOCOL.md`。普通物理失败正常统计，执行/源/数值/恢复错误停报、不自动重试。后面的“仅C/GPU0/名义评估”等是历史，以上为最新授权。

## 2026-10-16：用户授权GPU0独立配对随机化评估，GPU1正式训练继续

**独立评估已于北京时间01:59:45全部完成，耗时44.6分钟，原监督PID3069800已退出，不恢复或重复启动。** 三模型各492个ID案例；70M/230M/300M平均预算内净角783.82°/575.88°/595.99°，完整观察30s比例61.38%/75.81%/82.52%。提前物理失败保留有效前缀，不外推；保持到30s不等同持续旋转成功。模型间初态及实际属性回读差0，执行错误0、基础设施未完成0。最终 `comparison.json` / `.md` 已生成；GPU0评估进程已退出。2026-10-16 08:33正式GPU1训练约400.425M、正运行400→410M，无执行错误；400M名义评估30s净949.11°。其余启动细节以下保留为历史。

独立监督PID **3069800**，输出 `outputs/boya_c_paired_dr_eval_20261016_gpu0_v3/`，状态以该目录status.json及当前worker result.json为准，不重复启动。新增独立评估入口/监督、配对数据统计模块及CPU测试，原55个冻结源/配置保持。先九尺寸9环境1s验证，随后70M→230M→300M，每模型492环境=164训练缓存抓取×3新随机条件、最长30s或原物理失败终止；manifest、噪声/外力及模型已固定哈希。CPU6项测试、成对policy/full learner与源码身份核验通过。完成前不报告随机化能力结论。

物理GPU0已出现另一项目抓取验证进程，显存仍充足；按用户本次明确卡号选择记录共享负载运行评估，不操作其他项目进程。GPU1原正式续训监督PID2701917继续，1.5B预算与每10M名义评估不变。本次启动细节和边界记录在本地 `docs/experiments/2026-10-16-randomized-evaluation/RUN_STATUS.md`，实时进度不以此交接中的历史动作数判断。


## 最新状态：2026-10-15，代码已推送，210M断点续训首更新验证通过

**已按用户要求先push再恢复。** 远端main已确认提交 `13c3aa94edd5d9ebb6efbe5513a9793b30500ef0`，随后北京时间13:32:44后台启动独立恢复监督PID **2701917**，训练worker PID **2701974**。新输出 `outputs/boya_c_fresh1p5b_server_recovery_sync_v1/`，GPU1、16384环境，已完成场景构建及完整断点恢复，首个1604更新验证通过；模型/normalizers初值与210M精确一致，Adam2051840→2053120，learner/policy配对有限。不要重复启动。原正式训练在210,108,416动作/1603updates（14.01%）完整保存后因210M评估初始化错误停止，原目录和失败记录保留。最新运行状态只读新目录status/current_run/training result；旧主PID2227265已退出，旧status=error是历史记录。

恢复启动报告：`outputs/server_c50m_deployment/c1p5b_recovery_startup_report.json`；首更新证据 `outputs/server_c50m_deployment/c1p5b_recovery_first_update_verified.json`。截至13:44约210.764M/1608updates，早期完整吞吐约3734动作/s，自身显存约9770MiB；按当前共享负载及后续128次初始化+129次评估估计剩余约138.7小时，约10月21日 08:26（北京时间，负载变化会改ETA）。保持1.5B预算/每10M评估。55个原冻结+恢复源码联合身份已验证，安装引擎源未修改。本段是本地启动后最新记录；Git发布版本为13c3aa9。

错误：PhysX `setCMassLocalPose pose is not valid`，随后名义50g质量读回误差约0.05kg；根因待独立诊断，不推定为策略NaN或显存OOM。完整learner/policy哈希、有限性、配对、Adam步数2051840及52个冻结源/配置联合身份均已CPU核验通过。最后有效评估仍是200M（30秒、1238.48°），210M无有效评估。

证据：`outputs/server_c50m_deployment/c1p5b_210m_failure_audit.json` / `.md`；恢复源是 `outputs/boya_c_fresh1p5b_server_v1/actions_210000000/training/teacher_final.pth`。保留全部权重和失败记录；不得重新运行原fresh命令覆盖或重新训练。遵守执行错误停止汇报、不自动重试约束；人工诊断/恢复另行记录授权和验证结果。原引擎、抓取、算法、奖励、DR和冻结协议均未修改。


用户已授权：在服务器空闲 GPU 上测试合适的并行环境数，接入下面的 C 奖励，从零训练 **1.5B（1,500,000,000）累计环境动作**，每 10M 自动冻结评估，共 150 个节点。原 50M 是早期测试预算，用户已在 2026-10-14 明确改为参考原 Hora 正式上限。本次仅在服务器选定 GPU 1 上运行。此文件是明确要求公开的交接说明；`docs/`、`AGENTS.md`、权重和训练日志仍不发布。

### 2026-10-15 最新授权：先 push，再从210M恢复

用户已明确要求“先push再恢复”。**不再等待新的启动确认**；先确认本次恢复代码已经推送，再启动独立续训目录 `outputs/boya_c_fresh1p5b_server_recovery_sync_v1/`。本文件以下停机/待审阅描述是推送前历史；实际启动状态以新目录 `status.json`、当前训练 `result.json` 和PID为准。不得重跑旧fresh入口。

正式修复不改安装文件：新增 `tendonspin/physics/physx_property_sync.py` 和 `scripts/run_boya_with_property_sync.py`，对精确SHA固定的安装版PhysX，仅在质量/COM/环境ID的现有D2H copy完成后让CPU消费者继续；其他GPU/求解器操作保持异步。旧52个冻结源码/配置及完整learner合同原样保留，兼容入口3个新文件另存哈希与运行审计，不隐瞒运行时差异。

新增 `scripts/resume_boya_spin_progress_milestones.py`：核验旧节点和自己的210M完整learner，保存原失败记录，用新目录引用已验证的210M诊断评估并记录显式接纳，再继续220M～1500M共129段。每10M原名义30s评估、16384环境、minibatch512、seed43、C奖励、DR、抓取和原PhysX不变；执行/源/恢复/数值错误仍停报，不自动重试。

推送前验证：相关CPU测试11项通过；新生产入口在独立64环境配置副本上完成2updates/1024动作且模型更新、属性回读正确。完整只读历史/恢复预检 `outputs/boya_c_recovery_preflight_sync_v1/` 为prepared，21个节点校验完成；`--prepare-only` 不启动正式训练。独立小验证不计入正式1.5B。诊断候选三行安装文件patch继续保留但不应用，正式路线采用上述有来源审计的进程内同步。

### 2026-10-15 人工诊断结果：正式训练仍暂停

用户已明确允许开始独立诊断。已在隔离小场景复现安装版 PhysX 适配层的异步 GPU→pinned CPU 属性回读问题：原 `set_masses_index` / `set_coms_index` / 环境ID回读在 CPU 消费前缺少等待；冷缓冲实验产生0质量及与正式日志一致的无效COM姿态报错。原错误未保存实际缓冲内容，不将实验零质量当作当时直接观测。完整报告：`outputs/server_c50m_deployment/c1p5b_210m_diagnosis_report.md` / `.json`。

已生成未应用的三行候选 `outputs/server_c50m_deployment/isaaclab_physx_property_host_readback_sync_candidate.patch`，仅在现有D2H copy后等待stream。临时进程验证通过：单环境80次属性设置及冷缓冲、64环境80次属性设置和200次原DR reset均正确。52个冻结源/配置及安装文件原始哈希复核一致；**未修改安装环境、冻结代码、奖励、抓取、PPO或物理参数**。

使用进程内临时等待的独立210M冻结评估已完成：30秒未掉落，net **1345.4828°（3.7375圈）**，比200M约+8.64%，仍低于70M最佳1686.1693°。名义50g/摩擦0.5/COM0读回正确，与200M初态各项误差0。输出 `outputs/server_c50m_deployment/physx_sync_eval210m_20261015_v1/`，含同步wrapper来源；**尚未计入正式210M节点，原失败评估未覆盖**。

原64环境未同步的100次实际DR reset小样本也通过；历史训练并无逐次真实属性回读，因此不能排除或量化过去偶发错误，也不能将后期名义速度回退直接归因于此。200M有84.17%控制步达到0.5rad/s正向奖励上限；奖励分解提示速度与优化目标可能存在取舍，但未证明因果。

恢复草案 `outputs/server_c50m_deployment/c1p5b_210m_recovery_plan_draft.json` 已保存，未执行。安装适配层兼容补丁需明确审阅记录；随后只允许从自己的210M完整learner续训，使用新的continuation监督输出，先明确接纳210M独立评估，再从220M（实际220069888）继续剩余129个节点。原主入口fresh-only，不得对旧目录重跑。当前没有正式训练进程。

## 服务器接手进度：2026-10-14，已在 tmux 中恢复

用户已明确要求恢复工作，取消六档/20 分钟限制。最新偏好是**全程总耗时最短，计入每段初始化和评估；显存只须留余量，不要求接近满载**。用户已明确选择正式 **C1.5B** 上限，并在预算问题中确认其余设置保持，包括 minibatch512 和每10M自动评估；准备与验证完成后直接从零启动，无须重复确认。容量测试已结束，最终选择 **16384 环境**；正式配置与独立冻结协议已冻结。**正式任务已后台启动并完成健康首更新**，主监督 PID2227265、阶段PID2227276、训练PID2227277，启动北京时间2026-10-14 16:33:13。不得重复启动。输出 `outputs/boya_c_fresh1p5b_server_v1/`，启动记录 `outputs/server_c50m_deployment/c1p5b_formal_launch.json`；恢复时先读正式status/stage/training result并核对PID，不改冻结源码/配置/协议。首更新及policy/learner配对验证通过：零动作/updates、空Adam，131072动作/update1/Adam1280步；模型有限且变化。健康证据 `outputs/server_c50m_deployment/c1p5b_first_update_verified.json`。本次观测累计1179648动作/9updates，预热后早期完整速度5082动作/s；启动报告 `outputs/server_c50m_deployment/c1p5b_startup_report.json`，ETA约4.4–5.3天。正式训练与150节点评估仍在运行，并未完成1.5B。

### 已完成部署与验证

- 仓库 `/home/wangrenpeng/tendon-spin` 已按用户要求 pull，HEAD `e652db81449245588273470b4d9f725fb4e7e169`。服务器 C 代码已整理为独立 Git 版本；提交与推送状态以 Git 和 `outputs/server_c50m_deployment/` 的发布记录为准。保留工作区与全部训练输出，勿 reset/clean。旧 B 配置、抓取、物理引擎和 Hora PPO 没有改写。
- 缓存已取得并校验：SHA256 `6cb72539fc3c40d1e75a49983019c9b4be679a37bd5d8dcef406c06696aaf993`，164 状态，九组 `[3,1,5,10,12,30,20,41,42]`。无需再寻找或筛选缓存。
- GPU 1 为 RTX PRO 6000 Blackwell Server Edition，97,887 MiB，UUID `GPU-a3f9cc58-fe6f-96d3-ab1a-e1dc33b09e7a`；驱动 580.159.03、CUDA 13.0。绑定 `CUDA_VISIBLE_DEVICES=1`，Isaac/PPO 逻辑 cuda:0 实际均用 GPU 1。GPU 0 有他人任务，不干扰。
- 独立 `.venv` 已部署：Python 3.12.14、torch 2.11.0+cu130、Isaac Sim 6.1.0.0、Isaac Lab 3.0.0rc1、Warp 1.16.0、NumPy 2.3.1、SciPy 1.17.0、packaging 26.0、newton-usd-schemas 0.4.1。完整库存 `outputs/server_c50m_deployment/installed_environment.json`。未改共享 Sim 或系统 Python。
- Isaac Lab 官方安装 overrides 固定 Newton 1.5.2，与 Sim core 的 Newton 1.5.0 元数据存在一项已报告且按官方覆盖解决的依赖差异；**实际仿真后端经运行时类验证为 isaaclab_physx**，未选择 Newton。官方 wheel 与 overrides 保存在 `runtime/` 和 deployment 目录。
- 本机 import probe 完成，结构检查通过：22 joints 和 mimic 保留。USD `outputs/isaac_boya_import/boya/boya.usda`，SHA256 `4e4914a1aade2661cf33b13ed75bcbd6e77f1ff4651b2c61d1ca974f49f7cfa7`；导入身份 `docs/data/isaac_boya_import_phases.json`。
- C 原生 trainer/eval 已接入同一 `SpinProgressBoyaEnv.control_reward()`。控制开始 begin、每物理步 advance、结束 compute_reward；C 正向封顶、反向不截断、失败扣 16 一次。B 共享边界有行为一致性测试，原物理/DR/reset/PPO 不变。
- CPU 奖励/原生控制/初始化/resume/预算测试通过。真实 GPU 短训 64 环境、4 updates/2048 动作，首更新有限且参数变化；缓存、名义初态、随机化属性读回一致。独立姿态重算与原生 C 旋转项最大误差 `4.94e-8`。短冻结评估在 0.25s 掉落，属于未训练策略验证，不代表效果。自身完整 learner 恢复到第 5 update 成功，成对 policy/full learner 权重一致。
- 验证汇总 `outputs/server_c50m_deployment/short_validation_passed.json`；短训/评估/resume 在 `outputs/boya_c_short_server_v1/`，动作不计正式 1.5B。

### 容量与全程效率测试

原测试 `outputs/boya_c_capacity_server_v1/` 和扩展测试 `outputs/boya_c_capacity_extended_server_v1/` 都保留：

| 环境数 | 完整训练吞吐，动作/s | 峰值显存，MiB | 每段初始化及退出，秒 |
| --- | ---: | ---: | ---: |
| 1024 | 741 | 4001 | 36 |
| 2048 | 1370 | 4421 | 67 |
| 4096 | 2309 | 5173 | 128 |
| 8192 | 3843 | 6841 | 257 |
| 16384 | 5169 | 10021 | 532 |
| 24576 | 无效：材料数量上限 | 11487 | 未完成训练 |
| 32768 | 无效：材料数量上限 | — | 未完成训练 |

32768 档在初始化时多次报 `PhysX error: PxPhysics::createMaterial: limit of 64K materials reached`，随后 SIGSEGV（退出 -11），未完成 PPO update。**不是显存 OOM；已先向用户汇报，原测试及更大队列已停止。** 不重试该档，不修改引擎/材料/抓取/DR/PPO 去绕过。所有较小成功档日志无此错误。

补测低于失败档的 **24576 已结束**，同样触发材料上限并退出 -11，未完成 PPO 更新。补充测试已选择 **16384**，复用成功五档的全程效率测量：

- 输出 `outputs/boya_c_capacity_total_time_server_v1/benchmark.json`，启动记录 `outputs/server_c50m_deployment/capacity_total_time_launch.json`，日志 `capacity_total_time.log`；phase=completed，旧 PID 2200288 已退出；不恢复容量测试。
- 每档 fresh 2 updates 预热+8 updates测量，minibatch512/horizon8/mini-epochs5保持。记录采样、PPO、日志/保存开销及初始化/退出；没有资源 watchdog/cgroup/自动内存停机。
- 固定测试协议 `docs/experiments/2026-10-14-server-c50m/CAPACITY_PROTOCOL.md` 保留不改；其中早期显存/50M选择偏好已由本节最新用户授权覆盖。
- 补测命令的 `--target-actions 50000000` 是预算确认前的比较预览，**不能当正式预算**。已以1.5B/150次初始化重算并记录 `outputs/server_c50m_deployment/final_c1p5b_selection.json`；16384训练+初始化约102.8小时，另加评估，初步 ETA4.3–5.3天。
- 最终按 `实际向上取整动作/实测吞吐 + 150×实测每段初始化退出` 比较。冻结单环境30s评估是各环境数共有的额外开销，加入 ETA；不能只选最高稳态 FPS。预留4894MiB显存。
- `benchmark_boya_spin_progress.py` 已增加同运行时/源/DR/cache身份验证后的测量复用、动态档数/时限、实时资源记录、总用时选择及64K材料错误检测。该错误会判为容量失败并停止更大档，不计有效吞吐。

### 正式训练运行与后续接手

1. 容量测试已完成，原测试和临时渲染进程已退出；不要重复启动或恢复测试权重。
2. C配置已固定16384/1500000000，独立冻结 `docs/experiments/2026-10-14-server-c1p5b/PROTOCOL.md` 已生成；正式训练运行中不改源码、C配置或该协议；旧B保留。
3. 已后台启动 fresh seed43 `scripts/run_boya_spin_progress_milestones.py --num-envs 16384 --out outputs/boya_c_fresh1p5b_server_v1 --scene-config assets/config/boya_hora_randomized_c_scene.json --randomization-config assets/config/boya_hora_randomized_c.json --protocol docs/experiments/2026-10-14-server-c1p5b/PROTOCOL.md`，绑定 GPU 1、可靠后台运行。禁止用测试/历史权重初始化。
4. 首更新有限/模型变化/从零身份与完整 learner/policy 成对保存已确认；启动报告记录环境数、实测吞吐/显存、PID/输出、按 batch=num_envs×8 向上取整的实际 1.5B 动作与 update 数、ETA。每 10M 自动冻结名义原状态 30s 或失败即停评估；普通掉落继续，执行/源/resume/数值错误停止，1.5B 后停止。

新增 C 文件为 `isaac_spin_progress.py`、C config/scene、`train_boya_spin_progress.py`、`evaluate_boya_spin_progress.py`、`run_boya_spin_progress_stage.py`、`run_boya_spin_progress_milestones.py`、`spin_progress_milestone_checks.py`、`benchmark_boya_spin_progress.py` 及两份必要测试。所有权重/完整日志仍在 ignored `outputs/`，保留原历史。工具普通沙箱目前报 `bwrap: Creating new namespace failed`，经自动审批使用沙箱外命令；这是执行环境问题，不是更换仿真路线的理由。

### 部署审计与最新节点评估

用户要求发布服务器代码版本，范围为C代码、配置、必要测试及本交接；164缓存此前已在Git。`.venv/`、`runtime/`、本地协议、权重、完整日志、审计输出和demo视频不纳入本次提交。

对照交接与原B物理/DR/PPO审计未发现额外部署遗漏：52个冻结源/配置哈希未变，实际生成USD保留28碰撞排除；30M的30100物理样本全部有限，四个distal独立耦合误差最大1.51e-5rad，请求/转发力矩差0。C奖励独立重算最大误差3.48e-7，失败-16一次。10/20/30M完整learner与policy及normalizers一致，Adam计数/学习率和自身恢复链验证通过。证据 `outputs/server_c50m_deployment/c1p5b_deployment_audit.md`。

原有假设继续保留：电机armature尚未实测，采用A=4*D_effective*dt；质量reset不同比重算名义50g转动惯量；冻结名义评估关闭物理DR，但控制后仍有固定seed43的±0.02rad关节观测噪声。不要将它描述为零输入噪声评估或实机参数已标定。

30M名义冻结评估保持15.05秒，signed net+455.37°，随后工作区掉落；未达到完整30秒。正式任务已进入30→40M阶段，主监督PID2227265不变。接手时先读正式status及current_run的training/result，不根据本交接的历史动作数判断当前进度，不重复启动。

### 临时部署视频

已生成 `outputs/boya_c_deployment_demo_tmp/rendered/isaac_native.mp4`，4秒/20fps，斜侧+俯视，RTX原生渲染短验证真实轨迹；2048动作策略在0.25s掉落，用慢放和定格查看部署构型，不代表正式训练效果。`render.json` 记录完成状态、来源哈希及零新增物理/训练动作；已向用户提供文件链接。看完可删整个临时目录，保留原短验证记录。

## 环境与训练配置

| 项目 | 当前本地配置，服务器应优先保持 |
| --- | --- |
| 仿真版本 | Isaac Sim **6.1.0.0**；Isaac Lab **3.0.0rc1** |
| Python / PyTorch | **3.12.14 / 2.11.0** |
| 后端 / 求解器 | `isaaclab_physx` / GPU PhysX，`original_tgs16_4`；不是 `isaaclab_ov` |
| 碰撞 | 现有 CAD convex、自碰撞和 28 项排除；contact offset 0.1mm、CCD 关闭；无支撑地面 |
| 时间 | physics dt 0.0005s；100 物理步/控制；20Hz |
| 执行器 | 五指全部开放：TH4 + FF3 + MF3 + RF3 + LF3 = **16 动作**；双腕固定、四个 distal mimic 被动；现有外部 PD |
| 观测 / 算法 | 特权教师：96 维观测、9 维真实物理属性；Hora PPO、horizon 8、mini-epochs 5、minibatch 512、seed 43 |
| 物体 / 构型 | 名义白色圆柱 **48.77355 × 32mm、50g**；五指构型 `fivefinger_uniform48p8_side2deg_v1`，长指侧摆约 ±2°；原位置与方向不变 |
| 任务终止 | `boya_workspace`：既有工作区高度/侧向边界及确认；数值检查保持；接触少仅作诊断 |

使用现有兼容运行环境，不改共享环境、不自行切换引擎。记录服务器实际 GPU/驱动/CUDA、软件版本及代码 SHA；版本不兼容时先汇报具体问题。

## B 与 C 的区别及已完成工作

代码交接基点为 `a9f2d00`。**B** = `hora_pose_delta_drop16`：姿态差分速度投影到固定世界轴，旋转项双向截断 ±0.5rad/s，再加工作区失败成本 16。

**C** = `boya_object_spin_positive_cap_drop16`：每个物理步从 XYZW 姿态差分计算旋转增量，投影到相邻物体自身 Z 轴的负向平均轴，累加后除以控制周期；正向仍封顶 **0.5rad/s**，反向不截断。保留原线速度/手关节姿态/力矩/功耗系数 **−0.3/−0.3/−0.1/−2**，保留 terminal codes **9/10** 的成本 **16**，普通 timeout 不罚。关节姿态项不是物体倾斜惩罚。C 是 Boya 新奖励分支，不称原版 Hora 奖励。

- 实现：[spin_progress_reward.py](tendonspin/rl/spin_progress_reward.py)；配置：[boya_spin_progress_reward.json](assets/config/boya_spin_progress_reward.json)。
- 服务器接手前的历史验证：6项CPU解析检查和4条冻结轨迹重算；当时未做新训练或新物理执行。30M 同一完整有效 control 窗口的平均旋转分由 B 的 +0.226640 改为 C 的 −0.142165，与该窗口净反转 −3.821618° 一致；仅验证计算，尚未证明训练改善。
- **C 已在服务器接入并完成短验证。**原接入要求如下：控制开始时 `begin()`，每个物理步 `advance()`，控制结束调用 `compute_reward()`。它已经包含失败成本，勿重复扣 16。保留原 reset/timeout/数值与物理终止逻辑。
- 将 C 模块及配置加入 source hash、checkpoint/resume contract 和冻结评估；所有计划/日志/权重明确写 `arm=C` 与完整 reward profile。旧 B 脚本有硬编码 B 和 1024，不能直接运行它并声称训练了 C。

## 已筛选抓取与域随机化：复用，不重新搜索

沿用已完成的五指构型、九尺寸 **164 个** Isaac 筛选缓存和从第一轮 reset 开始的域随机化；不回退原四指抓取。五指参与操作并允许接触变化。现有准入要求稳定和至少两指组接触，**不是每个缓存同时五指接触的认证**，不得把它描述成 164 个全五指接触姿态；本交接不新增这种筛选门槛。

这份约 30KB 的抓取缓存已按用户授权加入 Git，服务器 `git pull` 后即可使用，无需另行复制。其他训练输出和权重仍忽略。缓存路径和校验值：

```text
outputs/boya_hora_scale_caches_v1/grasp_cache.npz
SHA256: 6cb72539fc3c40d1e75a49983019c9b4be679a37bd5d8dcef406c06696aaf993
各尺寸状态数: [3, 1, 5, 10, 12, 30, 20, 41, 42]
```

配置来源：[boya_hora_randomized_b.json](assets/config/boya_hora_randomized_b.json)、[boya_hora_randomized_scene.json](assets/config/boya_hora_randomized_scene.json)。新建 C 配置副本，仅替换奖励/组名/选定环境数；保持缓存 SHA、名义姿态和其余参数。

随机化已实现：尺寸中心 0.70:0.02:0.86、每环境固定 ±0.025 尺寸抖动；启动时摩擦 0.3–3、局部 COM 每轴 ±0.01m；reset 质量 0.01–0.25kg、P 比率 2.9/3–3.1/3、D 比率 0.9–1.1；关节观测噪声 ±0.02rad；每控制步概率 0.25 的 Gaussian×mass×2 外力及原衰减。`replicate_physics=False`；使用对应尺寸组缓存；质量变化时保留各尺寸名义惯量，与已有分支一致。来源是 [Hora v0.0.1 正式教师脚本](https://github.com/HaozhiQi/hora/blob/v0.0.1/scripts/train_s1.sh)，它实际启用外力，不能照 YAML 的 force=0。

新 checkout 先运行 `python scripts/prepare_local_runtime.py`，再用 `PYTHONPATH=. python scripts/probe_boya_import.py --out docs/data/isaac_boya_import_phases.json` 在本机生成 USD/导入身份。旧入口依赖本地 `docs` 协议；C 入口使用本交接或新本地协议作 `--protocol`，不能依赖未提交的历史实验文档。

## 空闲 GPU 上选择环境数，再正式训练

1. 用 `nvidia-smi` 确认空闲卡，选择可用且最快的一张，以 `CUDA_VISIBLE_DEVICES` 绑定；Isaac 和 PPO 使用同一卡。当前链是单 GPU；不启动重复正式任务充数，不干扰其他任务。
2. **1024 是本地选择，不是算法上限。**在同一 C 配置下，从 1024 起倍增环境数；原最多 6 档/20 分钟限制已按用户 2026-10-14 授权取消，可扩大档数和延长测试。每档独立短试，预热 2 updates，再测 8 updates；记录完整物理采样+PPO 的 actions/s、更新时间、显存峰值和主机内存，按用户最新偏好，选完整预算预计总耗时最短且留有保存/评估余量的档，计入每10M重建/评估。
3. 容量测试的 OOM 标为失败，该档及更大档停止，选已完成的最佳档；首档都失败或出现非容量执行/源/数值错误则汇报，不擅改 dt、碰撞、任务或算法。测试权重和动作不计正式 1.5B；测试不加资源 watchdog/cgroup/自动内存停机机制。
4. 将选定 `--num-envs` 贯穿 C trainer、stage、milestones、配置与 resume 身份；固定 horizon 8/minibatch 512/mini-epochs 5，不为提高吞吐修改奖励/控制频率/PPO 系数。按实际 `batch = num_envs × 8` 向上取整各 10M 节点，报告实际总动作与 update 数，不能继续硬写旧50M/1024时的6104 updates。
5. 确认 C 在原生训练/评估中生效、初态/cache/property 与配置一致、首更新有限且模型实际更新后，启动后台 **fresh seed43 C1.5B**：网络、normalizers、Adam、adaptive LR 和动作计数从零；不初始化自 B20M/B40M 或容量测试权重。用户已授权本交接所述训练，无需重复问启动许可。

## 保存、评估和结束

正式预算是所有环境累计 **1,500,000,000 控制动作**，不是每个环境1.5B，也不是update数。原Hora `third_party/hora/configs/train/AllegroHandHora.yaml` 的 `max_agent_steps` 为1.5B；原minibatch32768与本交接512不同，当前仅参考动作预算，不称完全复现。每10M（10/20/.../1500M，共150个节点）做一次冻结名义初态 **30s 或物理失败即停**的原状态连续评估，沿用现有离线分析。只在段间恢复自己的完整 C learner；仿真从匹配缓存重置，记录此事实。环境数正式启动后固定。

训练使用 headless、summary logs、TensorBoard、周期 policy/完整 learner 成对可恢复 checkpoint；仅评估保存逐物理步轨迹。保留所有旧 B/四指/五指权重与历史，不覆盖。评估报告有效时长、signed net/peak/backward、末 10s 净角、最大漂移/倾斜及真实停止原因；完整 30s 且净角和末段为正只是名义进展信号，不证明泛化、无限旋转或实机成功。

普通评估掉落按既定预算继续；正式训练遇执行/源/resume/数值错误则停并汇报，不自动重试或改路线。1.5B完成后停，不自动续训。启动后交付选定环境数、实测吞吐、进程/输出位置、实际预算和 ETA，结束后交付节点评估表；代码/配置、本交接及上述小型抓取缓存可 push，权重/完整日志和其他输出留在 ignored 目录。
