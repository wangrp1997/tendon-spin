# Research and delivery rules

This is the independent TendonSpin learning repository. Never write to the
original botyard-inhand or sharpa-rl-lab repositories. Read docs/experiment_state.md,
the relevant protocols, README files and result tables before any new experiment.

The user authorizes learned policies, existing environment reuse and concise
milestone commits and pushes. Do not reinstall Isaac Sim/Lab or alter global
environments. Retain original model-based and learned pilot evidence as history.

Declare object/grasp/orientation, actuator interface, observation privileges,
engine/collision configuration, timing, budgets and acceptance criteria. An engine
change is a separately evaluated configuration. Do not hide model, grasp or
controller substitutions. One frozen checkpoint/configuration must execute an
uninterrupted original-state episode before claiming a standalone result.
Never sum training resets, spliced policies or angles across engines. Report
signed valid-prefix endpoint angle, peak, backward motion and actual stop reason.
There is no required turn-count success target; finite motion is not indefinite
rotation. Preserve per-physics-step actions/state/contact gates and raw provenance.

Hora, AnyRotate and SharpaWave are required comparison targets. References and
partial ports are not validated original baselines. Superiority requires matched
task, observation and evaluation budgets with multiple held-out runs. Hardware
readiness requires an observation-limited policy and validated motor/tactile
interfaces, not just a privileged nominal simulation result. Never send physical
commands or claim real-world success from a simulated or dry-run deployment.

Keep large training archives, copied runtimes and weights under ignored directories.
Commit code, self-contained model assets, source/engine identity and small evidence.

The user created https://github.com/wangrp1997/tendon-spin and explicitly
authorized pushing this learning repository there. Continue concise milestone
commits and pushes to that remote. Keep training outputs and weights local.
Prefer reference-code reuse over new implementations; add repository/project/paper and exact version attribution in
file headers, preserve licenses and declare port changes. Hardware touch is a
point-force array plus net3D fingertip force. Driver116 is message-array capacity,
not a verified count of physical active taxels. Baseline virtual touch features
can be trained without synthesizing the entire raw grid.

The user now explicitly requires questions when reproduction hits a blocker.
Stop the dependent experiment and present the evidence and concrete options before
changing backend, grasp/size, task criteria, observation tier or algorithm route.
Do not use controller substitutions or indefinite parameter scans to conceal a
failure. Complete independent source/metric verification while required input is
pending; elapsed time is not approval. SOTA requires matched executed comparisons.

The user requests no extra checking passes beyond the requested work. Prioritize
the concrete diagnosis or fix and prompt delivery; do not append optional audit,
render-verification, or broader test work. When providing a video, state both the
source of its motion and the renderer; recorded Isaac poses drawn with MuJoCo
are not an Isaac-native recording.

The user authorizes all five fingers for the learning route:16 active motor
actions (TH4 + FF3 + MF3 + RF3 + LF3). Hold both wrist targets for now; keep
the four distal mimic joints passive. Use this layout in future grasp caches,
teacher/student and touch interfaces. Do not impose an extra small-finger lock
or amplitude multiplier. Preserve historical13-action results as their own runs.

The user now authorizes summary-only training logs: keep learning/episode statistics,
checkpoints and provenance; reserve full physics traces for separately recorded
policy evaluations or selected diagnostics. This supersedes full-step TRAINING
archive requirements for the declared1024learning route. All physical termination
checks remain. User explicitly declines the new resource watchdog/cgroup/automatic
memory-stop mechanisms for1024training. A10M cumulative budget and resumability were
authorized,then gated by a short native visual preview: deliver it for user review
before launching the long run. Do not start long training before that review.

The user reviewed the corrected floor/background video and now explicitly approves
starting the1024-env background10M cumulative training continuation. The preview
review gate is satisfied. Continue from the65536-action preview learner checkpoint,
keep the no-resource-watchdog and summary-log choices,save TensorBoard curves and
periodic resumable checkpoints. No further launch approval is needed for this stage.

Latest user correction: formal1024training must start FROM SCRATCH,not from the
preview checkpoint. The user permits first verifying resume on the already running
short continuation;stop and retain that run as resume_test. After verification,
launch fresh network,normalizers,Adam,seed43 and zero counts toward10M. Keep preview,
resume-test and formal TensorBoard runs separate. Formal later checkpoints remain
resumable. This supersedes the preceding instruction to resume the preview forformal.


Latest authorization after source audit: preserve all old weights and runs,correct
inherited5mm/15deg/12N task-done to the declared translatedHora height/time rule,
review,and launch a NEW fresh1024background10M run. This supersedes the earlier
blanket statement that all old physical task gates remain for this new branch.
Keep numerical checks and oldlimits as labeled diagnostics;sharedtrain/eval rule,
separate source/configuration identity and oldstrict shadow scores. No repeat
launch permission or new visual-review gate is needed;all other run preferences
(noresourcewatchdog,summarylogs,checkpoints,TensorBoard,push) persist.

Latest2026-10-09 user authorization: stop height-v2 and delete ONLY its weights
(done), preserve first strict-run weights. Replace v2 nominal-minus5mm termination
with declared Boya geometry-based manipulation-region height/lateral bounds and
brief confirmation; low finger contact is diagnostic, not an immediate reset.
Start a NEW fresh1024/10M run promptly, with necessary launch checks only.
No repeated approval/preview required. This supersedes v2 rule for the new branch;
retain old logs/profiles. Keep TensorBoard/checkpoints, no watchdog, stagepush.

Latest user authorization2026-10-09: implement and start cumulative continuation
of the CURRENT workspace10M teacher to50M total, with automatic frozen original-
state120s evaluation at20/30/40/50M. Restore fulllearnerstate; no fresh restart.
Reuse same task/reward/engine/28cache and numericalrules. Sequential segment
resume resets simulation fromcache; report explicitly. Stopafter50M or an execution
error/manualstop; no automatic retries or algorithm/criterion changes. Preserve
current10Mweights, logs and completedhistory; firststrictweights already deleted.
Keep no-watchdog/headless/TensorBoard/checkpoint/push preferences. Video requests
should deliver video without per-frame image clutter by default.

Latest2026-10-10 authorization: implement pose-derived rotation reward (completed,
0bdb2d3), explicitly migrate the preserved20M teacher to a separate reward branch
and START additional requested1M/rounded1,007,616 actions, same1024envs/original4
engine/task/cache/PPO/actions/observations. Preserve full learner state and label
ancestral actions separately; strict unchanged-contract resume stays enabled.
Final cumulative21,012,480/2565updates -> ONE frozen originalgrasp44/requested30s
evaluation with full trace and offline analysis -> STOP, no retries/extensions.
Read docs/experiments/2026-10-10-boya-pose-reward-continue1m/ before follow-up.
Keep all old weights/results, summary training logs, TensorBoard/checkpoints,
no resourcewatchdog and milestonepush. This bounded run has now completed and
stopped:1,007,616 new actions, final21,012,480/2565updates; ONE requested30s
evaluation valid1.9995s/net+25.428426deg/peak34.139185/backward10.944128, actual2.0s
below-region stop. Parent20M completed30s; holding regressed, no stable rotation
improvement. No follow-on budget/route change is authorized by this result.

Latest2026-10-10 user explicitly approves the single terminal-cost proposal and
says to start the final bounded trial. Restore ORIGINAL20M full learner state,
keep pose-derived rotation reward, add ONE raw cost16 at existing workspace
terminal codes9/10 only (no ordinary timeout cost). New separate branch,
requested200k/rounded204,800actions/25updates, cumulative20,209,664/2467updates;
then exactly ONE frozen originalgrasp44/requested30s evaluation plus offline
analysis and STOP. No retry, extension, coefficient scan or alternate route.
Read docs/experiments/2026-10-10-boya-pose-drop16-200k/PROTOCOL.md.
Acceptance: full30s and signed net angle above parent+30.426747deg; also report
peak/backward/actual stop. This is additional Boya reward shaping, absent from
original Hora, not a proven repair or matched-budget superiority claim.
Keep all prior weights/history, summary training/TensorBoard/checkpoints,
no resourcewatchdog, necessary checks only, and milestone commits/pushes.
This final trial has COMPLETED and STOPPED:204,800newactions/25updates,
soleevaluationactual5.95s/valid5.9495s/net-0.659489deg/peak68.653873/
backward75.388385,below-regionfailure. Bothdeclaredacceptanceconditionsfailed;
stable rotation unresolved. Original20M remains preserved. No follow-on budget,
retry, coefficient scan or alternate route is authorized by this result.

Latest2026-10-10 authorization supersedes the prior final-trial stop for NEW
fresh A/B experiments: each arm starts ZERO learner/network/normalizers/Adam/
adaptiveLR/RNGseed43, no20M/pilot/migration initialization. A=hora_pose_delta;
B=same plus existing workspace failure terminalcost16. Same1024envs/original4
engine/task/cache/PPO/16actions/privilegedobservations. Each requested10M,
rounded10,002,432actions/1221updates, with frozen originalgrasp44/requested30s
evaluations at cumulative1/3/5/10M and existingofflineanalysis. Continue the
fixed budget after ordinary physical evaluation failure; stop dependentchain
on manualstop/execution/source/resumeerror, no retry/extension/parameter scan.
Latest steering explicitly says START A FIRST and inspect actual GPU memory
before deciding simultaneous B execution. A is formal10M, not a short test;
B is currently not launched. No environment-count reduction or added memory
watchdog/cutoff. One-off memory readings are authorized for concurrency discussion.
See docs/experiments/2026-10-10-boya-pose-ab-fresh10m/PROTOCOL.md.
Keep all oldweights/history, summarytraining/TensorBoard/pairedcheckpoints,
headless/noresourcewatchdog, necessary checks only. User explicitly requests
launch milestone push and an acceptance ETA after startup. One seed and one
initial state per checkpoint support trend evaluation,not statistical superiority.

Latest2026-10-10 user now explicitly authorizes START B CONCURRENTLY after
reviewing A's memory. B launched2026-10-10 16:57:27Shanghai,outerPID2527031;
same fresh1024/seed43/10M schedule and shared immutableprotocol,onlyrewardprofile
hora_pose_delta_drop16 differs fromA. Keep A running and its pinned code/protocol
unchanged. Independent per-arm milestone evaluations may overlap peertraining;
recordactualwallstop if any,do not silentlychange budgets/physics/envcount.
See docs/experiments/2026-10-10-boya-pose-ab-fresh10m/CONCURRENT_LAUNCH.md.
Verify actualinitialpolicy/bothnormalizers equality and BfreshAdam/counts,save
one-off concurrentmemory/throughput evidence andpush;no extra tests/GPUtrials.
B first actualupdate has been verified:zeroancestry,8192actions/update1/Adamstep80,
initialpolicy/critic/bothnormalizers exactlyequalA;contractdiffonlyrewardconfiguration,
allsourcesidentical. BtrainingPID2527033,AtrainingPID2511383. Keep both running
to their fixed budgets unless manualstop/executionerror;do not change pinned
code/protocol. Actualsource/config/first-updateevidence retained in experimentdocs.

Fresh A/B10M has now COMPLETED and STOPPED: each10,002,432actions/1221updates,
all four frozen originalstate/requested30s evaluations plus offlineanalysis.
A completed2026-10-10 23:06:16Shanghai, finalvalid2.5495s/net+149.587268deg/
peak156.025928/backward6.731557, actual2.55s below-regionstop.
B completed23:19:14, finalvalid8.9495s/net+47.259685deg/peak302.394167/
backward281.898073, actual8.95s below-regionstop. Neither completed30s;
stable rotation unresolved. No automatic extension, retry, coefficient scan or
new route authorized by these results. Preserve all weights/history. Read
docs/experiments/2026-10-10-boya-pose-ab-fresh10m/RESULTS.md for fulltrend,
source evidence and the fixed-axis reward versus moving-axis score distinction.

Latest user explicitly authorizes the proposed A/B continuation to50M cumulative,
every10M frozen30s originalstate evaluation. Both launched2026-10-10 23:54:13Shanghai
(outerA3216699/B3216700), outputs/boya_hora1024_pose_ab_continue50m_v1/arm_a,arm_b.
Restore OWN10,002,432/1221 fulllearner; each adds40,001,536actions, final50,003,968/
6104updates. Nodes20/30/40/50M, one30s-or-failure episode each plus existinganalysis;
reuse existing10M result without another rollout. Same task/reward/engine/cache/
16actions/privilegedobservations/PPO/1024envs. No fresh restart/migration/LRreset.
Physics resets from same28cache at every fulllearner resume; report this explicitly.
Ordinary physical eval failure continues fixedbudget. Execution/source/resume/
manualerror stops dependentarm without retry/extension/parameter scan; stop50M,
no automatic500M. Preserve all parents; no-watchdog/headless/summary/TensorBoard/
pairedcheckpoint/milestonepush preferences persist. Read the continuation README
and immutable PROTOCOL.md in docs/experiments/2026-10-10-boya-pose-ab-continue50m/.
Do not modify pinned runner/helper/train/eval/analysis/core code or either protocol
while running. Necessary first actualresume identity verification remains allowed;
no extra GPUtrial/video/audit campaigns.
First actualresume verification is now complete: both initialmodels/normalizers
exact OWNparentmatch, corecontracts equal, first10,010,624/update1222/Adamstep97760
from97680, modelupdated and parenthashes preserved. Both were10,059,776/update1228
at2026-10-10 23:56:48Shanghai; trainingPIDsA3216719/B3216718, noerrors. Keep the
formal fixedbudget chains running; no additional launch approval is needed.
