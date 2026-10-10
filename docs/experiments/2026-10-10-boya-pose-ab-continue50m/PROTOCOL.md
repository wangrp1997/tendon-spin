# A/B own-learner continuation from10M to50M cumulative

The user explicitly approves the proposed A/B continuation after reviewing the
completed fresh10M results: restore each arm's own full10M learner, continue to
50M cumulative and evaluate every10M. This is an additional40M per arm, not a
fresh restart or approval to train to Hora's approximately500M paper budget.
Both arms run concurrently on the existing RTX5080. Read experiment_state,
root README and the [parent protocol/results](../2026-10-10-boya-pose-ab-fresh10m/README.md).

## Fixed sources and task

Reuse unchanged scripts/train_boya_hora.py, evaluate_boya_hora.py,
run_boya_hora_background.py and analyze_boya_small_budget.py from TendonSpin
commitd7d07f7. New run_boya_hora_ab_continuation.py changes only scheduling and
reuses fresh_milestones.command_for/evaluation_row/report/require_stage and
milestones.milestones. Keep inherited runtime source hashes and strict learner
contract equality; no explicit migration. This continuation protocol replaces
the launch-protocol document in new training provenance; protocol documents
are outside the unchanged core learner contract. Pin both old and new protocols
in the continuation plan and leave them immutable after launch.

A=hora_pose_delta; B=hora_pose_delta_drop16, one raw-16 cost at existing workspace
failure codes9/10, ordinary timeout7cost0. PPO reward multiplier.01 unchanged.
Measured WORLD XYZW shortest pose increments accumulated in float64 over100
physics steps/.05s, fixed negative original-cylinder-Z reward axis, rotation
scale1/clip[-.5,.5] and all linear/pose/torque/work costs unchanged. Fixed reward
axis and moving-cylinder-axis primary evaluation remain distinct. B is Boya
reward shaping, absent from original Hora; do not reinterpret its peak as net
rotation or claim the penalty is a proven repair.

40x32mm/50g cylinder, full gravity, originalgrasp44, no support floor. Original
center[-.365438990352,.018340188315,.106061099740]m, WXYZ
[.175062180804,-.901031779718,.346105303444,-.194180544127].
Five fingers,16 active motor actions TH4+FF3+MF3+RF3+LF3; both wrists held,
four distal mimics passive, no small-finger lock or amplitude multiplier.
Privileged teacher96proprio+9privileged/8latent; no DR/student/tactile actor.
v3 external clippedPD, uncalibrated armature4*D_effective*dt; CADconvex,
selfcollision/28excludes/friction.5/CCDoff. Existing IsaacSim6.1.0.0/
IsaacLab3.0.0rc1/GPUPhysX, original_tgs16_4: scene position16/255,
velocity4/255, articulation16/4, rigid16/1; effective4 by documented composition,
not a direct kernel counter. dt.0005,20Hz/100physicssteps, targetrate.35rad/s.

Same28-state cache outputs/boya_settled_cache_v2/grasp_cache.npz, SHA256
7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca.
Hora v0.0.1 PPO/network:1024envs, horizon8, minibatch512,5epochs,
gamma.99/tau.95; inherited adaptive LR/Adam/scheduler, no learning-rate reset.
boya_workspace: z>=.066801253194m, XYlower[-.459866091313,-.069149973487],
upper[-.259054954510,.076535408821]m, two20Hz outside confirmations. Low contact
is diagnostic; finite/jointspeed100/mimic.05 checks and numerical handling remain;
oldstrict shadow scores remain separate. Training timeout400controls/20s is not
an evaluation timeout. No task/backend/controller/observation/algorithm change.

## Learner ancestry and fixed budget

A origin outputs/boya_hora1024_pose_ab_fresh10m_v1/arm_a/actions_010000000,
checkpoint SHA256 f3b360c3efa128f1eb2296e0e562565e882f0e8025c7a08ea98b51cf28eeb3f1.
B origin corresponding arm_b, SHA256
b6093e4bd328e92adf1fdbfb638ff1e9f8727edb9d4984bb5ed31b3f1eecdf21.
Each starts here at10,002,432actions/1221updates; original fresh seed43 ancestry,
own policy/critic, both normalizers, Adam/LR/scheduler/RNG/progress counters all
restored. No fresh learner, no old workspace20M weights, no A/B state swapping.
Verify actual initial model/normalizers equal the respective parent and first
continued checkpoint is update1222/10,010,624actions, with Adam steps advanced
by80 per parameter from the inherited parent rather than reset.

| Requested cumulative actions | Rounded cumulative actions | PPO updates |
| ---: | ---: | ---: |
| 20,000,000 | 20,004,864 | 2442 |
| 30,000,000 | 30,007,296 | 3663 |
| 40,000,000 | 40,001,536 | 4883 |
| 50,000,000 | 50,003,968 | 6104 |

Each arm adds40,001,536actions/4883updates. Restore its OWN preceding full learner
for each segment; reset simulation from the same28-cache and discard unfinished
episodes/rollout. Segments do not form uninterrupted physics. Independent arm
milestones have no common wall-clock barrier; evaluations may overlap peer
training. Exactly four NEW frozen episodes per arm, eight total; reuse existing
10M results as history without another10M rollout.

## Evaluation, stop and acceptance

Each node train budget -> exactly ONE frozen final checkpoint/own-normalizer/
originalgrasp44 episode, requested30s or first physical failure, existing1500s
wallcap, full physics/reward trace -> existing offline analysis -> own learner
resume. Exact original joint_pos/joint_vel/object_state/commands reference:
outputs/boya_hora1024_workspace50m_v1/actions_020000000/evaluation/initial_state.npz.
This is an initial-state oracle only, not training initialization. Zero evaluation
resets/policy switches/learning. Numerical/execution/source/resume/manual error
stops the dependent arm without retries, extensions, coefficient scans or route
changes. Ordinary physical early failure is an outcome and does not cancel the
fixed budget. Stop normally after50M; no automatic500M stage.

Report signed valid-prefix moving-cylinder-axis endpoint angle, peak, backward
motion, actual duration/stop reason and drift/tilt/contact/force. Never sum reset
or policy angles. Report20–30s net only after the full30s is observed; otherwise
null. Primary finite-motion signal remains full30s AND positive net AND positive
20–30s net. No turn-count target or indefinite-rotation claim. Assess duration,
net and backward motion together at matched20/30/40/50M; one seed and one initial
state per node support learning trends only, not statistical superiority,
convergence/method invalidity, original-baseline reproduction, SOTA or hardware
readiness. No repeated evaluation/video/pilot campaign is authorized here.

## Outputs and launch constraints

Separate ignored outputs/boya_hora1024_pose_ab_continue50m_v1/arm_a and arm_b;
each actions_020000000 etc contains summary training, own TensorBoard, paired
first/every16/final checkpoints, evaluation full traces and analysis. Keep all
parent weights/logs/history. Commit code, protocol, small launch/evidence files;
leave weights/full traces/TensorBoard local. Existing headless/camerasoff/nice10/
OMP4/MKL4. No resource watchdog/cgroup/quota/memory-stop/trainingwallcutoff,
runtime reinstall/global changes/hardware commands/original-repository writes.
Necessary new scheduler CPU checks, syntax and actual first-resume identity only.

At previously measured concurrent~440–450actions/s per arm, added40M takes
about25–26hours plus evaluations/segment startup. First20M result about6–7hours
after launch; total estimate26–28hours, revised only from actual training rates.
This is a runtime estimate, not a learning guarantee.
