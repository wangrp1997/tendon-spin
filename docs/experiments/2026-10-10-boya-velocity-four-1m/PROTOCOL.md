# Original4 velocity iterations: matched fresh1024 / small1M control

2026-10-10. User explicitly approves the proposed original4-iteration fresh1M
control after the effective0 fresh1M result, and asks for prompt launch without
extra work. Execute ONE new training run and ONE final frozen evaluation using
the existing runner. No additional seeds, training configurations or budget extension.
Read experiment_state, root README and the effective0 fresh1M protocol/results.

## Frozen comparison conditions

Reference run: outputs/boya_hora1024_velocity0_small1m_v1, completed1,007,616
actions/123updates, seed43; originalgrasp44 evaluation valid1.1495s, signed
net+47.308647deg, peak60.645683deg, backward14.923018deg, stopped below region.
Preserve that checkpoint, source snapshots, logs and raw traces unchanged.

Same original40x32mm/50g cylinder, full gravity, originalgrasp44 pose:
center[-.365438990352,.018340188315,.106061099740]m and WXYZ quaternion
[.175062180804,-.901031779718,.346105303444,-.194180544127]. No support floor.
16 active finger actions TH4+FF3+MF3+RF3+LF3; held wrists,4 passive distal
mimics; v3 external clipped PD, armature4*D_effective*dt. Privileged96 proprio
history+9 privileged/8latent teacher; no DR/student/tactile actor.

Existing Isaac Sim6.1.0.0/Lab3.0.0rc1/GPU PhysX/TGS; original CAD convex
self-collision/28excludes, friction.5, CCDoff, dt.0005s, control20Hz,
100physics steps/action, targetrate.35rad/s. Restore the existing
original_tgs16_4 profile: scene positionmin/max16/255, velocitymin/max4/255,
articulation requests16/4, rigid-body requests16/1. The scene minimum makes
the declared effective velocity count4. This differs from0 only in declared
velocity-iteration fields; do not author a new4/4 rigid-body variant.
This is a separately identified engine configuration, not an engine reinstall.

Same28-state cache originally generated under4:
outputs/boya_settled_cache_v2/grasp_cache.npz, SHA256
7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca.
Fresh network/normalizers/Adam/RNG seed43, no checkpoint initialization/resume.
Reuse unchanged training/environment/controller/reward/evaluation source code.
Only protocol/controller/output identities differ in addition to the engine profile.

## Training budget and termination

Requested1,000,000 motor-action vectors, rounded to1,007,616/123complete updates;
1024envs, horizon8, minibatch512,5epochs, original Hora v0.0.1 PPO/reward,
initialLR.005, KL.02, gamma.99/tau.95. No reward/input/action-scale changes.
Same boya_workspace: z>=.066801253194m; XY lower[-.459866091313,-.069149973487]
and upper[-.259054954510,.076535408821]m; two20Hz outside samples confirm failure.
Low contact diagnostic only. Per-physics finite/jointspeed100/mimic.05 checks;
oldstrict shadow only. Training400controls/20s timeout, no training timeout in eval.

Headless/camerasoff/nice10/4threads; existing environment; summary training logs,
TensorBoard and first/every16/final full resumable checkpoints. No resource
watchdog/cgroup/memory stop/training wall cutoff. Separate output:
outputs/boya_hora1024_velocity4_small1m_v1. Retain all existing evidence/weights.
Expected wall time roughly20-30min based on the completed0 run; stop by the
fixed action budget, error or manual request, not by this estimate.

## Evaluation and decision

Automatically evaluate ONE frozen final checkpoint with its own normalizers,
originalgrasp44, requested30s/60000physics steps or first declared stop,
1500s wall cap,0resets/switches/training. Require exact initial q/qdot/object/
commands against outputs/boya_hora1024_workspace50m_v1/actions_020000000/
evaluation/initial_state.npz, the same oracle used for0. Full per-physics
actions/state/contact/gates plus reconstructed originalHora reward retained.
Then run the existing offline SciPy velocity/pose/reward analysis and STOP.

Compare both separately reported signed valid-prefix endpoint/peak/backward
angles, valid duration, drift/tilt/force maxima and actual stop reason. Improvement
requires retention and useful real rotation together; a large angle during a drop
or larger training return alone is insufficient. Signal comparisons must label
observed durations; do not compare unequal-prefix RMSE as a matched time window.
Incomplete windows labeled and absent windows null. No contact angle filtering,
spliced policies, reset sums or required turn-count threshold.
This single-seed control can inform which configuration to pursue but cannot
establish statistical superiority, indefinite rotation, validated originalHora/
AnyRotate/SharpaWave baselines or hardware readiness. If both fail early, report
that the1M comparison did not identify a working configuration; do not silently
increase training or change grasp/task/observations/algorithm/backend.

Use necessary actual-launch/source-identity checks only; no auxiliary GPU run,
new test suite, video or optional audit. Errors stop dependent stages with no
automatic retries. Any blocker needing a changed route requires user input.
