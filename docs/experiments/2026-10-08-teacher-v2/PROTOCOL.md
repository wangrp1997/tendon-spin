# TendonSpin teacher-v2 protocol

Read original latest index, privileged-PPO pilot, history audit and nonlearning
reassessment before this trial. New repository and packaged physics must first
replay the old execution and 95-feature observations exactly.

Configuration: original Boya 40x32mm/50g cylinder, grasp44/original orientation,
world -z gravity 9.81m/s2, friction .5; 13 original kp10 position actuators,
wrist/LF fixed, unchanged coupling, CAD contacts, native multiccd patched
MuJoCo3.13/.5ms. All genuine hand side/link/palm contacts allowed. Original
5mm/15deg/12N/1.5mm/support>=2 real fingers/no external/floor/numerical gates
run each physics step. First invalid frame is stored but excluded from score.

Teacher observes same 95 truth features; 20Hz bounded position increments
at .35rad/s with per-physics-step linear interpolation. No fixed gait or reset
within evaluations. No domain randomization/student/hardware claims in this trial.

New factor from PPO-v1: reward speed cap reduced from 1 to .2rad/s and rotation
coefficient 5/rad, giving maximum 1/s; unchanged failure penalty 5. All former
position/tilt/tracking/action costs retained. Thus any failed episode shorter
than5s has nonpositive return even before costs. This aligns incentive with
continued safe execution; it is not evidence that this change alone solves gait.
Backward motion is signed, without an independent failure gate.

Weights initialize from imported PPO-v1 u64; optimizer is restarted and provenance
saved. Seed44, 8 environments x256 rollout x128 updates=262144 action-step cap,
1200s training-phase wall budget (checked at update boundaries), training episode
30s. Frozen initialized/32/64/128 checkpoints each execute from original initial
state under a120s window. No manual rescue, policy switch or summed reset angles.
If the wall budget ends early, evaluate the last frozen completed update. Stored
commands/state/forces/flags/final integration state must independently replay
exactly. Training reward is not the comparison score. Report actual wall budget
granularity and unfinished windows, not an inferred indefinite capability.

Training uses existing GPU Torch environment and copied cp311 native runtime.
Isaac Sim availability is checked separately; no engine switch in this trial.
Isaac needs validated Boya collision/coupling/position interface before it can
become a training backend. Cross-engine results remain distinct.
