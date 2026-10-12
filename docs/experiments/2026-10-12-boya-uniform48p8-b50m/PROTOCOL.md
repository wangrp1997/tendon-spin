# Fresh B teacher with the migrated24-state grasp cache

User explicitly authorizes launch after reviewing migration. Reuse Hora v0.0.1
PPO and the completed four-contact B reward/budget/evaluation implementation;
this is a new grasp/object configuration, not continuation of B40M or a matched
original-Hora baseline. Original assets,28-cache and all old weights remain.

Object: exact cylinder48.7735468437074mm diameter ×32mm height,50g, full gravity.
Scene identity: assets/grasp/fivefinger_uniform48p8_side2deg_v1/isaac_learning_scene.json.
It freezes the native nominal anchor's named q/commands and pose, converted
WXYZ→XYZW with zero velocities, and canonical plan center/orientation. Reward
uses negative canonical cylinder-Z; signed primary evaluation uses the moving
cylinder axis. Both training/evaluation apply geometry before physics reset.
The original hand-based workspace bounds remain unchanged.

Training cache: outputs/boya_uniform48p8_isaac_transfer_v1/grasp_cache.npz,
SHA25625f790e6080a55db19654128bebf97a78545b3bea189ea9bbdf889d1b93d1c52.
24 distinct random states passed ONE Isaac0.5s initialization+0.5s formal old
cache screen and the unchanged B workspace checks. This is a five-finger posture
route, not24 certified simultaneous-five-contact holds; contact count is diagnostic.
At every training reset use cached q/pose/commands and zero joint/object velocities.
No MuJoCo solver state transfer; no30s Isaac stability claim before learning.

Same16 active actions TH4+FF3+MF3+RF3+LF3, held wrists, four passive distal
mimics, no small-finger lock/amplitude reduction. Teacher96 proprio+9privileged,
8latent, no student/touch/DR. Existing IsaacSim6.1.0.0/Lab3.0.0rc1/GPUPhysX,
original_tgs16_4, v4 clipped external PD, CADconvex/selfcollision/28excludes,
friction.5, contactoffset.1mm, CCDoff, no floor. dt.0005s,100steps/control20Hz.
Numerical safety checks unchanged. boya_workspace z>=.066801253194m,
XYlower[-.459866091313,-.069149973487], upper[-.259054954510,.076535408821],
two20Hz outside confirmations; low contact diagnostic, training timeout400controls.
The old5mm/15deg/12N rules remain cache admission/shadow diagnostics only.

B=hora_pose_delta_drop16: shortest WORLD pose increments accumulated float64,
fixed negative canonical-axis rotation scale1/clip[-.5,.5], unchanged Hora
linear/pose/torque/work penalties; ONE raw terminalcost16 at workspace9/10,
ordinary timeout7cost0, PPO rewardscale.01. The failure cost is Boya shaping.
Fresh seed43 network/normalizers/Adam/adaptiveLR/counts,1024envs/horizon8,
minibatch512/5epochs/gamma.99/tau.95/initialLR.005. No old learner initialization.

Requested cumulative milestones10/20/30/40/50M, rounded
10,002,432/20,004,864/30,007,296/40,001,536/50,003,968actions
(1221/2442/3663/4883/6104updates). Each later segment restores OWN full
learner/RNG/optimizer/normalizers/LR; physics resets from same24-cache and
unfinished episodes are discarded. Segments are not uninterrupted physics.

Each milestone: exactly ONE frozen own checkpoint/normalizer, fixed new nominal
anchor (not random cache),30s-or-first-failure episode,1500s wallcap, full traces
and existing offline analysis. Initial q/v/object/commands readback checked to
1e-6 absolute numerical tolerance against frozen nominal, separately from task
criteria; original-grasp44 evaluations retain their exact original oracle.
Report signed valid-prefix net/peak/backward, duration/actual stop, contact/force/
drift/tilt and20–30s net (null if incomplete). Finite-motion signal: full30s,
positive endpoint and positive final10s. Retain every checkpoint; best observed
eligible milestone by net angle is selection on these same episodes, not held-out
validation. One seed/initial state is trend evidence, not statistical superiority.

Ordinary physical evaluation failure continues the fixed budget. Execution,
source/resume/numerical/manual error stops without retry, scan or extension.
Stop at50M, no automatic500M. Headless/nice10/OMP4/MKL4, summary training logs,
TensorBoard, paired first/every16/final checkpoints, no resource watchdog,
cgroup/memory-stop/trainingwallcutoff. No extra GPU pilot/preview/evaluation.
Only syntax/configuration and actual first-update launch checks; milestonepush.
Outputs/weights/full traces stay ignored under outputs/boya_uniform48p8_b50m_v1.
