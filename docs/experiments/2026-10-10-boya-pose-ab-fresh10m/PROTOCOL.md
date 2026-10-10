# Fresh A/B pose-reward comparison, fixed10M per arm

2026-10-10 user explicitly approves two fresh groups, launch/push and an acceptance
ETA. After discussing single-GPU concurrency, latest steering is START A FIRST
and measure its actual memory. A is authorized to complete its formal10M schedule;
this is not an auxiliary short trial. B is prepared with the identical schedule,
but is not launched by A's runner. Decide concurrent feasibility from actual A
allocation; do not silently reduce the declared1024 environments or change physics.
Read root README, experiment_state, prior pose1M and drop16 results/protocol.

## Question and frozen comparison

Compare learning from ZERO under measured-pose rotation reward versus the same
reward plus fixed terminal failure cost16. A=hora_pose_delta;
B=hora_pose_delta_drop16, raw-16 once on existing workspace terminal codes9/10,
ordinary timeout7cost0, PPO reward multiplication.01 retained. No coefficient scan.
Both use shortest WORLD XYZW per-physics pose increments, float64 accumulation,
sum100/.05s then native tensor dtype, fixed negative original-cylinder-Z target
axis, original Hora rotation scale1/clip[-.5,.5] and linear/pose/torque/work costs.
This is a Boya reward adaptation, not validated original Hora. B cost16 is a
candidate calibrated from finite older trajectories, not a proven optimum.

Each arm starts NEW policy/critic, observation/value normalizers, Adam, LR.005,
adaptive scheduler and RNG seed43,0actions/updates. No initialize/resume/migration
from20M/21M or any pilot. Same random seed and source/PPO configuration must
produce identical initial policy/critic and both normalizers; preserve actual
initial tensors and first8192-action paired checkpoints for verification.
Later milestone resumes restore each arm's OWN full learner and RNG, retaining
strict contract equality, counts and lineage. Simulation resets from the same28
cache at each segment; unfinished episodes/rollout are discarded. Physics is not
uninterrupted across segments. Every evaluation is a separate frozen episode.

40x32mm/50g cylinder/fullgravity/originalgrasp44/no supportfloor. Center
[-.365438990352,.018340188315,.106061099740]m, WXYZ
[.175062180804,-.901031779718,.346105303444,-.194180544127].
All five fingers:16 active motor actions TH4+FF3+MF3+RF3+LF3, bothwrists held,
fourdistal mimics passive. No smallfinger lock/amplitude multiplier.
Privileged96proprio+9privileged/8latent teacher; no DR/student/tactile actor.
v3clippedexternalPD,uncalibratedarmature4*D_effective*dt;CADconvex/selfcollision/
28excludes/friction.5/CCDoff. IsaacSim6.1.0.0/Lab3.0.0rc1/GPUPhysX reused.
original_tgs16_4:sceneTGSposition16/255,velocity4/255,articulation16/4,rigid16/1;
effective4 by documented aggregation,notnativekernelcounter. dt.0005,
20Hz/100physicssteps,targetrate.35rad/s. Same28cache
outputs/boya_settled_cache_v2/grasp_cache.npz,SHA256
7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca.
Hora v0.0.1 network/PPO,1024envs,horizon8,minibatch512,5epochs,gamma.99/tau.95,
same initial learningrate.005 and unmodified adaptiveLR rule for botharms.

boya_workspace: z>=.066801253194m, XYlower[-.459866091313,-.069149973487],
upper[-.259054954510,.076535408821]m,two20Hz outsideconfirmations. Lowcontact
diagnostic;finite/jointspeed100/mimic.05 perphysics,numericalhandling unchanged;
oldstrict shadow diagnostics. Training400control/20stimeout,notforevaluation.
No backend/grasp/controller/observation/task criterion/PPO change.

## Budget and automatic milestones

Each arm cumulative FROM ZERO:

| Requested actions | Rounded actions | PPO updates |
| ---: | ---: | ---: |
| 1,000,000 | 1,007,616 | 123 |
| 3,000,000 | 3,006,464 | 367 |
| 5,000,000 | 5,005,312 | 611 |
| 10,000,000 | 10,002,432 | 1221 |

scripts/run_boya_hora_fresh_milestones.py reuses existing background/train/eval/
analysis entrypoints. Each completed budget -> exactlyONE frozenfinalmodel/
ownnormalizer/originalgrasp44 evaluation,requested30s orfirstphysicalfailure,
1500swallcap/fullphysics/rewardtrace -> existingofflineanalysis -> ownlearnerresume.
Exactly4newepisodes per arm,8total if B is launched. No extra preview/video/GPUtrial.
Match originalstate q/qdot/object/commands EXACTLY against preserved
outputs/boya_hora1024_workspace50m_v1/actions_020000000/evaluation/initial_state.npz;
reference is an initial-state oracle,never loaded as training weights/state.
0evaluationresets/switches/learning. Numerical/execution/source/resume/manualerror
stops dependentchain, no retry/extension or routechange. Ordinary physical early
evaluation failure is an outcome; it does NOT cancel the fixed learning budget.

Record signed valid-prefix moving-cylinder-axis endpoint/peak/backward,actual
duration/stopreason,drift/tilt/contact/force andcomplete trace. Report20–30s net
only iffull30s observed; otherwise null. No contactfilter/reset/policysums.
Check training loss/episode curves plus frozenmetrics at allfour milestones;
do not infer learned rotation from trainingreturn,peak alone or oneearlydrop.
Preliminary originalgrasp signal:full30s AND positive net AND positive20–30s net;
this describes finite motion,not indefinite rotation or an imposed turn-count goal.
Compare A/B only at matched cumulativebudget; old20M is historical context,
not equal-budget control. One seed and one state per checkpoint establish a
learningtrend,not statistical superiority/convergence/methodinvalidity/SOTA or
hardware-readiness. Budget exhaustion without improvement means not observed
within10M; it does not prove that the reward can never learn.

## Outputs, resource policy and launch

Ignored outputs/boya_hora1024_pose_ab_fresh10m_v1/arm_a and arm_b;
actions_XXXXXXXXX/training/evaluation/analysis per milestone. Summarytraining,
separate TensorBoard segmentcurves withcumulativeactionaxes,evaluationcurves,
first/every16/final pairedresumablecheckpoints. Weights/raw/TensorBoard staylocal.
Headless/camerasoff/nice10/OMP4/MKL4. No resourcewatchdog/cgroup/quota/memory-stop/
trainingwallcutoff. A one-off post-start NVIDIA memory reading supports the
concurrency discussion and does not stop or constrain training. No globalinstalls,
runtimechanges,hardwarecommands or writes to original repositories.
Necessary newwrapper schedule/stopCPUchecks,syntax andactualfreshfirstupdate
verification only. Push small launch/protocol/source/evidence;retainoldhistory.
Prior single-run measurements suggest A10M plus4evals about4–5hours,first1M
about30–40minutes. Both sequential about8–10hours; simultaneous completion ETA
must use measured concurrentthroughput,not an assumed2x speedup.
