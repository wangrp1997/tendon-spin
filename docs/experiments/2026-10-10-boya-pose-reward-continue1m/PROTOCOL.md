# Boya20M teacher: explicit pose-reward migration and bounded1M continuation

2026-10-10. User approves the proposed20M-teacher/new-reward/additional1M
continuation and one final30s frozen evaluation. Read experiment_state, root
README, the pose-reward implementation/verification and preceding0/4 fresh1M
results. Preserve all old weights, source snapshots, logs and primary scores.

## Learner migration and budget

Parent: outputs/boya_hora1024_workspace50m_v1/actions_020000000/training/
teacher_final.pth, SHA256
9781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e.
Parent20,004,864actions/2442updates, seed43. It uses original reported-velocity
reward, not the new pose reward. Preserve policy/critic, observation/value
normalizers, Adam moments/steps, adaptiveLR/scheduler, CPU/CUDA/numpy/Python RNG,
cumulative counters/times and best_rewards. LR starts at its saved
5.7805099719442054e-5, not.005. Critic/normalizers/Adam inherit the old reward
distribution and must adapt; this small budget does not guarantee convergence.

scripts/migrate_boya_pose_reward.py performs an explicit source-pinned branch
migration using migration_plan.json. Verify parent SHA/record/archive identities,
same cache/PPO/network/termination and all unaffected contract source hashes.
Reviewed source replacements: isaac_hora.py adds reward/profile wiring;
isaac_parallel.py adds the previously implemented optional0 profile, with its
original4 route unchanged. Add solver_profiles.py and rotation_reward.py identity.
Add explicit original4 engine and pose-reward configuration to the legacy contract.
Do not disable ordinary strict checkpoint equality. Change branch lineage and
attach parent ancestry/reward-start count; no old action is labeled pose-trained.
After serialization require exact equality of every learner field and original
progress counter, and confirm parent file unchanged. No physics in migration.

New ignored migrated checkpoint under outputs/boya_pose_reward_migration_v1/;
new run outputs/boya_hora1024_pose_reward_continue1m_v1. Its lineage is separate
from the old reported-reward run. Start with zero new-reward actions while retaining
20,004,864 ancestral actions. Requested extra1,000,000; target21,004,864 cumulative,
rounded21,012,480 total =1,007,616 new actions/123 PPO updates, final update2565.
Restore full learner then reset simulation from the SAME28-state cache, discarding
unfinished parent episodes/rollout. This is not an uninterrupted20M physics episode.

## Fixed task and changed reward

40x32mm/50g cylinder, fullgravity, originalgrasp44 center
[-.365438990352,.018340188315,.106061099740]m; WXYZ quaternion
[.175062180804,-.901031779718,.346105303444,-.194180544127]. No support floor.
16 active motor actions TH4+FF3+MF3+RF3+LF3, held wrists and4passive distal
mimics. v3external clippedPD, armature4*D_effective*dt; CADconvex/selfcollision/
28excludes/friction.5/CCDoff. Privileged96proprio+9privileged/8latent teacher,
no DR/student/tactile actor. Existing IsaacSim6.1.0.0/Lab3.0.0rc1/GPU PhysX.

original_tgs16_4: TGS scene position16/255, velocity4/255, articulation16/4,
rigidbody16/1, effective4 by documented aggregation, not a native kernel counter.
dt.0005s/control20Hz/100physics steps per action/targetrate.35rad/s.
Same28cache outputs/boya_settled_cache_v2/grasp_cache.npz, SHA256
7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca.
Same Hora v0.0.1 PPO/network:1024envs/horizon8/minibatch512/5epochs/gamma.99/tau.95.

Only changed learning objective: hora_reported_velocity -> hora_pose_delta.
Sum shortest world XYZW quaternion rotation vectors for100physical increments
and divide by.05s; float64 then native-dtype velocity into unchanged upstream
Hora reward. Seed after reset/percontrol, no reset jumps. Same fixed negative
original cylinderZ target axis, rotation scale1/clip[-.5,.5], axis eligibility
and other penalties. Primary score still uses moving cylinder axis; reward
fixed-axis projection and clipping do not equal primary net progress exactly.
This is a Boya reward adaptation, not validated original Hora reward/baseline.

Same boya_workspace: z>=.066801253194m; XY lower[-.459866091313,-.069149973487]
and upper[-.259054954510,.076535408821]m, two20Hz outside confirmations.
Lowcontact diagnostic only; per-physics finite/jointspeed100/mimic.05 checks,
oldstrict shadow only. Training400control/20s timeout, not used in evaluation.

## Execution, evaluation and decision

Reuse existing background supervisor. Headless/camerasoff/nice10/4threads,
summary training statistics, separate TensorBoard, first/every16/final paired
resumable checkpoints. No resourcewatchdog/cgroup/memory stop/training wallcutoff.
Stop at action budget, execution error or manual request. No retries/extensions,
extra seeds, auxiliary GPU trials, previews, videos or optional audits.
Necessary source/migration equality/syntax and actual first-update checks only.

After normal training budget completion, automatically run exactly ONE frozen
final checkpoint with its own observation normalizer, originalgrasp44, requested
30s or first declared stop,1500s wall cap,0resets/switches/training. Require exact
initialq/qdot/object/commands against outputs/boya_hora1024_workspace50m_v1/
actions_020000000/evaluation/initial_state.npz. Record full per-physics actions/
state/contact/gates and selected pose-reward diagnostics. Existing offline
analysis follows, then STOP. No new reward applied to old historical scores.

Reuse existing20M original4 same30s diagnostic outputs/boya_velocity_diagnostic_v2/
velocity_04, not another reference episode: valid30s/net+30.426747deg/
peak35.822706deg/backward37.240682deg, normal30s stop;20-30s net-2.461895deg.
Report new valid-prefix signed endpoint/peak/backward, retention and actual stop.
Preliminary improvement requires completing30s while increasing useful net
rotation; explicitly report backward-motion change and partial/unobserved windows.
Earlydrop/larger return alone is not improvement. No contact-filtered primary
score, cross-reset/policy angle sums or required turn-count threshold.
This is one ancestry-based feasibility trial, not equal-total-budget independent
training, statistical superiority, sustained rotation, SOTA or hardware proof.
If ineffective, record that result and discuss before increasing budget or
changing backend/grasp/task/observation/algorithm route.
