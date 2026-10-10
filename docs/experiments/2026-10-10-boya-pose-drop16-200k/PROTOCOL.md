# Approved final bounded trial: pose reward plus terminal cost16

2026-10-10. User explicitly approves the recorded
[proposal](../2026-10-10-boya-drop-penalty-proposal/PLAN.md) and says this is the
last opportunity. Execute exactly one declared branch, requested200k/newrounded
204,800actions/25updates, then exactly one requested30s frozen evaluation and STOP.
No automatic retry, extension, alternate coefficient or follow-on configuration.
Read root README, experiment_state, parent and failed pose1M results and proposal.

## One changed reward term

New explicit profile hora_pose_delta_drop16. Reuse the same per-physics world
XYZW pose-delta rotation signal, fixed negative original-cylinder-Z target axis,
float64 arithmetic/native-dtype velocity, rotation scale1/clip[-.5,.5], and
unchanged original Hora linear/pose/torque/work terms. Add raw reward-16 ONCE at
the terminal control for existing Boya workspace failure codes9/10 (below-region
or lateral-region failure). Ordinary training timeout/code7 gets no cost;
numerical rules retain their existing handling. No new termination criterion.
Upstream PPO still multiplies the whole reward by.01; no upstream source edits.
Shared RotationRewardSignal.terminal_cost is called once before training reset
and once before frozen evaluation stops. Record base reward, terminal cost and
failure code separately. Valid-prefix reward means exclude the invalid terminal
sample, so analysis also reports terminal controls/costs including that sample.

This is additional Boya reward shaping, absent from original Hora v0.0.1;
not an original validated baseline. Cost16 is the single existing-trajectory
calibrated candidate in the proposal, not a sweep or a proven causal repair.
Keep reward/engine/source identity in training results and paired contracts.
Old profiles remain explicit and receive no new terminal cost.

## Frozen learner, object and task

Start from ORIGINAL20M teacher, not the failed21M final checkpoint:
outputs/boya_hora1024_workspace50m_v1/actions_020000000/training/teacher_final.pth,
SHA2569781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e,
20,004,864actions/2442updates. Explicit separate migration/lineage; preserve every
serialized learner field (policy/critic, bothnormalizers,Adam/LR/scheduler,RNG,
counters/times/best_rewards). SavedLR5.7805099719442054e-5; no reset. Label all
old actions ancestral,0drop16 actions at migration. Verify exact learner equality
after serialization and parent unchanged. Strict ordinary resume equality stays.
Reset simulation from same28cache, discarding parent unfinished episodes/rollout.

40x32mm/50g/fullgravity/originalgrasp44/no support floor; center
[-.365438990352,.018340188315,.106061099740]m, WXYZ quaternion
[.175062180804,-.901031779718,.346105303444,-.194180544127].
16 active finger actions TH4+FF3+MF3+RF3+LF3,heldwrists/4passivemimics.
Privileged96proprio+9privileged/8latent teacher, noDR/student/tactile actor.
v3external clippedPD/A=4*D_effective*dt,CADconvex/selfcollision/28excludes/
friction.5/CCDoff. Existing IsaacSim6.1.0.0/Lab3.0.0rc1/GPU PhysX.
original_tgs16_4: sceneTGSposition16/255,velocity4/255,articulation16/4,
rigidbody16/1,effective4 by documented aggregation,notnativekernelcounter.
dt.0005/control20Hz/100physicssteps,targetrate.35rad/s.
Same28cache outputs/boya_settled_cache_v2/grasp_cache.npz,SHA256
7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca.
Same Hora v0.0.1 PPO/network,1024envs/horizon8/minibatch512/5epochs/gamma.99/tau.95.

boya_workspace unchanged: z>=.066801253194m; XY lower[-.459866091313,-.069149973487],
upper[-.259054954510,.076535408821]m, two20Hz outside confirmations. Lowcontact
diagnostic only; per-physics finite/jointspeed100/mimic.05 andoldstrictshadow unchanged.
Training400control/20s timeout, not used for evaluation. No other reward weight,
controller, observation, grasp, engine or algorithm change.

## Budget, evaluation and acceptance

Requested+200,000 -> cumulative request20,204,864 -> rounded20,209,664actions/
2467updates, exactly204,800newactions/25updates. New ignored output
outputs/boya_hora1024_pose_drop16_continue200k_v1; separate migrated checkpoint
outputs/boya_pose_drop16_migration_v1/teacher_20m_pose_drop16_branch.pth.
Headless/camerasoff/nice10/4threads,summarytraining/TensorBoard,
first/every16/final paired checkpoints, noresourcewatchdog/cgroup/memoryquota/
trainingwallcutoff. Stop budget/error/manualrequest. Necessary focused reward
tests,edited-file syntax,actual migration/first-update verification only;
no auxiliary GPU trial,preview,video,broader test/audit or coefficient scan.

Normal budget completion triggers ONE frozenfinal ownnormalizer/originalgrasp44
evaluation,30s orfirstdeclaredstop,1500swallcap,0resets/switches/training. Exact
initialq/qdot/object/commands against outputs/boya_hora1024_workspace50m_v1/
actions_020000000/evaluation/initial_state.npz. Full per-physics state/actions/
contact/gates,selected reward components; existing offline analysis,thenSTOP.
Primary angle remains signed moving-cylinder-axis valid-prefix endpoint/peak/
backward,actual duration andstop reason; no reset/policy sums or contact filtering.

Reuse original20M30s diagnostic:valid30s/net+30.426747deg/peak35.822706/
backward37.240682,normal30sstop. Preliminary filter: complete30s AND net>
30.426747deg; explicitly report backward change. Holding alone does not pass.
Report failure if either condition fails; no automatic new budget even if passed.
The previous no-terminal-cost branch had+1,007,616trainingactions andvalid1.9995s/
net+25.428426/peak34.139185/backward10.944128,below-regionstop. Different training
budgets prevent a matched-budget causal penalty claim. Single branch/one episode
is feasibility evidence, not statistical superiority/SOTA/indefinite rotation or
hardware proof. Preserve all old weights/logs/sources/scores;archive small new
evidence andpush,weights/raw/TensorBoard remainlocal.
