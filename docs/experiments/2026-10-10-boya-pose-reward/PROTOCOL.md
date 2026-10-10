# Pose-derived rotation reward: implementation and offline verification

2026-10-10. User approves replacing the rotation reward's reported angular
velocity with actual pose changes, and explicitly requests recording and push.
This stage implements the change and verifies existing data. Budget:0new
physics steps,0policy episodes,0training actions. No training launch is included.
Read experiment_state, root README, original4/0 fresh1M results and the frozen20M
velocity diagnostic protocol/results. Preserve all old weights/logs/scores.

## Reward identity and exact change

Upstream reference: github.com/HaozhiQi/hora v0.0.1, MIT,
In-Hand Object Rotation via Rapid Motor Adaptation, arXiv:2210.04887.
Keep vendored compute_hand_reward unchanged. Original input profile:
hora_reported_velocity, angular velocity at the final physics sample of a control.
New Boya input profile: hora_pose_delta. For each control, take100 consecutive
measured world XYZW quaternion pairs, normalize them, compute the shortest
world rotation vectors Log(q_i * inverse(q_previous)), sum and divide by.05s.
Calculate in float64, cast the resulting world angular velocity to native tensor
dtype, and pass it to the same upstream reward function.

Keep the fixed target axis (negative original-grasp cylinder Z in world frame),
original axis eligibility condition, rotation scale1, symmetric[-.5,.5]rad/s
clip, and linear/pose/torque/work penalties with scales-.3/-.3/-.1/-2.
Clipping follows the control-mean projection. Per-physics increments avoid
aliasing when a control rotates more than pi; each physics increment must remain
below pi. Quaternion sign flips do not count as rotation. Seed the accumulator
after each reset and anew at each control; reset orientation jumps earn no reward.
No contact mask, task gate, action change or reward-weight search is introduced.

This changes the reward objective and resulting advantages/value targets, so it
may change learned behavior. PPO math, architecture, observations, action interface,
physical/numerical termination and signed-prefix evaluation metrics are unchanged.
The primary angle still integrates the moving cylinder axis; the reward retains
the existing fixed target axis. These axes differ under tilt. Symmetric clipping
also means mean reward is not mathematically identical to net angular progress.
Do not claim that the objective is identical to the evaluator or that learning
has been repaired based on offline reward verification alone.

## Wiring and checkpoint rules

train_boya_hora.py and run_boya_hora_background.py default to hora_pose_delta;
explicit --reward-profile hora_reported_velocity selects original input semantics.
The low-level HoraBoyaEnv API keeps its legacy reported-velocity default for old
callers; training passes the selected profile explicitly. Evaluation follows the
training record; absent legacy reward identity means reported velocity, never pose.
Both train/eval use one shared RotationRewardSignal implementation. Record
profile/config/source identity and selected reward velocity in diagnostics.
The paired learner contract includes reward configuration and helper source hash;
different configurations cannot silently resume as the same learner. Old weights
cannot be relabeled as pose-trained. Existing strict source checks stay enabled.
Executing old archived records requires their matching sources, as before.
The proposed20M-to-new-reward continuation requires an explicit recorded branch
migration; it is not performed by this implementation stage.

Future experiments must provide a new protocol consistent with the selected
reward profile; historical original-reward protocols do not become pose-reward
protocols because a CLI default changed. No future budget is launched here.

## Frozen source trajectory and verification budget

Use ONLY the existing completed original4 frozen20M30s diagnostic:
outputs/boya_velocity_diagnostic_v2/velocity_04, original checkpointSHA256
9781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e.
Originalgrasp44,40x32mm/50g cylinder/fullgravity, center
[-.365438990352,.018340188315,.106061099740]m, WXYZquaternion
[.175062180804,-.901031779718,.346105303444,-.194180544127].
16finger actions TH4+FF3+MF3+RF3+LF3, held wrists,4passive distal mimics;
privileged96proprio+9privileged/8latent teacher, no DR/student/tactile actor.
Recorded IsaacSim6.1.0.0/Lab3.0.0rc1/GPU PhysX/TGS16/4, v3external clippedPD,
armature4*D_effective*dt, CAD convex self-collision/28excludes/friction.5/CCDoff,
dt.0005s/control20Hz/targetrate.35rad/s; original boya_workspace bounds,
two-control confirmation, lowcontact diagnostic, finite/jointspeed100/mimic.05.
No backend/object/grasp/observation/controller changes or physical commands.

Offline script verify_boya_pose_reward.py groups saved poses into600 complete
controls, uses the deployed accumulator, independently compares SciPy float64
rotation-vector means, and reproduces captured original rotation rewards.
Require maximum control-velocity disagreement<1e-5rad/s and original reward
reproduction error<1e-5; replace only rotation in counterfactual total rewards.
Keep original primary angle/prefix/stop metrics unchanged. Summarize0-5,5-20,
20-30s and whole observed prefix; incomplete/absent windows labeled.

Necessary CPU tests cover stationary/antipodal quaternions, reset jumps,
world-frame positive/negative rotation, more-than-pi control turns, Hora clipping,
incomplete controls, reward identity mismatch and paired learner next-update
restore. Static syntax check for edited files. No native rollout, video,
broader test suite, resource watchdog or additional training experiment.
