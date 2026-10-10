# Frozen20M: one effective-zero velocity-iteration episode

2026-10-10. User approves source/configuration verification followed by exactly
ONE new frozen-policy original-state30s episode at effective velocity iterations0,
position iterations16. Compare with the completed4-iteration archive; do NOT
rerun4, scan parameters, train, change rewards or substitute a controller.
This approval supersedes the earlier pair's restriction on follow-on evaluation.
Read experiment_state, workspace50M and velocity-diagnostic protocols/results,
and the PhysX official-document follow-up before execution.

## Frozen task, policy and budget

Checkpoint: outputs/boya_hora1024_workspace50m_v1/actions_020000000/training/teacher_final.pth;
SHA256 9781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e.
20,004,864 training actions, own frozen model/normalizer, deterministic bounded
mean, seed43, same observation history/noise schedule, no learning or DR.
Original grasp44: center [-.365438990352,.018340188315,.106061099740]m,
WXYZ quaternion [.175062180804,-.901031779718,.346105303444,-.194180544127].
Require exact initial q/qdot/object state/commands against both original20M and
archived4. Existing28cache constructs the environment only; never reset from it.

40×32mm/50g cylinder, full gravity; 16 finger motor actions TH4+FF3+MF3+RF3+LF3,
held wrists, four passive mimics, v3 external clipped PD and prototype armature
4*D_effective*dt. Privileged teacher:96 proprio history+9 privileged/8 latent.
Isaac Sim6.1.0.0/Lab3.0.0rc1, GPU PhysX/TGS, original CAD convex self-collision
and28 excludes, friction.5, CCD off, dt.0005s,20Hz/100 physics steps per action,
target rate.35rad/s. Original Hora reward formula is reconstructed unchanged.

Unchanged boya_workspace termination: center z>=.066801253194m; XY bounds
[-.459866091313,-.069149973487] to [-.259054954510,.076535408821]m, failure
after two consecutive20Hz outside samples. Low contact is diagnostic only.
Finite-state/jointspeed100rad/s/mimic.05 checks and oldstrict shadow unchanged.
No training20s timeout. One uninterrupted episode, no resets/policy switches.
30s/60000 physics steps maximum,1500s wall budget; stop at original failure,
execution error or manual stop. No automatic retry or additional physics run.
Headless/cameras off,4 threads/nice10, existing runtime; no watchdog/cgroup/global
environment changes, video, optional plots or broad verification passes.

## Isolated engine configuration

Reference: outputs/boya_velocity_diagnostic_v2/velocity_04, completed30s episode.
It has scene min/max velocity4/255, articulation request4, rigid-body requests1.
Position scene min/max16/255, articulation and all rigid bodies request16.

New configuration: scene min/max velocity0/0; ALL rigid-body and articulation
velocity requests0, including hand links and cylinder. Position requests/bounds
remain identical. The scene maximum0 explicitly prevents a residual/default
request from raising the count. These are implementations of the single changed
factor, effective velocity iterations4->0; report all authored changes explicitly.
No contact, solver type, actuator, task, observation or algorithm change.

Use evaluator-local PhysxCfg wrapper for scene bounds and existing scene_setup
hook for actor attributes AFTER spawning and BEFORE sim.reset initializes PhysX.
Never edit original hashed core, installed packages or source assets. Compare
pre-override actor requests with archived4, then verify all composed requests and
scene bounds after reset, runtime/source hashes, and both exact initial states
BEFORE episode integration. Assert only the declared iteration fields differ.

Installed PhysxCfg documents max-actor request clamped to scene min/max.
Effective0 is established by this rule plus composed0 actor requests and0/0 scene
bounds before/after initialization. Record available native tensor solver/iteration
methods. Current tensor and PhysX Python declarations expose no direct kernel
iteration counter; do not label USD readback as a measured internal counter.
If a configuration or initial-state blocker arises, stop dependent execution and
present evidence/options before changing route. Elapsed time is not approval.

## Metrics and interpretation

Reuse advance/measure, source checks, frozen policy, termination and score_prefix
from the completed diagnostic. Preserve every physics-step action/state/contact/
gates and cloned raw PhysX pose/velocity before and after Lab access; original
state, control observations, reconstructed reward terms and provenance retained.
Large traces/weights remain ignored/local; version small evidence and code.

Reuse independent SciPy float64 WORLD quaternion differences. Predeclared windows:
0–5s,5–20s,20–30s and common observed valid prefix with archived4. Report vector
velocity/pose RMSE, fixed-original-axis raw/pose rates, moving-axis pose rate,
contact fraction, original rotation/penalty/reward means and raw/Lab agreement.
Missing windows stay missing. No contact-based filtering of primary angles.
Primary scores: signed valid-prefix endpoint net, peak, backward motion, valid
duration, actual stop reason and drift/tilt/force maxima. Never pool configurations.

This is a single-policy/single-initial-state configuration diagnostic. Closed-loop
contact trajectories may diverge; better agreement supports iteration sensitivity,
not full mechanism proof or physical validation. It is NOT original Hora8/0
reproduction, statistical baseline validation, SOTA or hardware readiness.
No required turn-count threshold and no claim finite motion is indefinite rotation.
Preserve the existing4/16 results and all training history. No follow-on training
or permanent engine/reward change is authorized by this diagnostic.
