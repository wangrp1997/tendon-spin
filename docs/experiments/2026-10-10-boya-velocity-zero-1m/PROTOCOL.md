# Effective-zero velocity iterations: fresh1024 / small1M budget

2026-10-10. User accepts the proposed small-budget fresh teacher training under
effective0 velocity iterations, retaining the Hora reward. This explicitly
authorizes the new training stage after the diagnostic-only approval.
Choose1,000,000 motor-action vectors as the bounded first budget, rounded to
1,007,616 actions /123 complete8192-action PPO updates. No automatic extension,
retry, seed sweep, reward change or additional training configuration.
Read experiment_state, workspace10M/50M protocols/results and effective-zero
diagnostic86cf44f before launch. This is a separately identified engine configuration.

## Task, observations and engine

Same original40×32mm/50g cylinder, full gravity, originalgrasp44 reference:
center [-.365438990352,.018340188315,.106061099740]m; WXYZ quaternion
[.175062180804,-.901031779718,.346105303444,-.194180544127].
16 active finger motor actions TH4+FF3+MF3+RF3+LF3, both wrists held,
four passive distal mimics; v3 external clipped PD and prototype armature
4*D_effective*dt. No controller or object/grasp substitution, no support floor.
Privileged teacher96 proprio history+9 privileged/8 latent, no DR, student or
tactile actor. Hardware readiness and comparative baseline validation are not claimed.

Reuse Isaac Sim6.1.0.0/Lab3.0.0rc1/GPU PhysX/TGS, original CAD convex
self-collision/28 excludes, friction.5, CCD off, dt.0005s, control20Hz/100 physics
steps per action, target rate.35rad/s. Only changed engine factor: effective
velocity iterations4->0. Scene velocitymin/max0/0 and every rigid-body/articulation
request0; position scene16/255 and every actor request16 retained.
Apply all actor overrides after spawn and before PhysX reset. Verify all1024
environments (25rigid bodies+1articulation each), composed requests and scene
iteration bounds before/after initialization. Persist solver_configuration.json.
Effective0 is configuration plus documented max-actor/clamp semantics, not a
direct kernel iteration counter measurement. Same profile is required by evaluation
and paired learner checkpoints; restore rejects a different profile/source contract.

Training reuses the same28-state cache produced under the original4 configuration,
outputs/boya_settled_cache_v2/grasp_cache.npz, SHA256
7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca.
This cache reuse is explicit; it is not a newly generated0-configuration grasp cache.
Training resets sample from it, while final evaluation restores originalgrasp44.

## Algorithm, termination and logging

Fresh network, observation/value normalizers, Adam and RNG seed43; no old weights
or learner state loaded. Original Hora v0.0.1 PPO and reward formula retained:
1024 environments, horizon8, minibatch512,5 epochs, initialLR.005, KL.02,
gamma.99/tau.95. No changed reward weights or pose-difference reward input.

Same boya_workspace termination for training/evaluation: z>=.066801253194m and
XY bounds [-.459866091313,-.069149973487] to [-.259054954510,.076535408821]m,
two consecutive20Hz out-of-region samples confirm failure. Low finger contact is
diagnostic, not a reset. Finite-state/jointspeed100rad/s/mimic.05 checks remain
per physics step; oldstrict scores are labeled shadow diagnostics only.
Training400-control/20s timeout; final frozen evaluation has no training timeout.

Headless/cameras off/nice10/4 threads; existing Python environment, process-only
EULA. No resource watchdog/cgroup/GPU memory cutoff/training wall cutoff.
Summary training/episode statistics and TensorBoard, complete paired resumable
checkpoint after first/every16/final update. No full-step training archive.
Output outputs/boya_hora1024_velocity0_small1m_v1, separate from all previous runs.
Code/protocol/engine identity and small launch evidence committed; weights/raw
traces remain ignored/local. All previous weights/results/source snapshots retained.

## Final evaluation and interpretation

After completing the action budget, automatically run exactly ONE frozen final
policy with its own normalizer from originalgrasp44,30s/60000 physics steps or
first declared physical/numerical stop,1500s wall budget. Require exact original
q/qdot/object state/commands against archived original20M initial_state.npz.
No resets, policy switches or training actions in this episode. Preserve every
physics-step action/state/contact/gates, control observations and reconstructed
original Hora reward terms. This is the existing evaluation path with declared
solver identity and optional reward recording, not a substituted controller.

Report signed valid-prefix endpoint net/peak/backward motion, duration, actual
stop reason and drift/tilt/force maxima. No contact-based angle filtering or
turn-count success gate. Independently derive SciPy float64 WORLD pose rates and
compare with reported object angular velocity; record fixed-original/moving-axis
rates, full-vector RMSE, contact labels and reward means. Windows0–5s,5–20s,
20–30s and whole valid prefix; incomplete windows labeled, missing ones null.
Terminal0.2s motion is labeled posthoc explanation, never a replacement score.

Assess whether retention and actual rotation improve together and whether positive
rotation reward corresponds to actual turning. Rising return alone is insufficient.
One small-budget run can support feasibility or reveal continued failure; it cannot
establish0 superiority over4-trained20M/50M policies, statistical convergence,
indefinite rotation, originalHora8/0 reproduction, SOTA or hardware success.
No matched-budget4-policy evaluation is authorized in this stage.

Supervisor stops after training/evaluation/offline analysis. An execution error or
manual stop prevents dependent stages, with no automatic retries. If reproduction
hits a blocker, preserve evidence and ask before changing engine/grasp/task/
observations/algorithm route. Native configuration checks happen in the actual
authorized run, not an auxiliary GPU preview. Necessary syntax and paired-learner
contract test only; no broad audit, optional video, plots or extra checking passes.
