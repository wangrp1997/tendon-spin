# Research and delivery rules

This is the independent TendonSpin learning repository. Never write to the
original botyard-inhand or sharpa-rl-lab repositories. Read docs/experiment_state.md,
the relevant protocols, README files and result tables before any new experiment.

The user authorizes learned policies, existing environment reuse and concise
milestone commits and pushes. Do not reinstall Isaac Sim/Lab or alter global
environments. Retain original model-based and learned pilot evidence as history.

Declare object/grasp/orientation, actuator interface, observation privileges,
engine/collision configuration, timing, budgets and acceptance criteria. An engine
change is a separately evaluated configuration. Do not hide model, grasp or
controller substitutions. One frozen checkpoint/configuration must execute an
uninterrupted original-state episode before claiming a standalone result.
Never sum training resets, spliced policies or angles across engines. Report
signed valid-prefix endpoint angle, peak, backward motion and actual stop reason.
There is no required turn-count success target; finite motion is not indefinite
rotation. Preserve per-physics-step actions/state/contact gates and raw provenance.

Hora, AnyRotate and SharpaWave are required comparison targets. References and
partial ports are not validated original baselines. Superiority requires matched
task, observation and evaluation budgets with multiple held-out runs. Hardware
readiness requires an observation-limited policy and validated motor/tactile
interfaces, not just a privileged nominal simulation result. Never send physical
commands or claim real-world success from a simulated or dry-run deployment.

Keep large training archives, copied runtimes and weights under ignored directories.
Commit code, self-contained model assets, source/engine identity and small evidence.

The user has paused all pushes for this new repository until they create the
remote. Continue local milestone commits. Prefer reference-code reuse over new
implementations; add repository/project/paper and exact version attribution in
file headers, preserve licenses and declare port changes. Hardware touch is a
point-force array plus net3D fingertip force. Driver116 is message-array capacity,
not a verified count of physical active taxels. Baseline virtual touch features
can be trained without synthesizing the entire raw grid.

The user now explicitly requires questions when reproduction hits a blocker.
Stop the dependent experiment and present the evidence and concrete options before
changing backend, grasp/size, task criteria, observation tier or algorithm route.
Do not use controller substitutions or indefinite parameter scans to conceal a
failure. Complete independent source/metric verification while required input is
pending; elapsed time is not approval. SOTA requires matched executed comparisons.

The user requests no extra checking passes beyond the requested work. Prioritize
the concrete diagnosis or fix and prompt delivery; do not append optional audit,
render-verification, or broader test work. When providing a video, state both the
source of its motion and the renderer; recorded Isaac poses drawn with MuJoCo
are not an Isaac-native recording.

The user authorizes all five fingers for the learning route:16 active motor
actions (TH4 + FF3 + MF3 + RF3 + LF3). Hold both wrist targets for now; keep
the four distal mimic joints passive. Use this layout in future grasp caches,
teacher/student and touch interfaces. Do not impose an extra small-finger lock
or amplitude multiplier. Preserve historical13-action results as their own runs.

The user now authorizes summary-only training logs: keep learning/episode statistics,
checkpoints and provenance; reserve full physics traces for separately recorded
policy evaluations or selected diagnostics. This supersedes full-step TRAINING
archive requirements for the declared1024learning route. All physical termination
checks remain. User explicitly declines the new resource watchdog/cgroup/automatic
memory-stop mechanisms for1024training. A10M cumulative budget and resumability were
authorized,then gated by a short native visual preview: deliver it for user review
before launching the long run. Do not start long training before that review.

The user reviewed the corrected floor/background video and now explicitly approves
starting the1024-env background10M cumulative training continuation. The preview
review gate is satisfied. Continue from the65536-action preview learner checkpoint,
keep the no-resource-watchdog and summary-log choices,save TensorBoard curves and
periodic resumable checkpoints. No further launch approval is needed for this stage.

Latest user correction: formal1024training must start FROM SCRATCH,not from the
preview checkpoint. The user permits first verifying resume on the already running
short continuation;stop and retain that run as resume_test. After verification,
launch fresh network,normalizers,Adam,seed43 and zero counts toward10M. Keep preview,
resume-test and formal TensorBoard runs separate. Formal later checkpoints remain
resumable. This supersedes the preceding instruction to resume the preview forformal.
