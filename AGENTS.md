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
