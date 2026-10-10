# TendonSpin

Independent learning project for continuous cylinder rotation on the Boya
tendon-driven dexterous hand. Initial scope: privileged PPO teacher, then calibrated
dynamics randomization and observation-history/tactile student for deployment.
Hora, AnyRotate and SharpaWave are comparison targets; outperforming them and
hardware robustness are goals to be evaluated, not current claims.

The original botyard-inhand repository is read-only. Packaged assets are local:
24 CAD meshes, original stable grasp, actuator/collision model, source manifests,
selected research history, baseline reference code and original learned pilot.
There are no symlinks into the old repository. Existing Python/Isaac/Torch
environments are reused; no new environment installation is required on this host.

Native MuJoCo is the initial backend to preserve the verified motor/contact model.
Available Isaac Sim/Lab is assessed separately for GPU training; environment
availability alone does not validate a transferred tendon-driven hand model.
See docs/backend_decision.md, docs/baselines.md and docs/deployment.md.

First completed trial: final teacher ran120s with18.397394deg net rotation;
most progress was early and final30s reversed. Sustained rotation is unresolved.
[Actual results/video](docs/experiments/2026-10-08-teacher-v2/README.md).

Latest bounded teacher trial (2026-10-10): fresh1,007,616 actions with effective0
velocity iterations completed and stopped. Its single frozen original-grasp
evaluation dropped at1.15s: signed valid-prefix net+47.309deg, peak60.646deg,
backward14.923deg. Stable holding and rotation remain unresolved.
[Results and signal analysis](docs/experiments/2026-10-10-boya-velocity-zero-1m/README.md).

The original4-iteration matched fresh1M control also completed and stopped:
valid1.4995s/net+23.792deg/peak66.298deg/backward42.507deg, below-region failure.
Both configurations dropped within1.5s;0 did not improve retention and rotation
together. No stable configuration was identified at this budget.
[Matched control results](docs/experiments/2026-10-10-boya-velocity-four-1m/README.md).

Run migration verification:

```bash
PYTHONPATH=runtime/mujoco-patched:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rw/miniconda3/envs/free/bin/python scripts/verify_migration.py
```

Run the bounded teacher-v2 trial:

```bash
/home/rw/miniconda3/envs/env_isaaclab/bin/python -u -m tendonspin.rl.train \
  --initialize outputs/imported/ppo_v1/checkpoints/update_0064.pt \
  --out outputs/teacher_v2/NEW_RUN
```

Check results in docs/experiment_state.md and docs/experiments before rerunning.
Large runtime/checkpoints/raw logs stay under ignored runtime/outputs. Small
results, provenance and model assets are versioned. Training resets are never
reported as a continuous rotation episode; every evaluation uses one frozen policy.

Reference networks are reused directly through
[tendonspin/baselines/reference_models.py](tendonspin/baselines/reference_models.py).
Every added adapter declares repositories, versions, papers and changes in its
header. [Virtual-touch notes](docs/reference_reuse.md) explain why a full raw
force-point grid is unnecessary for the initial teacher/student baseline.
The hardware array is real sensing; its ROS message capacity is not a measured
taxel count. Baseline ports and hardware student execution remain pending.

Remote repository: [wangrp1997/tendon-spin](https://github.com/wangrp1997/tendon-spin).
The user created this remote and authorized milestone pushes on2026-10-08.
Training checkpoints, TensorBoard events and large raw logs remain local under
ignored outputs/; code, packaged model assets and small evidence are versioned.
