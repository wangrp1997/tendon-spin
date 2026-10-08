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
