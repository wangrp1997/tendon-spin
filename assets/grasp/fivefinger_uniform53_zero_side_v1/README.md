# Optional near-zero-side grasp

52.9882mm diameter ×32mm height,50g upright cylinder, with near-zero index/little
finger lateral targets. Native grasp tools use the separately identified patched MuJoCo runtime.

Load with `load_native_grasp(root, "fivefinger_uniform53_zero_side_v1")`.
`scene_visual.xml` supplies white BOTYARD branding.

Sampling can perturb all16 finger motors while holding the wrists and keeping
four mimics passive. Five-finger contact is diagnostic, not a cache admission requirement.
Saved cache and policy artifacts remain local; native grasp samples require
separate migration before use in Isaac learning.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=runtime/mujoco-patched:.:scripts python scripts/collect_boya_mujoco_grasps.py \
  --grasp-profile fivefinger_uniform53_zero_side_v1 --object-profile boya_uniform53 \
  --reuse-profile-hold --perturb-degrees 2 --batches 1 --candidates-per-batch 64 \
  --workers 4 --seed 42 --out outputs/boya_uniform53_small2deg_v1
```
