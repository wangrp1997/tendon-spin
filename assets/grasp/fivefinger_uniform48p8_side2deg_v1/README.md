# Optional48.8mm small-side grasp

48.7735468437×32mm/50g upright cylinder with a small lateral-angle grasp configuration.
Use `load_native_grasp(root, "fivefinger_uniform48p8_side2deg_v1")`.
For Isaac learning, `isaac_learning_scene.json` specifies geometry, nominal state,
canonical rotation reference and the required cache identity.

All16 finger actuators remain active, wrists held, four distal mimics passive.
Native grasp sampling uses the separately identified patched MuJoCo runtime.
Nominal grasp data and sampled caches are separate artifacts; contact count is
diagnostic rather than a requirement that every initial state touch all five fingers.
Saved cache and policy artifacts remain local.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=runtime/mujoco-patched:.:scripts python scripts/collect_boya_mujoco_grasps.py \
  --grasp-profile fivefinger_uniform48p8_side2deg_v1 --object-profile boya_uniform48p8 \
  --reuse-profile-hold --perturb-degrees 2 --fixed-initial-joints FFJ4 MFJ4 RFJ4 LFJ4 \
  --batches 1 --candidates-per-batch 64 --workers 4 --seed 42 \
  --out outputs/boya_uniform48p8_small2deg_v1
```
