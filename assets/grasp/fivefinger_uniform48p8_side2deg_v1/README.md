# Optional48.8mm small-side grasp

Exact48.7735468437×32mm/50g upright cylinder. ONE native patched MuJoCo3.13
0.5s initialization+30s formal hold passed, five fingers throughout; max drift
0.250mm/axis tilt0.358°. Planned long side<=1.8°, observed max2.074°.
Use `load_native_grasp(root, "fivefinger_uniform48p8_side2deg_v1")`.
`grasp_top.png` shows recorded final-state top/side poses with native MuJoCo rendering.
No rotation, Isaac transfer or hardware validation is claimed.

Fixed sampling: seed42,63 random states plus1 nominal anchor, four CPU workers;
thumb/flexion joints±2° relative to settled seed, four long side joints unchanged
for initialization only. All16 finger actuators remain active, wrists held,
four distal mimics passive; same frozen preload/native CAD/friction/engine.
Each episode0.5s init+0.5s formal, dt0.0005/control20Hz, old5mm/15°/100rad_s/12N/
.05rad gates and>=2contact groups in>=90% final0.1s. No five-finger requirement.
Reuse the already completed nominal30s; no retry, extra hold or training.
Sampled states are1s screens and require separate Isaac transfer before Isaac RL.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=runtime/mujoco-patched:.:scripts python scripts/collect_boya_mujoco_grasps.py \
  --grasp-profile fivefinger_uniform48p8_side2deg_v1 --object-profile boya_uniform48p8 \
  --reuse-profile-hold --perturb-degrees 2 --fixed-initial-joints FFJ4 MFJ4 RFJ4 LFJ4 \
  --batches 1 --candidates-per-batch 64 --workers 4 --seed 42 \
  --out outputs/boya_uniform48p8_small2deg_v1
```
