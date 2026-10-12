# Optional near-zero-side grasp

52.9882mm diameter ×32mm height,50g upright cylinder. Planned FF/LF side0°;
settled sides−0.162°/+0.149°. Native patched MuJoCo3.13 only. ONE0.5s
initialization+30s formal hold passed: five-finger contact100%, max drift0.143mm,
axis tilt0.217°. Holding does not validate rotation, perturbations or Isaac.

Load with `load_native_grasp(root, "fivefinger_uniform53_zero_side_v1")`.
`scene_visual.xml` supplies white BOTYARD branding; `grasp_top.png` combines
recorded final-state top/side views rendered with MuJoCo, with zero render physics.
`validation.json` and `provenance/` retain execution/source identity.

Small sampling: one64-candidate batch, seed42,63 random±2° perturbations of all
16 finger motors plus1 repeated anchor. Wrists held, four mimics passive,
same native controller/CAD/friction/dt0.0005 and old0.5s init+0.5s formal gates
(5mm/15°/100rad_s/12N/.05rad and two contact groups in90% of final0.1s).
Five-finger contact is diagnostic, never required. No resampling or new training.
Reuse the completed nominal hold; sampled states are1s screens, not30s results.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=runtime/mujoco-patched:.:scripts python scripts/collect_boya_mujoco_grasps.py \
  --grasp-profile fivefinger_uniform53_zero_side_v1 --object-profile boya_uniform53 \
  --reuse-profile-hold --perturb-degrees 2 --batches 1 --candidates-per-batch 64 \
  --workers 4 --seed 42 --out outputs/boya_uniform53_small2deg_v1
```
