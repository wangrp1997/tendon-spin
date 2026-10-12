# Five-finger hold and bounded cache result

48×60mm /50g cylinder, optional `fivefinger_sharpa_cache28_v1`, native patched MuJoCo3.13.0; frozen targets, no learned policy or external support.

- Nominal uninterrupted30s passed: five-finger contact100%; maximum drift0.111439mm, tilt0.143300°, link normal2.091255N.
- Valid endpoint/peak+0.026383°, backward4.44e-9°; normal30s time-limit stop. This measures holding, not learned rotation.
- Fixed512 episodes completed in78.3s total wall time:504 random+8 repeated anchors.26 random states accepted, all26 distinct; anchors8/8 passed.
- Same old admission, no per-finger bearing-force gate. Accepted states' minimum finger count in final0.1s:20 three-finger,6 four-finger;0 with all-five contact in≥90% of that tail. This is not a five-finger random cache.
- Random rejection:476 drift>5mm and2 tilt>15° at formal start. No retries or extra batches.
- Only nominal grasp has30s validation; random states have0.5s initialization+0.5s formal screens. No training launched.

Native cache: `outputs/boya_fivefinger_mujoco_hold30_cache_v1/grasp_cache.npz`. Full native integration states plus named22 joint q/v,18 commands and object state. It is separate from the old28-state Isaac cache; Isaac transfer has not been validated.

MuJoCo nominal handoff: `load_native_grasp(root, name="fivefinger_sharpa_cache28_v1")` from `tendonspin.physics.grasp_profiles`. Scene, named state and exact saved integration state are in `assets/grasp/fivefinger_sharpa_cache28_v1/`. Existing original defaults remain.

Small source/config/result evidence: [hold30_cache_result.json](hold30_cache_result.json). Full traces and cache stay local under `outputs/`.
