# TendonSpin experiment state

2026-10-08: independent repository created from botyard-inhand commit
204d197a9606fb7266e884f3b2e6110195be01cb. No original repository files are written.
24 CAD meshes, original40x32mm/50g/grasp44 stable state, original physical XML,
copied patched MuJoCo3.13/cp311 runtime and selected history are local.
Mesh paths are packaged; there are no original-repository runtime imports.

Migration completed: full imported u64 controls/qpos/qvel/motor/contact metrics/
flags/final integration state and all95-feature observations replay with error0.
See [migration evidence](data/migration.json); this is an old-record diagnostic,
not a new controller score.

Independent training smoke completed:1env×32steps×2updates=64 actions/6400
physics steps; initialized/final policies each run0.5s from original state and
reach their diagnostic time limits. Physical replay and stored-input inference
error0;5 PPO/deployment interface tests pass. [Smoke evidence](data/smoke.json).
A random student test validates schema/timing/bounds only, not a learned policy.

Historical PPO-v1:131072 actions, final2.506s/94.622083deg, position gate
failure with TH/FF/RF supporting;95truth features, not deployed. Historical
scores remain under docs/history and outputs/imported, never reattributed.

Teacher-v2 is ready but the formal128-update trial has not started. New factor:
rotation reward bounded to1/s at.2rad/s, same5-point failure penalty and physical
gates, explicitly warm-started u64 weights with new optimizer. See
[fixed protocol](experiments/2026-10-08-teacher-v2/PROTOCOL.md).
Hora/AnyRotate/Sharpa references are copied and source identity retained; no
Boya port or SOTA comparison is complete. Observation-limited student training,
randomization calibration and hardware evaluation remain pending.
