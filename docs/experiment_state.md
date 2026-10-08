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

[Teacher-v2 completed](experiments/2026-10-08-teacher-v2/README.md):128
updates/262144actions/26182209physics steps/934train episodes/1028.31s total wall,
update-budget stop. Original-state120s evaluations: imported initialization
2.506s/94.622083deg drift;u32 120s/14.833574deg time limit;u64
110.804s/1.466386deg drift;u128 120s/18.397394deg time limit. Every physics
and stored-observation inference replay error0; reset0/switch0. Original18
source snapshots are intact. Current-file attribution comments were added only
after completion; no reattribution to new adapters.

Finalu128 last30s−.200283deg and15.774726deg already at5s indicate mostly
holding after early motion; repeated rotation remains unresolved. u32 meets the
predeclared finite120s/positive-tail sign criterion but with only+.146616deg
in its last30s, not proof of repeated gaits. Freeze this reward diagnostic;
no reward/gain/candidate scan is opened. Full baseline reproduction remains next.

[Archived action diagnosis](experiments/2026-10-08-teacher-v2/data/action_diagnosis.json):
0new integration. After30s six position targets (THJ4/THJ1/FFJ4/MFJ2/RFJ4/RFJ2)
remain at software bounds with outward actions100%; clipping removes requested
motion. Last30s mean cumulative target/actual travel per motor .001926/.002434rad.
These are command limits, not proof of physical hard stops. Reward reconstructed
to1.78e−14; no isolated claim that reward or sample budget alone caused failure.

[Reference reuse](reference_reuse.md): original Hora ActorCritic/embedding/
normalizers directly loaded, Sharpa parameterized original30-frameTCN adapted
to13-action/26-channel Boya. At original width,TCN outputs agree exactly;
student tensor check updates only the history encoder.8tests pass; these
network/feature diagnostics are0task episodes, not trained baseline successes.
Hora/AnyRotate/Sharpa reference identity and licenses are retained in references.

User confirms point-force tactile arrays and net3D fingertip forces.116 is
local ROS message capacity, not confirmed physical active-point count. Following
AnyRotate/Sharpa, initial simulation uses contact-feature observations and does
not need the full raw grid. Calibration, full teacher/history/tactile port,
student training and hardware evaluation are pending. Keep nominal teacher
truth distinct from virtual sensing and real measured student observations.

[Original-paper/baseline audit](baseline_evidence.md): Hora Figure3 uses radians,
23.96rad≈3.81turns in a30s real-world window, mean normalizedTTF.98; AnyRotate
palm-upz1.57turns/30s. Neither proves indefinite rotation. Sharpa quantitative
lifetime/paper is not verified. Hora reports≈500M training actions; v2 has262144
new actions and131072 in its warm-start history. Our sole fixed grasp was CAD
static-force optimized/preloaded/unsupported-held and reloaded, not a paper grasp
cache; its saved bytes match the original. Complete Hora pipeline is the priority
requested by the user; task/cache/randomization/teacher/student training are still
pending, and any Boya port must retain its distinct hand/engine identity.

User pauses push until the new remote is created/identified. Continue concise
local milestone commits; original repository remains outside this work's writes.
