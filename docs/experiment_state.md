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
palm-upz1.57turns/30s. Neither proves indefinite rotation. The RL Lab quantitative rotation lifetime remains unverified; the follow-up
separately verifies20s TacBPM paper results and official Sharpa source identity. Hora reports≈500M training actions; v2 has262144
new actions and131072 in its warm-start history. Our sole fixed grasp was CAD
static-force optimized/preloaded/unsupported-held and reloaded, not a paper grasp
cache; its saved bytes match the original. Complete Hora pipeline is the priority
requested by the user; task/cache/randomization/teacher/student training are still
pending, and any Boya port must retain its distinct hand/engine identity.

User pauses push until the new remote is created/identified. Continue concise
local milestone commits; original repository remains outside this work's writes.

[Metric literature/source follow-up](research/2026-10-08-rotation-metrics/README.md):
8full-text sources, bounded OpenAlex/Crossref searches; arXiv broad API429/timeout
retained. Fixed-window angle+lifetime convention; preserve120s Boya primary,
add30s, first-target/unreached/tail/backward reporting.4teacher archives offline
rescored,0new physics, identical old endpoint angles. Window-specific physical
failure separated from full-source episode failure.4analytic metric tests pass.
Official Sharpa95ccda3d... pinned separately:3task ASTs differ from user fork,
models/normalizer/PPO ASTs match;20s resets/GIF looping do not prove indefinite
rotation. TacBPM20s signed806/945deg examples are a different2026 paper/method.

[Isaac official-headless diagnostic](experiments/2026-10-08-isaaclab-probe/README.md):
user accepted process-only EULA and this one<=5min follow-up after direct150s
startup timeout. AppLauncher returned/reset/3nonrender physics steps returned,
exit0/3.951799s; normal fast shutdown terminates process before post-close marker,
which was not observed and not claimed. Existing environment unchanged; no Boya
model/teacher/student/train actions. Startup blocker has a working official
entrypoint. Next mandatory boundary is collision/coupling/13input/grasp transfer,
then [complete Hora Boya port](hora_boya_port.md), not another v2 reward scan.


[Boya structural import](experiments/2026-10-08-boya-isaac-import/PROTOCOL.md):
4.440463s/exit0;22 revolute names,4 NewtonMimicAPI schemas and CAD collisions
present. Zero dynamics; schema presence does not verify physical coupling.

[Boya physical-v1 failed transfer](experiments/2026-10-08-boya-isaac-physical/data/failure_diagnosis.json):
34steps/.017s, nonfinite joints. Offline audit confirms WXYZ passed into Lab3
XYZW and outputs misinterpreted: root error180deg, distal position errors up to
181.6mm. Retract the raw phase's original-state-restored assertion; no eligible
original-grasp prefix, max-tilt0 invalid. Efforts were requested, not total actual
torque; zero contact matrix used wrong paths and is not proof of no contact.
Explicit damping risk quantified offline, not isolated full-cause evidence.

User now approves quaternion/contact-path/solver-damping correction and one fresh
single-scene<=300s verification. [Physical-v2 protocol](experiments/2026-10-08-boya-isaac-physical-v2/PROTOCOL.md)
declares unchanged physical grasp/size/gains/coupling, pre-step FK/state/solver
readback gates and separate v2 evidence. Large cache/PPO work remains dependent
on physical-contract verification. No reward/gain/grasp/backend scan authorized.


[Approved physical-v2 completed](experiments/2026-10-08-boya-isaac-physical-v2/README.md):
5.071286s process/exit0; original-state FK/commands/solver-damping readback verified.
All24 body poses match within.000135mm/.000284deg. Zero actions, unchanged commands,
sourcePD reconstructed within1.49e−8Nm and actuator/PhysX-actuation effort error0.
Actual6steps/3ms then drift5.856909mm; speed3620.99rad/s, max link normal668.87N,
coupling errors up to4.16rad. Planned.5s hold incomplete; no certified physical
prefix/rotation score/training success. New sensor matrix24×3 is observable;
self-contact impulses and complete physical gates still absent. CCD effectively
GPU-disabled. V1 pose failure and raw tilt/error attribution corrected separately.

Offline XML/USD audit finds28 source named collision excludes not explicitly
ported; adjacent PhysX filtering may be automatic, causal contribution unknown.
NewtonMimicAPI is current documented schema, so missing legacy mimic API alone
is not evidence of a bug. Runtime coupling/damping/inertia/contact behavior needs
isolation. No further launch or parameter scan performed. Stop dependent large
cache/PPO work pending user's choice of a bounded minimal interface diagnosis.
3 coordinate regression tests passed; existing environment/original repos unchanged.
