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

The initial push pause ended when the user created and identified
https://github.com/wangrp1997/tendon-spin on2026-10-08 and explicitly requested
pushing this learning project. Continue concise milestone commits and pushes;
training outputs and weights remain local. Original repository remains outside
this work's writes.

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


[用户要求的在线核查](experiments/2026-10-08-boya-isaac-physical-v2/ONLINE_REFERENCES.md):
0new physics. Primary PhysX5.6.1 documents drive/contact/limit/mimic competition,
post-mimic limit/contact ordering and TGS/PGS friction differences. Actual native
four active mimic solref=[.002,1],solimp=[.95,.995,.001,.5,2]; explicit compliance
mapping missing from USD. This and28 named exclude gaps are concrete transfer
issues, not proof of an isolated SDK6.1 defect or a successful repair. v2 external
PD differs from the implicit-drive example; older issue4129 supplementary only.
Bounded minimal mimic/damping response isolation recommended, no extra launch.


[Latest Isaac failure video provenance](experiments/2026-10-08-boya-isaac-physical-v2/data/video_replay.json):
user-requested offline replay of initial+6 recorded world-body poses, including
terminal failure, two CAD views. Actual3ms held as10s/300frames/30fps; on-screen
labels distinguish recorded poses from new execution. Source NPZ/model hashes
verified, body-to-geometry roundtrip error8.66e-15.0new physics/training; no
rotation or repaired-interface claim. Next physical diagnosis remains pending.


[User-requested Sharpa code/asset comparison](experiments/2026-10-08-boya-isaac-physical-v2/SHARPA_COMPARISON.md):
0new physics. Sharpa22 active joints have authored motor armature0.00012–0.0032kg·m²
and external PD damping before actuator clipping. Boya has no explicit armature
configuration and routes source passive damping through solver viscous friction;
4mimic joints add a separate transfer requirement. These are concrete configuration
differences, not isolated causal proof or a tested fix. The latest video uses
MuJoCo CAD rendering of recorded Isaac poses, not an Isaac-native screen capture.
User now requests no additional checks beyond the requested work.


[Sharpa-style Boya actuator adaptation v3 completed](experiments/2026-10-08-boya-isaac-sharpa-v3/README.md):
user approved adaptation and quick delivery, superseding the pending v2 diagnosis
choice for this bounded repair. One fixed boya_sharpa_pd_hold_v3 original-grasp
episode,10000steps/5.000s,84.90s wall,time-limit stop,0reset/switch/train actions.
Max drift.125847mm,speed.904rad/s,link normal4.025412N,coupling2.988e-6rad.
Predeclared short-hold criterion passed; v2's3ms divergence not observed in v3.
Changes: external PD with total motor clipping, slave damping reflected to master,
solver friction0,declared uncalibrated armature A=4*D_effective*dt,28source collision
excludes restored,explicit NewtonMimic coefficients. Multiple factors changed;
not single-factor causation or physical-model equivalence. Native Isaac RTX video
recorded during this same run. No additional tests/scans; no rotation, full baseline
or hardware result. Complete Hora task/cache/training remains pending; additional
full contact/penetration validation was not silently claimed. Local commit only.


[User-authorized five-finger action layout](data/boya_fingers16_action_layout.json):
current learning/Isaac adapter defaults now16 actions (TH4+FF3+MF3+RF3+LF3),
wrists held,4distal mimics passive. Original LF limits/rates apply; no additional
small-finger amplitude restriction. Hora actor/student16 outputs and32-channel
proprioceptive frames; virtual touch5 fingers; standalone untrained deployment
MLP schema versioned to63 features. Current runner identity=v4 fingers16,
actions/record dimensions derived from layout; historicalv3 remains13-action5s
hold.0new physical execution/training in this interface change. Native teacher-v2
code/records retain their historical13-action contract.
Focused existing reference/deployment tests:5passed with existing env_isaaclab;
16-action student inference/history-only adaptation and five-finger measured-input
command boundary exercised. No extra simulator launch or training run.


[First16-input parallel grasp cache](experiments/2026-10-08-boya-parallel-cache/README.md):
64 Isaac environments ×8 batches ×.5s,82.539s wall,512 candidate episodes including
8 nominal anchors. All anchors passed;1/504 random candidates passed. Rejected first
for link normal>12N:400,drift>5mm:100,tilt>15deg:3. Original size/orientation/gravity
and v3 dynamics retained;16 finger actions,wrists held,passive mimics.
Hora±.25rad sampling adapted with source preload and declared Boya physical gates.
Per-step raw traces and one cache state saved;0PPO actions,not rotation/baseline success.
No complete penetration certification. Use only as limited PPO integration cache;
large-scale training still needs broader valid reset coverage. No parameter scan.


[Hora original-PPO Boya nominal pilot completed](experiments/2026-10-08-boya-hora-pilot/README.md):
64envs,16updates,8192actions/819200physics steps,121.031s wall,update-budget stop.
Original pinned PPO/ActorCritic/attribute encoder/reward reused with documented
Boya physics/obs/action/termination and dependency-loader adaptations. All losses
finite,model changed,final and every4-update checkpoints saved.760 exploratory
episodes ended(drift488,tilt198,force74). Single accepted cache state;no DR/student,
no original-state frozen evaluation,no sustained rotation/full baseline/hardware claim.
No global installs or original-repo writes. Per-physics-step training data retained.

[Cache sampling diagnosis](experiments/2026-10-08-boya-parallel-cache/sampling_diagnosis.json):
0new physics. Force-rejection first-step quartiles are all1 (max33).26 random
candidates meet final.1s criteria alone,which does NOT revise the1/504 full-prefix
accepted cache. Asked user whether to separately declare.5s initialization then
.5s strict holding in one bounded follow-up; no follow-up launched pending answer.


User approved one two-phase follow-up: .5s initialization settling then .5s strict
hold screening,504 random candidates/8anchors/<=300s. No four-finger requirement:
old and new cache rules both require2 observed hand groups in90% of final.1s.
[Predeclared protocol](experiments/2026-10-08-boya-settled-cache/PROTOCOL.md)
retains original pose reference, all formal physical thresholds and initial diagnostics.
This supersedes pending-answer text above; no change to historical1/504 result.


[Approved two-phase cache completed](experiments/2026-10-08-boya-settled-cache/README.md):
64envs×8batches×(.5s init+.5s strict hold),16000batched steps,157.103s wall,
batch-budget stop.28/504 random candidates accepted;8/8 nominal anchors passed.
Formal first failures:drift474,tilt2. Initialization diagnostics retained; no mid-
episode resets/switches or changed object reference. Same16input/size/gravity/PD.
Historicalv1 full-prefix1/504 is separate and unchanged; different screening phase
is not a controller performance improvement ratio. New28-state cache saved under
outputs/boya_settled_cache_v2/grasp_cache.npz.0training actions this run.
Accepted final.1s minimum observed finger-count distribution:2fingers10,3fingers15,
4fingers3. Both cache versions require>=2 hand groups in90% of final.1s,not4fingers.
Counts alone do not certify force closure. Full robustness and baseline pipeline
remain pending; no unsupported continuous-rotation/cache-reset success claim.


[Cache28 learning stage](experiments/2026-10-08-boya-hora-cache28/README.md)
initially launched64envs from scratch,then explicitly interrupted when user requested
more parallelism with desktop headroom. Last completed12updates/6144actions;
no evaluation. SIGINT shutdown exposed incomplete-control trace export KeyError:reward;
last result remains stale update-completed text,not a final stop record. Complete
12rollouts retained,incomplete13th not saved; do not infer its action total.

[128-environment training started](experiments/2026-10-08-boya-hora-cache28-128/README.md):
new fresh seed43,28-state cache,128envs×horizon8×64updates=65536requested actions;
minibatch512,5epochs,same nominal task/physics/reward. User requested modest GPU
scale-up and responsiveness;headless/no-camera,nice+10,Torch/OMP/MKL4threads.
First2updates ran atabout100actions/s;observed whole-card useabout3.8GiB/16GiB.
Sampled device-memory12GiB cap and1800s training cap. Checkpoint every16updates;
separate final frozen original-state120s evaluator automatically queued after
training. Current stage is RUNNING,not a rotation result or completed full baseline.
Live state under outputs/boya_hora_cache28_env128_stage1/stage.json and training/
result.json. New counts exclude interrupted64env samples. Train stop signals now
request saving at a PPO-update boundary. No concurrent old64env job remains.


[128-env stage completed](experiments/2026-10-08-boya-hora-cache28-128/README.md):
65536actions/64updates/6553600physics steps,635.979s train process,104.283actions/s
training loop,max sampled3838MiB GPU. Normal update-budget end. Separate fixed-final
original-state evaluation actually executed: .4845s total/.484s valid prefix,
+34.581409deg net,drift>5mm stop,0resets/switches,120s window incomplete. No sustained
rotation or complete physical/hardware certification. All jobs now exited. This
supersedes the earlier RUNNING launch entry without altering its snapshot.

User authorizes one bounded512env throughput trial (8updates/32768actions,<=300s),
[protocol](experiments/2026-10-08-boya-env512-throughput/PROTOCOL.md). Same task,
nominal dynamics,28grasp cache;fresh training identity,not continued/spliced control.
CPU nice+10,4threads,whole-device10GiB sampled stop cap,1Hz resource monitoring.
Interrupt-export bug repaired with explicit missing-control-field masks; no PPO
or physics changes. No further simulator check/parameter scan added.


[512env throughput trial completed](experiments/2026-10-08-boya-env512-throughput/README.md):
8updates/32768actions,106.023s process,update-budget/exit0.
Training-loop349.97actions/s versus128 first32768
103.92/s (~3.37x). Max sampledwhole-device
4008MiB/16303MiB,GPU utilization peak
78%,nice+10/4threads. No desktop
latency measurement or512frozen-policy evaluation; resource test only,not rotation success.
All processes exited.512recommended for subsequent training,no further run launched.

User questioned short training budget. Rechecked Hora PDF p15 Optimization Details:
16384envs,8steps,5epochs,batch32768,~100000gradient updates,~500Magent steps.
65536actions=0.0131%of that; prior10.6min stage was budget completion,NOT converged
training. Repo1.5Bconfig is a cap,not paper measured total. Paper7000h is simulated
real-time-equivalent,not wall training. At current512 throughput,500M arithmetic
extrapolation~16.5days; no success/time guarantee.


User requests further speed beyond350actions/s. One separately declared1024-env
throughput follow-up is authorized (same32768sample budget,4updates,horizon8,
minibatch512,seed43). Read the512results;GPU4GiB/median63%left headroom.
[Protocol](experiments/2026-10-08-boya-env1024-throughput/PROTOCOL.md) retains
all task/physics conditions and full logging; no larger sweep or policy success claim.


[1024env throughput completed](experiments/2026-10-08-boya-env1024-throughput/README.md):
4updates/32768actions,75.147s process,update-budget/exit0. Training-loop
581.17actions/s versus512349.97/s,
+66.1%at same sample budget. Peak observed
whole-card4226MiB (~4.13GiB),GPUutil peak75%,childRSS~6.73GiB. nice+10/4threads,
headless/camerasoff,full logs and physics/control intervals retained. No frozen
rotation evaluation or measured desktop latency.1024 is current throughput winner;
no bigger scan or further training launched. Preserve all old measurements and
128-env early rotation result separately; sample budgets still far below paper.


User authorizes one 2048-env trial and explicit desktop resource protection.
[Protocol](experiments/2026-10-08-boya-env2048-guarded/PROTOCOL.md):2updates/32768actions,
same nominal task/cache/seed/physics; new factors2048envs,independent watchdog and
systemd per-job CPU4core quota/12GiB soft+16GiB hard RAM/no swap/nice10 limits.
Synthetic-telemetry protection verification passed10decision cases and2tiny actual
process cases(cooperative exit and forced cgroup kill),effective limits read back.
0robot physics in those checks. Actual2048 throughput result pending; no further
scale scan or long training authorized in this bounded trial. GPU watchdog10GiB
stop/12GiBkill;systemMemAvailable12GiBstop/8GiBkill;external300s runtime cap.


[2048 guarded trial completed by protective stop](experiments/2026-10-08-boya-env2048-guarded/README.md):
54.280s process/exit0,14336of32768actions,0completedPPOupdates. System memory
PSI full avg10 crossed predeclared5%line at49.203s;cooperative save and exit followed
in5.08s,noTERM/KILL. Saved current policy/optimizer/RNG and interrupted raw trace;
no new gradient update and no comparable full2048 training speed. Child/service exited.
PeakGPU4532MiB(4.43GiB),peakchildRSS8493.54MiB,cgroup10021.90MiB,
minsystemMemAvailable39980.16MiB;no capacity-limit trigger. GlobalPSI source is not
isolated;do not infer2048hardware infeasibility or guaranteed1024guarded performance.
Two synthetic tiny-child protection cases plus10policy checks passed,separate from
real2048data. Retain1024as preferred next size from historical581.17actions/s,
which predates these cgroup limits. No further scan/long train/frozen-policy evaluation
launched;no sustained-rotation claim. New status supersedes preceding pending entry.


User approves lightweight training logs and no new resource-stop guard for1024;
requests first10M actions with continuation,then asks to see short visualization
BEFORE long training. [Preview protocol](experiments/2026-10-08-boya-hora-1024-preview/PROTOCOL.md):
fresh1024×8×8=65536actions,summary-only training;unchanged physics/PPO;paired atomic
resumable checkpoints and CPU original-PPO next-update equivalence passed. Thenone
nativeRTX5s-window original-state frozen evaluation with fulltrace,physicalgate stop,
0.2x labelled video. Preview pending;long10Mrun NOT launched and requires user review.


[1024 lightweight/resumable short preview completed](experiments/2026-10-08-boya-hora-1024-preview/README.md):
8updates/65536actions,100.520s process,800.10actions/s loop,updatebudget/exit0.
Summary logging + checkpoints/source/TensorBoard total13.72MB,no training fulltrace;
allphysicsgates unchanged,noresourcewatchdog/cgroup/automaticGPUstop. Fresh seed43;
paired learner snapshots support explicit cache-reset resume,CPU original-PPO next-
update equivalence passed. Final frozen original-state nativeRTX evaluation:
.3560sactual/.3555svalid,+24.05925deg net,drift>5mmstop,0reset/switch,5swindowincomplete.
Video4.75s includes labeled.2xplayback +1sinitial/2sterminalstills;not4.75s execution.
Full evaluation traces retained. No sustained/converged/baseline/hardware claim.
1000万 long training NOT started;latest user explicitly wants video review first.
Future continuation from this exact final checkpoint includes65536in10002432rounded
target;do not sum independent historical runs. This supersedes preceding previewpending.


User reports white preview. Actual saved frame confirms no visible hand/cylinder;
previous video-delivery claim is corrected: encoded successfully but visually unusable.
Original numerical execution remains separately recorded. [Rendering correction](experiments/2026-10-08-boya-hora-1024-preview/VIDEO_FIX.md):
RTX per-env partitioning hides env_0 geometry from external/World spectator camera.
One single-env rerecord with documented process-only partitioning=0;original source
hash checks and checkpoint/task/physics intact. Corrected execution/video pending;
1000万 training remains unstarted awaiting user review. No package/global changes.


[Video delivery corrected and restyled](experiments/2026-10-08-boya-hora-1024-preview/VIDEO_FIX.md):
Original encodedpreview was white-only and not a usable visual result; headline/link
explicitly corrected. Single-env RTXpartition=0 rerecord,all original source hashes
checked,samecheckpoint/physics/task: .356sactual/.3555svalid,+24.05925deg,drift>5mm,
0reset/switch. Separate execution preserved underboya_hora1024_preview_visible_v2.
User then requests Sharpa-style floor/background. NativeIsaacRTX studio_v3 renders
its archived measured world-link poses with checker referencefloor/shadows/orange
object;floorCollisionAPI absent,0newphysics/training/nointerpolation. Viewed initial
and terminalframes: hand/object/floor visible. New video4.75s includes explicitslowmo/
stills,not extra execution. Previewwrapper and renderer only;physics/training sources
untouched,existing resume contract remains compatible. Long10M still NOT launched.


User approved corrected visual preview and explicitly requests background formal
training and TensorBoard. Prior video-review gate is satisfied. [10Mprotocol](experiments/2026-10-08-boya-hora-1024-10m/PROTOCOL.md):
resume exact65536-action/8update checkpoint to10002432roundedcumulativetarget with
1024envs,same physics/PPO/cache;no guard/resourcecutoff. Summarylogs/pairedsnapshots
first+every16updates,TBeventsflushedeachupdate. Preview TB event tags read successfully;
add existing totalreward/episode-length/terminationstatistics toformalcurves.
Launch pending;one120s-window original-state frozen final evaluation queued only on
normal targetcompletion. No training/convergence/rotation result claimed yet.


User corrects initialization: formal training FROM SCRATCH. Then authorizes testing
resume first using the already launched continuation. The existingboya_hora1024_10m_v1
is now resume_test,notformal;gracefulstoprequested. Preserve actualsamples/lineage and
source/protocol,nopooling. After resumedmodel/norms/Adam/count verification,launch
[separate fresh10Mprotocol](experiments/2026-10-08-boya-hora-1024-fresh10m/PROTOCOL.md)
with fresh seed43 network/normalizers/Adam and no --resume. TB labelsformal_fresh,
resume_test,preview separate. This supersedes the previous continuation launchintent.


Resume integration PASSED without additional simulation launches:model+obs/value
normalizers restoreexactly,Adam640→720,actions65536→73728,epoch8→9,weightschange,
finite losses. [Stopped continuation](experiments/2026-10-08-boya-hora-1024-10m/README.md)
records21completeupdates/172032cumulative/106496added inlast-update snapshot;periodic
checkpoint16/131072islastsafe. Requestedstop hitIsaac-installedSIGTERMhandler(exit143),
no normalfinalsave/eval;oldresultstatus isstale,notcompletion. ReinstallPythonstop
handlersafterIsaacinitializationforfreshrun. Formalzero-start stilllaunchpending;
none ofresume-testdata willbe counted asformaltraining.


[Fresh formal1024 training is RUNNING](experiments/2026-10-08-boya-hora-1024-fresh10m/README.md):
user correction applied after actualresume-testverification;freshseed43 model/norms/
Adam,zero counts,no--resume. Firstcheckpointupdate1/actions8192/Adam80verified;
initialmodel matches freshseed and differsfrompreviewtrained,separatelineage.
Launchsnapshot14updates/114688actions,~849.2/s;
target10002432roundedsamples,nonepooled. Noresourcewatchdog/cgroup/autocutoff.
Summarylogs,pairedcheckpointsevery16updates,TBhttp://127.0.0.1:6006 formal_fresh/.
ActualHTTP confirmsreward/loss/rotation/duration/terminationcurves;preview/resume_test
separate. StagePID2030161,trainPID2030163;background detached.
Finaloriginal-state120s-windowevalqueuedafterbudget,NOT executed. No convergence/
rotation/completebaseline/hardwareclaim. Supersedesfreshlaunchpendingstatusabove.


[Sim-to-real citation follow-up](research/2026-10-08-sim2real-followups/README.md):
0newphysics/training/hardwarecommands. SemanticScholar returns220 Hora citations and71
AnyRotate citations(overlapping);OpenAlex429sharedbudget retained;arXiv targetedmetadata
and6new fulltexts verified. DexNDM:real-data joint dynamics+residual actions;
DexCtrl:learned action+PD adjustment;PTLD:real tactile latent supervision with external
pose during collection;ReDex:human finger corrections then standaloneBC;
WM-Craftnet:predictive recurrent context;MPC-scaffoldedSAC:actual real-world policy
updates but workingMPC+OptiTrack prerequisites. Online latent/gain changes,offline
model fitting,and online weight updates kept distinct. No validated Boya safety
ortransfer claim;current nominal teacher unchanged. Prioritize verifying actuator
response and real tactile supervision prerequisites before selecting a new method.


[Teacher stopping criteria source review](research/2026-10-08-training-stop/README.md):
Hora and official Sharpa PPO stop on agent-step caps and save best recent reward;
no reward-value graduation branch. Hora paper reports ~500M samples (public YAML
cap1.5B); Sharpa YAML300M. Paper evaluations use finite-window rotation andTTF/TTT,
not universal required turns. Boya120s signed-prefix metric retained; current5mm/
15deg/12N gates differ from reference terminations. Proposed20-rollout/18full-window
and3-checkpoint/<5%gain heuristics are explicitly DISCUSSION ONLY,not paper thresholds
or new active acceptance rules. Poor-performance plateau means pause/diagnose,not
successful graduation; nominal teacher completion does not complete sim2real.
0newphysics/training/evaluation launches;existing10M and queued single evaluation
remain their original experiment. Source hashes and exact evidence archived.


[Termination mismatch follow-up](research/2026-10-08-training-stop/README.md):
User challenges reproduction mismatch; confirmed inheritedBoya5mm/15deg/12N are
trainingdone+evaluationstop,not mere logs. Horaoriginal only objectheight/time;
Sharpa relativeheight±20mm/time;AnyRotate keypointdistance.1/axis45deg. Currentport
is customtermination,NOT completeHora reproduction. Proposed numeric graduation
heuristics in preceding review are NOT adopted;priority is aligning source semantics.
No code,threshold,trainingbudget or execution changes;historicalresults retained.

[Fresh10M and queued final evaluation COMPLETED](experiments/2026-10-08-boya-hora-1024-fresh10m/README.md):
10002432actions/1221updates,trainexit0/updatebudget. Frozenfinal originalgrasp44,
0reset/0switch,120srequested/2.1485sactual/2.148svalid,+63.626313degnet+peak,
1.042972degbackward;evalexit0,stoptilt>15deg. Firstinvalid tilt15.005651deg,
drift2.559715mm,maxnormal3.683823N,finite. Customtilt-limit termination is not
established drop or originalHora capacity limit;no execution afterstop. Noautoresume,
DR,student or baselinesuccess claim. Small evidence saved;rawtraces+weightsignored.
Supersedes freshRUNNING/queued-eval status;priorrecords preserved.


[Height termination correction reviewed; fresh restart authorized](experiments/2026-10-08-boya-hora-height-10m/README.md):
User explicitly preserves weights and authorizes correction/review/newbackground10M.
Newhora_height profile directly extracts originalHora height/time method;translate
canonical.65/reset.645 toBoya nominal.10606109974/reset.10106109974m,onefixedplane
across cache resets. Heightchecked20Hz;old3D5mm/15deg/12N nowdiagnostics/shadowscores.
Numericalfinite/speed/mimic remain. Train+evalshareexactspec;resumecontract recordsit.
10focusedtests+compile/diffcheckpassed. OriginalPPO/reward unchanged;action/engine/
28cache/noDR/privileged differences documented,not completeHora reproduction.
1024freshseed43/10002432roundedactions,nooldweights;launchpending,0newGPUtests.
Oldweights/results untouched. Newprotocol+review saved before launch.
