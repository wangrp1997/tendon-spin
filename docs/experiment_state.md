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


[Corrected height-rule fresh training RUNNING](experiments/2026-10-08-boya-hora-height-10m/README.md):
Actualdetached launch supervisor2433176/train2433207,1024headlessenvironments,
newzero-initseed43,nooldweights. Initialnetwork exactlymatches verifiedfreshseed43;
checkpointprofile/lineage verified. Snapshot11updates/90112actions,738.8/s.
Newtaskdone counts olddrift/tilt/force=0;diagnostics retained. TBactualHTTP confirms
hora_height_fresh andoldcurves. Requested10M/rounded10002432;120s originalstate
finaleval queuedonlyon normalbudgetcompletion. No convergence/rotationclaim.
Code84109c7pushed;oldoutputs preserved. Supersedes preceding launchpending status.


[Matched teacher comparison prepared](experiments/2026-10-08-boya-hora-matched-eval/README.md):
User authorizes firstitem:old/new checkpoint commonconditions. Newindependent
comparison code reuses existing frozen evaluator unchanged;runtime source hashes
and originaltraininglineages remain distinct. Samegrasp44/heighttermination/120s
(+30s supplementary)/10002432trainactions/seed43/ownnormalizers. Reuse queuednew
finalevaluation,then only1additionaloldcheckpoint episode afterstagecompletion.
13focusedCPUtests+compile/diffcheckpass;0newphysics. Actualmanifest/queuepending.
No old63.626313deg score inserted as if commoncriteria. Initialstate/source/model/
checkpoint/count mismatches block reporting. Singlepair,notSOTAstatisticalbenchmark.


[Matched comparison QUEUED](experiments/2026-10-08-boya-hora-matched-eval/README.md):
Realmanifest preparation passed source/observationAST/cache/budgetidentity checks.
DetachedCPUwaiterPID2540127,waitingfortrainingandexistingfinalevaluation;0additional
GPUepisodes launched. On normalcompletion,reuse newevaluation and runONE oldpolicy
under commonheight rules,thencompare actualinitialstates and emitJSON/CSV/Markdown.
Oldpolicy's ownnormalizer andoriginaltrainingprovenance retained viaexplicit
inferenceexecutioncontract;activehashedtraining/evaluator files untouched.
Pendingtable has no claimed scores. Supersedes preceding queuepending state.


[Height-rule training and matched queue STOPPED by user](experiments/2026-10-08-boya-hora-height-10m/README.md):
2026-10-09 00:07 user requested stop before discussing real drop detection. SIGTERM
ended learner at complete update495/4,055,040actions; resumable teacher_final saved,
oldweights/logs preserved. stop_reason=signal requested stop at update boundary;
status=completed denotes orderly exit,NOT10M completion. Supervisorphase=stopped,
finalevaluation notlaunched; comparisonqueuephase=stopped/0additionalGPUepisodes.
AllthreePIDs exited. No fixedpolicy score or matched comparison exists. SourceHora
.645mplane is real,but.650mcreationheight is not its cachedresetdistribution:
Boya nominalminus5mm remains an unvalidated crosshand assumption,not proof ofdrop.
Heightcounts cannot establish loss ofsupport or that previouslearning was useless.
No rulechange/restart/newexperiment. Supersedes priorRUNNING/QUEUED;fullhistory retained.


[Stopped height-run weights DELETED by explicit user request](experiments/2026-10-08-boya-hora-height-10m/weight_removal.json):
Removed only thisrun33.pthfiles/100,979,714bytes,includinginitial/periodic/final.
Olderweights andalllogs/curves/identities retained. Previoussavedcheckpoint statements
are historical;thisrun nolongerresumable. Checkpointmetadata preserved ashistory;
automaticcomparison remains stopped. No newterminationrule/restart/experiment.
Discuss geometry-relative escape plus persistent motion/contact evidence; exactBoya
boundary remains undecided,not a claim that heightcrossing equals actualdrop.


[Workspace-rule fresh training authorized](experiments/2026-10-09-boya-workspace-10m/README.md):
User requests repair/restart promptly,no extra checking. FixedCAD-derivedlowerz
.066801253194m(palmtop+cylindersphere),39.26mmdescentfromnominal; middle/distalXY
mesh-envelope expanded by objectradius, allfivefingers.2consecutive20Hzoutside
samples required; partialreset clears history. Contactcountdiagnostic only.
Explicit taskregionapproximation,not verifiedrecoverability. Sameengine/grasp/
PPO/reward/28cache/privileges/budget; newfreshv3 planned. Firstold80weights retained.
Formerheightcomparison remains stopped,not silentlymigrated. Launchpending.


[Workspace-rule fresh training RUNNING](experiments/2026-10-09-boya-workspace-10m/README.md):
Actual2026-10-09 00:32:56launch supervisor2621876/train2621877;
freshseed43,noresume,1024headless,10002432roundedbudget. Snapshot7updates/
57344actions/738.0actionspersecond;checkpoint saved.
8targetedterminationtests+compile passed,noextraGPUruns. TBworkspace_freshadded,
oldcurves/firststrict80weights retained. Geometryrule/source identity archived;
queuedfinalevaluation only after normalbudget. Oldmatchedqueue remains stopped.
Supersedes launchpending;no convergence or tasksuccess claim.


[First/third FINAL checkpoint comparison QUEUED](experiments/2026-10-09-boya-workspace-matched-eval/README.md):
User explicitly enables two-weight automatic evaluation after currenttraining.
NewCPUwaiterPID2650612 confirmedwaiting/0additionalGPUepisodes. Sameoriginalgrasp44/
currentboya_workspace/120s(+30s)/10002432actions/seed43/ownnormalizers. Uses each
run FINALcheckpoint,not bestrewardselection. Reuse newexistingfinaleval then
oneoldpolicyepisode. Priorv2heightqueue staysstopped; this is anewplan.
11CPUcomparisonchecks andactualmanifestpreparepassed. Narrowknowncontactdiagnostic
sourceextension accepted only after entiremoduleASTcomparison; otherphysics/core
hashes andobs/scoringASTs preserved. Activehashedtrain/evaluator/geometry untouched.
No currentmatchedscore,no extraGPUruns duringtraining. Errors/manualstop prevent
automaticlaunch/retry. Singleseedpair,notSOTA/indefiniterotationproof.


[Workspace10M training and two-final-checkpoint comparison COMPLETED](experiments/2026-10-09-boya-workspace-matched-eval/README.md):
Training normalbudget10002432actions/1221updates,finished2026-10-09 04:21:50;
comparisonfinished04:25. Samegrasp44/workspace/120s/seed43/ownnormalizers; independent
initialstates match exactly,0resets/0switches,2episodes total(newreused+1old).
Oldfinal:+142.645560deg/3.3495svalid,peak184.816125/backward43.463820;
newfinal:+182.935623deg/5.5995svalid,peak182.935623/backward1.266059.
Bothstop belowBoya manipulationregion; neither completed30/120s. Validprefix
maxdrift/tilt/normal:old98.229mm/91.575deg/76.173N,new74.425mm/77.904deg/56.906N.
New+40.290deg and+2.250s inthispair,but still largetilt/earlyfailure;not stable or
indefinite rotation/SOTA/hardwareproof. Traininglast10reward16.414,meanvalid4.591s,
meannet96.050deg; keep separate fromfrozenresults. Oldstrictshadow new7.499deg/.290s
(drift),old63.626deg/2.148s(tilt); historicalscores preserved,not pooled.
Supervisors/train/queue exited;weights/rawdata retained. Smallcompletionevidence
archived,no newexperiment orautoresume. Supersedes priorRUNNING/QUEUED status.


[First formal run weights DELETED after completed comparison](experiments/2026-10-09-boya-workspace-matched-eval/old_weight_removal.json):
Explicituserrequest: deleteoldweights anddelivernew182.94DEGREEevaluationvideo.
Removed firststrict10M80.pthfiles/247715579bytes only; currentworkspacev3weights
andallrawtraces/logs/results preserved. Historicalcheckpointpaths/hashes retained,
not availableoldweights. Video requested from exact5.600s archivednewepisode,
not182.94seconds,no newcontrollerexecution.


[Newfinal evaluation video DELIVERED](experiments/2026-10-09-boya-workspace-10m/evaluation_video.json):
ExistingIsaacRTXstudiorenderer replayed exactarchivedworldposes ofnewfinalepisode,
5.600sphysical/+182.935623degnet,0.2xslowmotion includingterminalframe. No newphysics,
controllerexecution orMuJoCorendering; referencefloorvisualonly. Encodercompleted.
Video outputs/boya_workspace182deg_video_v1/isaac_native.mp4. Noextracheckpass.


[Workspace continuation to50M with10M evaluations AUTHORIZED](experiments/2026-10-09-boya-workspace-50m/README.md):
User explicitly asks implementation+resume. Currentv3checkpoint10002432/1221,
fulllearnerstate retained. Newfactorlargerbudgetwithdeclaredcacheresetbetween
segments; originalPPO/reward/physics/criteria unmodified. Sequential targets
20004864,30007296,40001536,50003968;4neworiginalgrasp44/120sevaluations,existing
10Mscore reusedonly. Wrapperrecords percheckpointnet/time/backward/tilt/stops,
JSON/CSV/Markdown+TBcurves; no anglepooling/images/bestrewardselection.3necessary
CPU scheduling/stoptests passed; launchpending. No extraGPUtrial/watchdog.


[Workspace50M milestone continuation RUNNING](experiments/2026-10-09-boya-workspace-50m/README.md):
Actual2026-10-09 08:59:42 launch orchestrator3453674/supervisor3453685/train3453686.
Restored1221updates/10002432actions withoriginalfinalSHA90419cc9...,model/norms/Adam/
LR/RNG/lineage; subsequent actualupdate1228/10059776actions,
newsession57344actions andcheckpoint observed. Targets20/30/40/50M
sequentialtrain+originalgrasp120seval, then stop. TBworkspace_continue50m added;
10Mbaseline reused, no latermetricsyet.3CPUtests+compilepassed,no extraGPUprobe.
Coretraining/physics/evaluator unchanged; neworchestrator only. Supersedeslaunchpending.


[Workspace50M COMPLETED; authorized offline diagnosis completed](research/2026-10-10-workspace50m-diagnosis/README.md):
Final50,003,968actions/6104updates,normalbudgetcompletion;orchestratorcompleted,
fourmilestoneevaluationscompleted. Eachfrozenowncheckpoint/normalizer,originalgrasp44,
0resets/0switches/0training. 10/20/30/40/50M netdeg182.935623/26.488664/67.009596/
115.525333/226.888668; validseconds5.5995/85/82.7/7.8495/35.8495;
peakdeg182.935623/35.822706/67.031312/115.525333/226.888668;
backwarddeg1.266059/85.410726/45.670142/2.004248/43.489802.
20/30Mstop1500swallbudget,not120scompletion;othersbelowworkspace.
50Mfirst30snet35.059deg,last.2s172.271deg,lastvisiblehandcontact35.722s,
114.649degafterlastcontact. Originalprimaryscoresretained,notfiltered/replaced.
Offline0newphysics/learning: commonfirst5s10M129.051degvs50M42.097deg.
20M20-30sactualpose-2.462degbutreconstructedrotationreward+.36769/control;
allphysicsreportedmoving-axisomega+.32288rad/svsactualpose-.00430rad/s.
Persistentvelocity/posemismatch,fixed/movingaxisdifferenceandtargetboundsare
directdiagnosticevidence;enginecause/forgettingcausalitynotestablished.
50M20-30s4/16targetscontinuouslyoutwardblocked. Last100trainingupdates
10M->50Mmeanvalid4.527->16.243s,net99.133->53.216deg,rotationreward.3453->.2465,
cumulativereward15.427->38.471. Ownmodel+normalizeractionreplaymaxerror1.88e-6.
Recommendreward/poseconsistencybeforeboundedshortcomparison;nochangetoengine,
grasp,observations,reward,criteriaorbudget;nonewtraininglaunched.
Weights/logs/rawtracespreserved.Smallscript/evidence/CSV/figurearchived;
supersedespreviousRUNNINGstatus. Oneepisodepercheckpoint,notindefiniterotation,
matchedSOTAorhardwarevalidation.


[Velocity/pose source-chain follow-up](research/2026-10-10-workspace50m-diagnosis/SOURCE_CHAIN.md):
User challenges premature reward-change recommendation. CurrentinstalledAPI/source
audit finds XYZW/worldomega/linear-angularordering/directstateconcatenation and
perstepfetch/update path consistent; no evidence of simple order/frame/refresh error.
Independent float64SciPy worldrotvec matchesarchivedspin within.000149deg.
20/30/50M20-30sfullomega-versusposerateRMSE.46945/.42260/.40432rad/s duringcontact;
50M35.73-35.8495s239no-visible-contactsamplesRMSE.00627rad/s at19.5rad/s.
Plus/minusonephysicsstep doesnotresolve20Mdiscrepancy. Localizescontact-associated
velocity/poseoutput mismatch,NOT causalproof ofTGS/positioncorrection/enginebug.
Prioradvice to immediatelychange rewardwaspremature: firstlocalizecause,thenchoose
adaptation/enginefix orseparatelydeclaredrewardvariant. OriginalHoraformulareused;
no basis to claimHora/Sharpaoriginalmethoddefect. Currentruntimehashesarchivednow,
notretroactivetrainingdependencyhashes.0newphysics/training/configurationchanges;
oldmetrics/weights/rawtracesretained. Sourcechain script+smallevidence saved.


[Frozen20M raw-velocity / iteration diagnostic COMPLETED](experiments/2026-10-10-boya-velocity-diagnostic/README.md):
User approved exactly two originalgrasp44/frozen20M/30s episodes, no learning.
Same20,004,864-action checkpoint/normalizer/seed43; initialq/qdot/object/commands
exactlymatch archived20M andeachother, corehashesverified. ActualUSD/config
onlydifference scene minvelocityiterations4->16; actor/articulationrequests
unchanged, TGSposition16, originalreward/observations/termination retained.
Original4:30.0000svalid/net30.426747deg/peak35.822706/backward37.240682,
normal30stime.16:28.3995svalid/net226.507333deg/peak226.507333/backward33.963902,
actual28.4000sstopbelowBoya manipulationregion.16at25s15.934230deg;
lastvalid.2s177.071443deg, visiblehandcontact21.25% ofsamples, finalvalidsample
hascontactagain;notallafterlastcontact. Keepprimaryscore,nocontactfiltering.
Maxdrift/tilt/normal4:18.488mm/14.889deg/85.236N;
16:154.888mm/119.923deg/100.203N. Oldstrictshadow4:8.755371deg/.7015s;
16:8.562029deg/.7025s,both5mmdrift.
Allrawbefore/afterLabandraw-versusLabpose/velocitymaxdifferences0.
Predeclared5-20s/both100%visiblecontact:fullomega/poseRMSE.585356->.859179rad/s,
actualnet13.038095->-1.769584deg, originalHora rotationterm+.347928->+.246675.
Original20-30sreproduces-2.461895degbutrotationterm+.367686/RMSE.469447.
16doesnoteliminatediscrepancy; rawvelocity/posemismatchalreadybelowLabreading,
specificsolvermechanismnotproven,noevidenceHora/Sharpaoriginalformuladefect.
16logsTGS>4velocityiterationrecentbehaviorchangewarning,verbatimretained.
V1setupfailureproduced0recordedepisodesteps/noinitialortracefiles;preserved,
fixednewloggeronlythenmanuallylaunchedV2. Exactly2actualepisodes,0training,
0resets/switches,normalchildexits,noextraGPUtrial/retry/video/watchdog.
Smallprotocol/scripts/results/hashes/execution/failureevidencearchived;
rawtrajectoriesandweightslocal. Necessarysyntax/source/initial/configcheckspassed.
Recommendvelocity/poseinputconsistencybeforeanyshorttrainingvariant; nofollowon
traininglaunched. Singleseedconfigurationdiagnostic,notsolvercausal/SOTA/hardwareproof.


[PhysX official-document web follow-up](research/2026-10-10-workspace50m-diagnosis/PHYSX_WEB_FOLLOWUP.md):
User requests internet search to distinguish engine/API causes. OfficialPhysX5.4.1
SolverIterations explicitly says reportedbodyvelocity neednotmatch finite-difference
bodymotion undercontacts/constraints; splitimpulse andTGSsubstep semantics documented.
Thus previous blanket wording velocitycorruption/numericalcontradiction was too strong:
documentedsolver/API semantics can explain a difference, notproof ofalocalenginebug.
Raw/Labexactagreement andmeasuredangle/rewardmismatch remainvalid observations;
specificcause/magnitude ofthisBoya contactcase stillunverified.
Newreferenceconfigurationfinding: vendoredHoraTGS yamlposition8/velocity0,
Boya scene16/4 plusarticulation16/4; recentdiagnosticB scene16/16.
Referenceyaml isnotanexecutedoriginalbaseline. Scene minimum0alone doesnotforce
effective0whenactors/articulationrequestmore. No0iterationtriallaunched.
FixedofficialCHANGELOGcommit517a007/v5.3.0 explainsformerTGS>4velocityiterations
wereconvertedtopositioniterations,nowhonoredliterally;matchesdiagnosticwarning.
Relatedpublicissues543/549notestablishedlocalhits;549requiresoppositeflags
fromouractualcontactlastfalse/externalforceseveryiterationtrue.
Officialpagesreadinfull;quotes/URLs/versions/hashesandaccesslimitsarchived.
0newphysics/training, nocore/configuration/rewardchanges. Prioritizematched
effective-solver/input-semantics verification beforechoosingafix orrewardvariant.


[Effective-zero velocity iteration diagnostic COMPLETED](experiments/2026-10-10-boya-velocity-zero/README.md):
User approved source/config verification plus exactly ONE originalgrasp44/frozen20M/
requested30s episode at position16/effectivevelocity0, reusing completed4; no rerun4.
Hora v0.0.1 default TGS8/0 transmits through taskdict/VecTask setattr to gym.create_sim;
no later vendored-source override found. Source audit only, not executed IsaacGym baseline.
Installed PhysxCfg max-actor/clamp semantics: scene velocitymin/max0/0 plus all25rigid
bodies and1articulation requests0, authored before reset and unchanged after initialization.
Position scene16/255 and all actor requests16 retained; only declared velocity fields differ.
No exposed native tensor iteration counter; effective0 established by composed configuration
and documented aggregation, not a claimed kernel measurement. Core/runtime hashes match4;
initialq/qdot/object/commands exactly match archived20M and4. Ownmodel/normalizer/seed43,
same observations/controller/reward/task/engine package/collision/dt, no resets/switches.
Zero valid9.0995s/net+254.360830deg/peak254.360830/backward6.129595;
actual9.1000s stop belowBoya manipulationregion, not complete30s. Maxdrift73.342mm/
tilt114.329deg/normal46.516N. Lastvalid.2s adds78.860580deg, visiblecontact68%,
finalvalidsample hascontact; preserveprimaryscore, no contactfiltering/successclaim.
Reused4:30.0000s/net+30.426747/peak35.822706/backward37.240682, normal30sstop.
Matched0-5s4->0: independent actualnet19.850709->32.176620deg;
fullomega/poseRMSE.549631->.273469rad/s, contactRMSE.548774->.194413;
raw/pose fixedaxis means .230851/.067873 -> .105176/.103940rad/s;
contact99.91%/99.97%, originalHora rotationterm+.238682->+.115013/control.
Common0-9.0995s fullRMSE.603245->.372778/contactRMSE.602842->.267102.
Rawbefore/afterLab andrawversusLab maxdifferences all0. Residualvectorerror remains;
0improvesagreement in thispair, notfullmechanism/enginebug/physicalfidelity proof.
Largerangle with earlydrop isnotstablecontinuousrotation. Old4-trainedpolicy under
newconfiguration isnota0-trainedpolicy. Candidateforfuture shortbudget matchedtraining,
not authorization tolaunch one.0trainingactions,1newphysicalepisode,0automaticretries.
25s/20-30s unobservedfor0. Correctedoffline25sfield fromclampedendpoint tonull;
retained firstanalysis output/log, rerananalysisonly, no physicsreplay.
10rawchunks/18200physicalsteps,18199valid; allraw/weightslocal; smallcode/protocol/
sourcehashes/comparison/execution archived. NoextraGPUtrials/tests/watchdog/video.


[Effective-zero small-budget teacher LAUNCH HISTORY; completed below](experiments/2026-10-10-boya-velocity-zero-1m/README.md):
Latestuser explicitly approves smallbudget freshteacher training with originalHora
reward under effective0 velocityiterations. New requested1,000,000/rounded1,007,616
actions,123updates,1024envs/seed43/newnetwork/normalizers/Adam/RNG, no resume.
Output outputs/boya_hora1024_velocity0_small1m_v1; start2026-10-10 12:41:56Shanghai,
supervisorPID2107045/trainingPID2107046. Launchsnapshot10updates/81,920actions;
first8192-action pairedcheckpoint verified, modelchanged/Adamstate nonempty.
New shared engineprofile tgs16_velocity0 in environment/train/eval/checkpointcontract;
default original_tgs16_4 retained forexplicitoriginalconfiguration. Sources/identity
versioned separately; alloldweights/results/source snapshots retained unchanged.
Actualall1024envs/25,600rigidbodies/1,024articulations pass composed0velocity/16position
requests before/after initialization,scenevelocitybounds0/0. Nointernalcounterclaim.
Reuse same28cache generatedunder4,originalHoraPPO/reward,privileged96+9/8latent,
16fingeractions/heldwrists/4passivemimics,object/grasp/collision/PD/timing/workspace
termination/numericalrules unchanged. Summarytraining/TensorBoard/pairedfirst-every16-final
checkpoints,noresourcewatchdog/memoryquota/trainingwallcutoff/video/extraGPUpreview.
Budgetcompletion autoexecutes ONE frozenfinal/originalgrasp44/30s-or-firstfailure/
1500swall evaluation,exactinitialstate check,fullphysics trace andoriginalreward
reconstruction; offlineSciPy velocity/pose/contact/reward analysis, thenSTOP.
Error/manualstop haltsdependentstages,noretry/automaticextension. No matchedbudget4
evaluation orsuperiorityclaim; old20M/50M arehistory,not equaltrainingbudgetcontrols.
Necessarysyntax/1CPUpairedlearnernextupdaterestore+engineprofilemismatchrejection
andactualfirstupdate/checkpoint validation passed. Startupisnotfinalrotationevidence;
finalevaluationpending. Smallprotocol/implementation/launchprovenance saved andpushed.


[Effective-zero small-budget teacher COMPLETED and STOPPED](experiments/2026-10-10-boya-velocity-zero-1m/README.md):
Fresh1,007,616actions/123updates completed normally; training/evaluation/analysis
exit0, totalwall1321.55s, noautomaticextension. FinalcheckpointSHA256
9b5b4ddb3ecb31fcfd1ba5ce4315bb5ad6320dca913c143bca0c481023fca5d4.
ONE frozenownmodel/normalizer/originalgrasp44 episode,0resets/switches/training;
requested30s,actual1.1500s,valid1.1495s,net+47.308647deg,peak60.645683,
backward14.923018,stopobjectbelowBoyamanipulationregion. Maxvaliddrift90.503mm,
tilt129.050deg,normal40.849N. Last.2snet-13.055deg isposthoc,notfilteredscore.
Observed0-1.1495sonly:fixed-original-axis reported/pose means1.266862/1.263919rad/s,
fullvectorRMSE.418496/contactRMSE.438184,visiblehandcontact91.214%; independent
SciPyposeintegral47.308632deg. OriginalHora rotation/totalrewardmeans+.430300/
+.320276percontrol. First5swindowpartial;5-20/20-30sunobserved,null.
Meanaxialagreementdoesnotestablishfullvelocityconsistencyorstableholding;
shortturnthenearlydropremains,nostable/sustainedrotationclaim. Lastupdate223
completedtrainingepisodes allbelowregion,meanvalid1.749949s;notstandaloneevidence.
Old20M/50Marenotmatchedbudgetcontrols. Finalsmallresults/stage/training-summary/
solveridentitiesarchived,weights/TensorBoard/rawtraceslocal,oldhistorypreserved.
Noadditionaltests/GPUepisodes/video/training;nofollow-onbudgetstarted.


[Original4 small-budget matched control LAUNCH HISTORY; completed below](experiments/2026-10-10-boya-velocity-four-1m/README.md):
Latestuser explicitly approves original4 fresh1M control and requests prompt execution.
Started2026-10-10 13:20:32Shanghai;supervisor2171808/training2171842. Output
outputs/boya_hora1024_velocity4_small1m_v1;freshnetwork/normalizers/Adam/RNGseed43,
1024envs,requested1M/rounded1,007,616actions/123updates. Noresumefrom0oroldweights.
Reuse unchanged common25training/environment/controller/reward/evaluation sources,
allSHA256match completed0run;cachehashmatch. Only protocol/experiment/output identity
and declaredvelocityiterationfieldsdiffer. original_tgs16_4 scenevelocity4/255,
articulation16/4,rigidbody16/1,position16/255;effective4 bydocumentedaggregation,
notnativekernelcounter. Sameobject/grasp44/28cache/actions/privilegedobservations/
HoraPPO/reward/PD/collision/dt/workspace/numericalrules/evaluationbudget.
Summarytraining/TensorBoard/first-every16-finalpairedcheckpoints;noresourcewatchdog,
memoryquota/trainingwallcutoff/video/auxiliaryGPUtrial/newtests. AutomaticallyONE
frozenfinaloriginalstate30s-or-firstfailure/1500swalleval,exactinitialreference,
fullphysics/rewardrecording,existingofflineanalysis,thenSTOP;noretry/extension.
Comparevaliddurationandrealnet/peak/backward/stopreasonwithcompleted0fresh1M,
notold20M/50M. Labelsignalwindowsbyactualobservationduration. Single-seedcontrol
informsconfigurationchoice,notstatisticalsuperiorityorSOTA/hardwarevalidation.
Ifbothfailearly,reportnoidentifiedworkingconfigurationandaskbeforechangedroute.
Actualsnapshot10updates/81,920actions;first8192-actionpairedcheckpointgenerated,
actualPPO/network/taskconfigexactlymatches0run. Offlineanalysisinterpretationtext
madeconfiguration-neutral;analysiscalculationsandtrain/evalcodeunchanged.
Protocol/sourceidentity/launchevidencearchived;finalcomparisonpending.


[Original4 fresh1M control COMPLETED; both configurations stopped](experiments/2026-10-10-boya-velocity-four-1m/README.md):
Original4 completed1,007,616actions/123updates normally;train/eval/analysisexit0,
trainingwall1440.27s,total1472.25s. FinalcheckpointSHA256
5db6e8c9140835ff3d6c7ba237072eaccccf818e916594ead97518dc751e213c.
2026-10-10 14:00Shanghai read:supervisor/lastchildbothnotalive,nofollow-ontraining.
ONE frozenownmodel/normalizer/originalgrasp44 episode;initialq/qdot/object/commands
exactlymatchcommonreference,0resets/switches/learning. Requested30s,actual1.5000s,
valid1.4995s,net+23.791725deg,peak66.298258,backward42.506534;
stopobjectbelowBoyamanipulationregion. Maxvaliddrift76.293mm/tilt108.280deg/
normal61.376N,last.2snet-38.212618deg(posthoconly,primaryscoreunchanged).
Matchedfresh0reference:same1,007,616actions/seed43/task/reward/PPO/config/corecode/
cache/evalbudget,valid1.1495s,net+47.308647deg,peak60.645683,backward14.923018,
samebelowregionstop. Original4retains.35slongerbutlessnet/morebackward;
0doesnotsimultaneouslyimproveretentionandrotation. Bothdropwithin1.5s;
noverifiedworkingstableconfigurationidentifiedat1M,nolong-termconvergenceclaim.
4observed0-1.4995sonly:reported/posefixedaxismeans.770223/.666301rad/s,
fullvectorRMSE.199096/contactRMSE.201243,visiblehandcontact97.866%;SciPypose
integral23.791692deg,originalrotation/totalreward+.427745/+.310150percontrol.
First5spartial;5-20/20-30sunobserved. Unequalvalidprefixsignalstatisticsarenot
matched-timeRMSEcomparison. Lastupdate234resettrainingepisodesmeanvalid1.838175s,
230below/4lateral;notstandalonemotion. Single-seed/oneepisodeeach,nosuperiority/
SOTA/hardwareclaim. Smallfinalevidence/two-runsummaryarchived;weights/raw/TBlocal.
Noadditionalphysics/tests/video/learning;noautomaticextensionorroutechange.


[Pose-derived rotation reward IMPLEMENTED; OFFLINE verification completed](experiments/2026-10-10-boya-pose-reward/README.md):
User explicitly approves replacing rotation reward velocity input with actual pose
changes and requests recording/push. New training/background defaults hora_pose_delta;
original hora_reported_velocity retained explicitly, low-level API legacy default
retained, legacy evaluation records interpreted as reported velocity. Shared train/eval
accumulator: shortest WORLD XYZW rotation vectors each.0005s, sum100/.05s, float64
arithmetic then native velocity dtype. Fixed original target axis, Hora formula,
clip[-.5,.5]/scale1 and all other penalty terms retained. PPO/architecture/obs/actions/
engine/physical/numericalrules/primarymetrics unchanged; reward objective/advantages/
value targets and resulting behavior change. Vendored Hora source remains untouched.
Rewardconfiguration and helperhash in results/pairedresumecontract; mismatches rejected,
legacy weights never silently relabeled pose-trained; strict sourceguard retained.
Old20M->newreward requires explicit recorded branch migration, not performed here.
Existing frozen20M/original4/30s trace only,600controls offline rescored;0newphysics/
learning/policyepisodes.20-30s originalrotationmean+.367686 ->pose-.004294, actual
fixedtargetaxismean-.004294rad/s; total+.233122 ->-.138857, othertermsretained.
Torch accumulator/SciPy controlvelocitymaxerror2.84e-8rad/s, originalreward reproduction
maxerror5.97e-8. Three rotationCPUcases plus existingpairednextupdaterestore passed;
rewardidentitymismatch rejected.9editedPythonfilessyntaxpassed. Stationary1.90e-19rad/s
roundoff uses1e-12testtolerance; no rewarddeadzone ornewphysicalthreshold introduced.
Primarymovingcylinderaxis remains distinct fromfixedrewardaxis; clippingstillmeans
rewardmeanisnotnetprogress. No learnedrotation/stabilityimprovementclaim, native
runtime/throughputnotvalidatedhere. Smallprotocol/codeidentity/offlineevidence saved;
alloldweights/rawtraces/scoresretained. Newtrainingnotstarted,nogpu/video/broadtests.


[Pose-reward20M branch continuation LAUNCH HISTORY; completed below](experiments/2026-10-10-boya-pose-reward-continue1m/README.md):
User approves20Mteacher/newreward/additional1M andonefinal30soriginalstateeval.
ExplicitCPU migration fromparentSHA2569781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e,
20,004,864actions/2442updates. Allserializedlearnerfields exactaftercopy:policy/critic,
obs/value normalizers,Adam/LR/scheduler,RNG,counters/times/best_rewards preserved;
originalfileunchanged. Reviewedtwoadapter-sourceupdates/twonewprofilehelpers,
samePPO/network/cache/termination/unaffectedsourcehashes verified. Legacycontract
gets explicitoriginal4engine/newpose-rewardidentity,newlineage andparentprovenance;
ordinarystrictresumeequality remainsenabled. Nooldaction countedaspose-trained.
InheritedLR5.7805099719442054e-5;oldcritic/Adamdistributionmustadapt,noconvergencepromise.
Migration0physics/actions;simulationresetfromsame28cache,notcontinuousparentepisode.

Detachedstart2026-10-10 14:46:08Shanghai;supervisor2311583/training2311585,
outputs/boya_hora1024_pose_reward_continue1m_v1.14:47:55snapshot2450updates/
20,070,400cumulativeactions,65,536newposeactions/8updates. Firstnew2443checkpoint
8192newactionsverified:initialmodel/bothnormalizersexactparentmatch,strictcontract
equalmigrationtarget,Adamstepsold+80perparameter,modelchangedafteractualPPOupdate,
branchprovenancepersisted. ThreeeditedPythonfilessyntaxpassed;noauxiliaryGPUtrial/
newtestcampaign. IndependentTensorBoardeventsandpairedcheckpointswriting.

Requested+1M,rounded+1,007,616/123updates;final21,012,480/2565cumulativetarget.
Onlyrewardinputchanges;original4engine,grasp44/40x32mm/50g,16fingeractions/heldwrists/
4passivemimics,v3PD/collision/dt,privileged96+9/8latent,1024envs/PPO/workspace/numerical
rulesunchanged. Summarytraining/headless/first-every16-finalpairedcheckpoints,
noresourcewatchdog/memoryquota/trainingwallcutoff. BudgetthenONEfrozenownnormalizer/
originalgrasp44/30s-or-firstfailure/1500swalleval,exactinitialreference/fullphysics/
poserewarddiagnostics/offlineanalysis,thenSTOP;error/manualstopnoretry/extension.
Reuse20Moriginal4same30sreference:valid30s/net+30.426747/peak35.822706/backward37.240682,
normal30sstop. Compareactualretention/net/backwardandobservedtail,noreturn-only
improvementorreset/contact-filteredsums. Singleancestrybranch feasibility,notmatched-
total-budgetstatisticalsuperiority/SOTA/indefiniterotation/hardware. Noresultyet.
Migrationplan/result/protocol/command/sourcehashes/actuallaunchsnapshotarchived;
alloldweights/historyandnewweights/raw/TensorBoardremainlocal. Milestonepushauthorized.


[Pose-reward20M+1M continuation COMPLETED and STOPPED; holding regressed](experiments/2026-10-10-boya-pose-reward-continue1m/README.md):
New1,007,616actions/123updates completed normally, final21,012,480actions/2565updates;
trainingwall1379.23s,total1423.17s,train/eval/analysisexit0,stagecompleted.
2026-10-10 15:23:48Shanghai read:supervisor/lastchildbothnotalive,nofollow-ontraining.
FinalSHA256472f8e679ed4037096a083c67b16cf40b896e3917437c0d00a6cae6848ade2e2.
ExactlyONEfrozenfinalmodel/ownnormalizer/originalgrasp44evaluation,initialq/qdot/object/
commandsmaxerrorsall0,0resets/switches/training. Requested30s,actual2.0000s,
valid1.9995s/net+25.428426deg/peak34.139185/backward10.944128,
stopobjectbelowBoyamanipulationregion. Reused20Mreferencevalid30s/net+30.426747/
peak35.822706/backward37.240682,normal30sstop. Holdingclearlyregressedandvalid-prefix
netdidnotincrease;smallerbackwardover2sisnot30sbackwardimprovement. Newrewardinput
verificationdoesnotestablishstablelearning;thisboundedcontinuationfaileditsgoal.
Maxvaliddrift71.974mm/tilt38.234deg/normal59.735N,SciPyangle+25.428332deg;
5-20/20-30sunobserved,null. Last.2snet-6.760degposthoconly,nofilteredscore.
Observedprefixfixedaxisactualmean+.233365rad/s,selectedposerewardrotation+.280245;
axis/clipping/windowdifferencesremain,norewardmean-as-netprogressclaim.
Lastupdate113trainingresetsallbelowregion,meanvalid3.7203s,notstandaloneevidence.
Singleancestrybranch/oneevaluationcannotisolateforgetting/critic/Adam/rewardweight
causation,notindependentmatchedbudget/statistical/SOTA/hardwareproof.
Smallfinalcomparison/eval/signal/training-summary/config/stageevidencearchived;
weights/raw/fullupdates/TensorBoardlocal,oldhistorypreserved. Nofurtherphysics,
tests/video/audits/retries/extensionsorconfigurationchanges. Awaitdiscussion.


[Next-step terminal-cost PROPOSAL ONLY; not implemented/launched](experiments/2026-10-10-boya-drop-penalty-proposal/PLAN.md):
User asks how to decide the next plan. Existing reward arrays only,0newphysics/
training/actions/episodes. gamma.99 rawdiscounted40control/2s dropreturn+6.696855;
original20M first400control/20s counterfactual pose-rewardreturn-3.763435.
Include actualterminalreward-.565460; terminaldiscountgamma^39=.675729.
Cost15.480006 ties these two finite paths; propose integer16, giving dropreturn
-4.114809. Parenttruncation hasnotimeoutcriticbootstrap;thisisNOTproof ofoptimal
dropbehavior ortruePPOvalue,onlyonecandidatecalibration,nopenaltyscan.
Recommend original20M/newseparatebranch/fulllearnerpreserved,pose-reward plus
ONEcost16 at existingbelow/lateral workspace terminal codes9/10,noordinarytimeout
penalty;allotherterms/PPO/task/engine/interfacesunchanged. Requested200k/newrounded
204,800actions/25updates,final20,209,664/2467cumulative,thenONEsame30soriginalstate
evaluation andSTOP. Preliminarygoal30sretentionANDnet>parent30.426747deg,
reportpeak/backward/actualstop;notmatchedtrainingbudgetcausal/statisticalproof.
Estimatedtraining5min+eval9minifcomplete. Proposal/evidence written,NOrewardcode
edited/NOnewtraininglaunched. Requiresuserdecisionbeforenewrewardstage.
