# User-approved1024 background continuation to10M cumulative actions

Read experiment_state,preview protocol/results/RESUME and rendered-video correction.
User reviewed the corrected video and explicitly authorizes formal background training
and TensorBoard. This supersedes the prior wait-for-video-review state. No environment
scan/restart from scratch;no resource watchdog/cgroup quotas/automatic GPU-memory stop.

Resume outputs/boya_hora1024_preview_v1/training/teacher_final.pth:
SHA256677a1fff1459549a9da3b77fd50ba1e00488a52d215d356c589e1d9e1d028412,
65536agentactions/8updates. Same lineage;restore model,obs/value normalizers,Adam,
LR/scheduler,RNG,counters. New simulation episodes reset from the same28-state cache;
this is learner continuation,not exact intermediate physics replay.

1024envs×8horizon=8192actions/update. Target10000000cumulative rounds to10002432/
1221completeupdates;this process adds9936896actions/1213updates. First resumed and
every16completedupdate checkpoints;atomic latest index;final weights saved on normal
budget or requested signal stop. No sum with independent historical pilot/throughput
runs. Seed43 metadata,existingIsaac environment,no installation/global settings.

Physics/task/controller are the same declared nominal teacher port:40mm diameter×32mm
cylinder50g,originalgrasp44center/orientation,worldgravity−9.81,16fingerinputs,heldwrists,
4passivemimics,v3 uncalibratedarmature/clippedexternalPD,28collisionexclusions,CADconvex
selfcollision,TGS16/4,dt.0005s/control20Hz,.35rad/s. Privileged96+9input/noise±.02rad,
originalpinnedHoraPPO/reward,5epochs/minibatch512/initialconfiguredlr.005(actualrestored),
KL.02/clip.2/gamma.99/tau.95. NoDR/student/completebaseline/hardwareclaim.
Everyphysicalstep retainsfinite/5mm/15deg/12N/100rad/s/.05radmimic criteria;no4fingercondition.
No reference floor in physics: the user's floor was visual replay decoration only.

Training is headless/camerasoff,nice+10,4Torch/OMP/MKLthreads. Resource and training
wall cutoffs are disabled (--wall-s0,--max-gpu-memory-mib0);stop normally at action
budget. Detached stage process logs status/PIDs and waits for child exit;it does not
monitor or kill on resource utilization. SIGTERM/INT requests graceful training stop
at PPO-update boundary and skips queued evaluation. No automatic restart or method
changes on failure. User can close chat while the background process continues.

Summary-only training logs:per-update losses,reward,valid-prefixnet/peak/backward/
duration(mean,median,p90,max),terminationfractions,LR/KL/entropy/speed,source/cache/config
and paired checkpoints. No per-physics-step training archive;temporary episode rows
cleared eachupdate. TensorBoard flusheseachupdate;host127.0.0.1:6006,preview/formalruns
separate with cumulativeagentstepaxis. Prior preview has loss/rotationcomponent/mean
rotation/duration;total episodereward and stopfractions are new scalar exports from
existing computations,not PPO/reward changes. Trainingrun aggregates are exploratory
statistics,not frozen original-state performance or indefinite-rotation evidence.

After successful target-budget completion,one separate frozen final-checkpoint
original-state evaluation:1env,120s simulationwindow,firstphysicalgate or1500swall,
0resets/controllerswitches,deterministicmean action,noise seed43,fullper-stepevidence.
No evaluation on trainingerror/manualstop. The stage may outlive this conversation;
queued evaluation is not executed evidence. Report eventual actualnet/peak/backward,
validduration and stopping reason. Save stage exit/result rather than claim trained
policy success from losses or reaching10M. Local milestone commit only;pushpaused.
