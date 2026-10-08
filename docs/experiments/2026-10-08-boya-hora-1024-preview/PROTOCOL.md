# 1024-env short training and native Isaac policy preview before long training

User authorized summary logs and a first10M-action cumulative training budget with
resume,explicitly declined the new resource watchdog/cgroup/automatic memory-stop
mechanisms. Latest steering requires a short visual preview BEFORE the long stage;
long training remains pending user review. Read experiment index,128nominal protocol,
1024speed results and2048guarded outcome. No further environment-count sweep.

This preview:1024environments,8PPOupdates×horizon8=65536actions,fresh seed43 learner.
Same28-state settled cache,original40mm diameter×32mm cylinder/50g,grasp44 reference
orientation/center,fullworldgravity−9.81,16finger inputs/heldwrists/4passivemimics.
IsaacSim6.1/Lab3,TGS16/4,dt.0005s/control20Hz,.35rad/s,externalclippedPD and declared
uncalibratedv3armature,28exclusions/CADconvexselfcollisions. Privileged nominal teacher,
96proprio+9privileged input,±.02radnoise,sameupstreamHoraPPO/reward/Adam/normalizers,
minibatch512/5epochs/initiallr.005/KL.02. NoDR/student/fullbaseline/hardwareclaim.
Per-physics-step finite,5mm,15deg,12N,100rad/s,.05radmimic checks unchanged. No4fingergate.

New logging mode is explicitly authorized:training summary statistics and weights,
no per-physics-step array allocation/CPU transfer/disk dump;bulk transfer completed
training-episode summaries each control step,aggregate and clear at each update.
Keep mean/median/p90/max valid net,peak,backward,duration plusreasoncounts,losses,
rewards,TensorBoard/config/source/cache identity. This supersedes the older full-
training-trace rule ONLY for this declared training route. No retrospectively inferred
single-episode success from these aggregates. Evaluation still saves all steps.

Atomic checkpoints pair policy,observation AND value normalization,Adam state,adaptive
learning rate/scheduler,completedupdates/actions,RNG,task/sourcecontract and lineage.
On resume reset physics from the same cache and discard unfinished episodes;learner
continues,not bitwise physics continuation. Resume target is cumulative actions rounded
up to full8192-actionrollouts. A future10Mtarget rounds to10002432,including this
65536-actionpreview only if actually resumed from its checkpoint;never pool independent
old128/512/1024 runs. Checkpoints first/every4update/final forpreview;every16forlongrun.
CPU-only original-PPO test confirms next update matches after the same declared reset
(weights,both normalizers,Adam,LR,counters),and rejects mismatched task contracts.
This is a learner-state correctness check,not a robot performance experiment.

Training has no wall timeout,GPU polling/automatic cutoffs,watchdog or resource quotas.
Use existing environment,headless,no camera for training,nice+10,4Torch/OMP/MKLthreads.
No full10Mbackground run until user reviews preview. No installs/globalsettings.

After the8updates,one separate frozen final-checkpoint evaluation:original nominal
state,zeroresets/switches,deterministic mean action,noise seed43,unchanged physical
criteria,up to5s simulation/1500swall. NativeIsaacRTX camera/light added by optional
scene callback before initialreset;camera changes rendering only. All actual actions,
poses/forces/gates retained. Stop on first physical gate;no manual continuation.
Video labels actual simulation time,valid net angle,drift,training count and termination;
0.2x slow motion by repeating real camera frames at50ms physics spacing,initial1s
andterminal2s labeled stills. No motion interpolation or invented execution duration.
No claim early checkpoint is converged or continuously rotating. Frozen evaluation
is the review artifact;record source hashes before training and do not edit them
between training and evaluation. Commit small evidence locally;remote push paused.
