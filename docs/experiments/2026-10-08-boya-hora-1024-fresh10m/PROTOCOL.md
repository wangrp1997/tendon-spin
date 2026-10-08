# Fresh1024 nominal teacher training after user-requested resume verification

Read experiment index,previewresults and existing10Mcontinuation protocol.
User correction: test resume first,then start formal background training FROM ZERO.
The already-running continuation is stopped and retained as resume_test;its samples,
weights,optimizer/normalizers are NOT inputs to formal training. No further approval
needed after successful resume verification. Preview video review already passed.

Fresh seed43,network,obs/value normalizers,Adam/RNG;zero completedupdates/actions.
No --resume argument. Same28accepted-state grasp cache is reset data,not learned
weights.1024×8horizon=8192agentactions/update;10000000requested roundedup to
10002432actions/1221completeupdates. No counts pooled with any previous run.

Same nominal task/physics/PPO as preview:40mm diameter×32mm/50g,grasp44referencecenter/
orientation/full−9.81gravity,16fingerinputs/heldwrists/4passivemimics,v3uncalibrated
armature/clippedexternalPD,28collisionexcludes/CADconvexselfcollisions,TGS16/4,
dt.0005s/control20Hz/.35rad/s. PinnedoriginalHoraPPO/reward/96+9privilegedobs/noise±.02,
minibatch512/5epochs/initialLR.005/clip.2/KL.02/gamma.99/tau.95. No DR/student/full
baseline or real deployment claim. Finite/5mm/15deg/12N/100rad/s/.05mimic physics
checks every step. No4-finger gate. Preview floor is rendering-only,absent in dynamics.

Headless,camerasoff,nice+10,4Torch/OMP/MKLthreads,existingIsaacenvironment.
Noresourcewatchdog,cgroupquota,GPUmemorystoportrainingwalltimeout. Stop atactionbudget,
processerror or user signal;no auto-restart or method change. SIGTERM/INT saves at
PPO-updateboundary. Detached process survives closing chat. Pairmodel,normalizers,
Adam,LR/scheduler,RNG/counters in atomic checkpoints first/every16update/final;
latest_checkpoint.json identifies latest safe resume state for future continuation.

Summary-only training logs and TensorBoard:losses,totalrecentepisodereward,rotation
reward/net/peak/backward/validduration mean/median/p90/max,terminationfractions,LR/KL/
entropy/speed. Per-update flush. No full training trajectories;per-stepphysical checks
still run;completedepisode rows cleared after aggregation. TB binds127.0.0.1:6006:
formal_fresh separately from preview and resume_test. Curves measure exploratory
trainingepisodes,not standalone evaluation or indefinite rotation.

Resume verification uses existing continuing run:initial model/obs/value norms must
exactly match source checkpoint;first resumed optimizerstepcount must advance from
source Adam state,updatedweightsmustchange,cumulativecounts/lineagemustcontinue,
alllossesfinite. Prior CPU originalPPO test already established exact next-update
match after declared reset;this checks actual Isaac/CUDAintegration,without extra
simulatorlaunch. Actualstop and checkpoint evidence retained separately.

After successful formal targetcompletion,one separate frozen final-policy original-
state episode,120swindow/firstphysicalgate/1500swall,zeroresets/switches,seed43,
fullphysics/action/contactlogs. No evaluation onmanualstop/trainingerror. Queued
assessment is not an executed result. No success promised bytrainingbudget completion.
Local milestone commit only;remote pushes remain paused.
