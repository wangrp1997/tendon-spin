# Hora height-termination correction: fresh1024 teacher,10M

User explicitly authorizes preserving old weights, correcting/reviewing termination
against source, and starting another fresh background run. Read experiment_state,
fresh10M final result (+63.626313deg/2.148s/tilt>15deg), source review and Hora port
checklist. New factor: replace inherited strict task-done gates with translated
Hora height/time semantics. No pooling with previous policy; no old weights loaded.

## Declared task and implementation

Same40mm diameter×32mm cylinder/50g,grasp44original pose/full gravity−9.81,
16activefingerinputs,heldwrists,4passive1:1couplings,Isaac6.1/Lab3rc1,v3external
clippedPD,uncalibrated A=4*D_effective*dt armature,28collisionexcludes/CADconvex
selfcollision,TGS16/4,physicsdt.0005/control20Hz/.35rad/s commands. No external
support/floor dynamics. Existing28-state cache is reused unchanged; its original
strict generation rules and limited diversity remain documented.

Profile hora_height reuses the exact extracted Hora v0.0.1 check_termination
method. Source Allegro nominal scene z=.650,reset_height_threshold=.645. Translate
one fixed plane by Boya nominal z=.10606109973956034: reset plane
.10106109973956034m above each environment origin. This mapping is an explicit
cross-hand coordinate assumption,not independently measured real drop clearance.
It is fixed across cache resets; does not follow a drifting object. All28cached
initial center heights(.104861669… .108861171m) are above this plane.

Task-done: object center below that plane at the end of a control step,or400
controlsteps=20s. Height/time source inequalities retained. No3Ddrift5mm,
tilt15deg ornormal12N task-done; these remain diagnostics and labeled legacy
prefix scores. Boya numerical validity remains per-physics-step:nonfinite,
>100rad/s orcouplingerror>.05rad. Nonfinite aborts the run; other numerical gate
terminations remain explicitly labeled,not paper task success/failure evidence.
Existing motor effort and joint/target limits remain the declared Boya interface.

Same original Hora PPO/ActorCritic/propertyencoder/reward:96proprio+9privileged,
3actorframes/30history,noise±.02,8latent,originalcoefficients. NoDR/student/full
Hora reproduction claim. Differences reviewed in REVIEW.md,including actionstep,
initialtargetbias,engine/dynamics,smallcache,physicssubsteps/axis and nominalattrs.

## Execution budget and persistence

Fresh seed43 network/normalizers/Adam/RNG,zero counters,no--resume.1024environments,
8steps/update,minibatch512,5epochs,initialLR.005,KL.02,gamma.99,tau.95.
10000000requested rounded to10002432actions/1221updates. Headless/camerasoff,
nice10,4Torch/OMP/MKLthreads. Existing environment and process-onlyEULA acceptance.
Noresourcewatchdog/cgroup/GPU-memory cutoffs; no global install or setting changes.

Output outputs/boya_hora1024_height10m_v2,controller
hora_boya1024_height_teacher_fresh10m_v2. NewTensorBoardlabelhora_height_fresh;
oldformal_fresh/preview/resume_test retained. Summary-onlytraininglog withPPO,
reward/episode/netangle/duration and termination counters; diagnostic episode maxima
for displacement/tilt/force and oldstrict validprefix/net angle. Atomic paired
learnercheckpoint first/every16updates/final,including model/norms/optimizer/LR/RNG.
Termination profile+translated height+source identity enter resume contract.
No auto-restart/continuation after10M; signal/error records actual reason.

## Queued fixed-policy evaluation

After normal actionbudget completion only:one uninterrupted originalgrasp44 final
frozenpolicy evaluation,120s requested/first new-profile failure/1500s wallbudget,
0resets/0switches. Same TaskTermination module and exact recorded spec must match
training; no cache start. Full perphysicsstep action/state/force evidence retained.
Report actual termination,30/120s signed moving-cylinder-axis netangle,peak,
backwardmotion,validduration and final30sprogress if present. Reward uses fixed
initial-axis projection as in existingBoya adaptation; axes are declared,not
interchanged. No fixedturncount passgate.

Also score oldstrict prefix offline from that same uninterrupted execution and
label it legacy_strict_shadow. It is not another rollout or continuation score.
Main new-profile numbers are not directly ranked against old strict main scores.
Longer survival/angle after changing definition alone does not establish learning
improvement. No extra evaluation on manualstop/error,or safety/hardware/SOTAclaim.

## Review before launch

Targeted CPU checks passed:10 unittest cases across source termination/translation,
legacy/capturedtilt regression,numerical separation,rotation scoring and original
PPO resume equivalence. Pythoncompile andgitdiffcheck passed. No auxiliary GPU
experiment required; observe actual launch/first completed update and checkpoint.
