# Resume workspace teacher to50M with10M evaluation milestones

User explicitly authorizes implementing evaluation every additional10M actions and
continuing the CURRENT workspace teacher to50M total. Read experiment_state and
workspace10M protocol/results:10,002,432actions,finalfrozen+182.935623deg/5.5995s,
stopbelowmanipulationregion,77.904degmaxtilt. Earlierstrictweights were deleted
byuser; they are not part ofthiscontinuation. New factor: larger cumulativebudget
with declared learner-preserving restarts and repeated fixed-policy evaluation.

## Controller and environment

Reuse scripts/train_boya_hora.py,run_boya_hora_background.py,evaluate_boya_hora.py
UNCHANGED. Start from outputs/boya_hora1024_workspace10m_v3/training/teacher_final.pth,
SHA25690419cc9cc845ab09b319b73533777e983f6ecfd85998aa04955ba96d9ee9ebb.
Restore model,observation/value normalizers,Adam,adaptiveLR/scheduler,RNG,stepcounts
and originallineage. InitialrestoredLR5.7805099719442054e-5,not reset to.005.
At each resume, discard unfinishedsim episodes and reset from SAME28statecache;
this is continuing the learner,not uninterrupted physics or bitwiseepisode replay.
Evaluation never modifies the saved learner state used for the next segment.

Same40x32mm/50g cylinder,grasp44/fullgravity,16fingeractions/heldwrists,
4passivecouplings,Isaac6.1/Lab3rc1,v3clippedPD,uncalibratedA=4*D_effective*dt,
28excludes/CADconvexselfcollision,TGS16/4,dt.0005/control20Hz/.35rad/s.
SameHoraPPO/reward/96proprio+9privileged/8latent/noDR/no student,no tactileactor.
1024headless/camerasoff/8stepsperupdate/minibatch512/5epochs/seed43/gamma.99/tau.95.
Sameboya_workspace: centerheightlower.066801253194m;XYlower
[-.459866091313,-.069149973487],upper[-.259054954510,.076535408821]m;
2consecutive20Hzoutside samples triggerfailure. Fewerthan2contactfingers is
NOT a termination rule.400control/20strainingtimeouts; numericalfinite/speed100/
mimic.05 checks perphysicsstep. Oldstrictlimits remain diagnostics only.
Taskregion remains a geometryapproximation,not validatedcontrollability/droptruth.

## Schedule and automation

New detached scripts/run_boya_hora_milestones.py sequentially invokes existing
training+evaluation supervisor; GPUtraining andevaluation do not overlap.
- Existing10,002,432evaluation is reused as baseline,not rerun.
- Cumulative20M ->20,004,864actualactions ->evaluate ->resume thischeckpoint.
- Cumulative30M ->30,007,296actualactions ->evaluate ->resume thischeckpoint.
- Cumulative40M ->40,001,536actualactions ->evaluate ->resume thischeckpoint.
- Cumulative50M ->50,003,968actualactions ->evaluate ->STOP.

8192-action PPO batches explain rounding. Additional40,001,536actions total.
ExactlyFOUR new frozen evaluations, one per milestone; not repeated old/new pairs.
Everyevaluation originalgrasp44,ownnormalizer,seed43,0resets/0switches/0training,
120ssimulationwindow or firstdeclaredfailure/1500swallbudget,30ssupplementary.
Training20sepisodetimeouts do not cut offevaluation. Signedvalidprefixmoving-axis
netangle/peak/backward/time,stopreason,and drift/tilt/forcemaxima retained.
Initialq/qdot/object/commands compared with independently initialized10Mbaseline.
Oneepisode percheckpoint,not statisticalvalidation/SOTA/convergence/sim2realproof.
No automatic choice ofbestcheckpoint,change ofobjective/criteria or budgetextension.

Wrapper verifies completebudget/evaluation before advancing, precedingcheckpoint
hash+lineage,source/asset+task contracts. Amanualstop,source mismatch orprocesserror
stops chain with recordedreason,without retry or alternatecontroller. Ordinary
physical/task earlyfailure is recorded as that milestone's evaluation outcome;
no unapproved improvementthreshold is imposed on the authorized50M budget.

Output outputs/boya_hora1024_workspace50m_v1; each cumulative target has an
independent actions_XXXXXXXXX/training andevaluation directory. Summarytraininglogs,
fullpairedcheckpoint first/every16/final, fulltracesonlyevaluation. Keep10Mweights.
JSON/CSV/Markdown milestone table and TensorBoard evaluationcurves updated after
eachevaluation; trainingcurves use cumulativeaction axes in separatesegmentruns.
No image/video rendering inthisautomation. nice10/4threads,process-onlyEULA,
noresourcewatchdog/cgroup/memorycutoff/trainingwalllimit. Nohardwarecommands.

## Necessary validation and launch

Three CPU schedule/stop regressions; reuse alreadyverified learnerrestore path.
Confirm actualrestored counts andfirstnewupdate/checkpoint duringauthorizedrun,
not another shortGPUtrial. No additional hyperparameterscan or broadtestcampaign.
