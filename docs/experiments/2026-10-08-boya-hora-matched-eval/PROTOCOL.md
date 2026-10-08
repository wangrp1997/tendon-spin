# Matched old/new teacher evaluation

User authorizes preparing the first proposed item: old/new checkpoints evaluated
under common conditions after the new background training. Read experiment_state,
oldfresh10M final (+63.626313deg/2.148s under15deg cutoff),newheight10M protocol and
source review. New factor: evaluate the OLD checkpoint under the NEW height rule,
then compare with the new checkpoint's already queued final evaluation. Preserve
original old-rule results; never relabel them as a new common-rule execution.

## Fixed conditions

Both checkpoints must contain10002432trainingactions,finalpaired checkpoints from
oldboya_hora1024_fresh10m_v1 andnewboya_hora1024_height10m_v2. Each actor retains its
OWN observation normalizer. No gradients,adaptation-weight updates,model switching,
cache-state start,resets or rollout continuation. Seed43 and deterministic actor
inference; the existing joint-observation noise remains identically seeded.

Same40×32mm/50g cylinder,originalgrasp44/handorientation/fullgravity,16activefinger
commands/heldwrists/4passivecouplings. SameIsaac6.1/Lab3rc1,v3clippedexternalPD,
uncalibratedarmature,CADconvexselfcollision/28excludes,TGS16/4,dt.0005,control20Hz,
.35rad/s targetrate,limits. Sameprivileged96+9teacherinputs;noDR/student/tactileclaim.
Physics/action/model/normalizer/scoring source hashes must match across training
lineages. Archived_frame/observe/spin_increment ASTs must match. The declared
termination/logging changes are retained in lineage metadata.

Both use hora_height: one fixed reset plane.10106109973956033m above each origin,
checked at control boundaries; perphysics-step nonfinite/speed100/mimic.05 checks.
No5mm3Ddrift/15degtilt/12Nforce taskstop; oldstrict shadow remains diagnostic.
Independent evaluation budget120simulatedseconds or first commonrule violation;
1500swallbudget including startup. No training20sreset duringevaluation.
Headless/camerasoff,4threads,nice10,noresourcewatchdog/cgroup/GPUmemorycutoff.

Compare actual initialjointpositions/velocities,objectstate and motorcommands from
separateinitial_state.npz withatol1e-6/rtol0. These are independently initialized
runs,not state transfers. Failedidentity/initialstate verification blocks reporting.

## Execution and provenance

A detached CPU-only waiter polls the existingnewrun'sstage every30s. It launches
noGPUscene while training or the already queuednewpolicy evaluation is active.
Only after normaltrainingbudget completion AND successful completion of that
existing evaluation: validate frozenfinal artifacts,then run ONE additional old
checkpoint evaluation. On dependencyerror/manualstop/source mismatch,save error
and do not launch or retry. Exclusivequeueclaim prevents duplicate launch.

Reuse scripts/evaluate_boya_hora.py unchanged. Oldinference uses a separately
named evaluation_execution_contract_not_training_record,containing old training
record/path/hash/checkpoint/sources and the new execution sources/rule. The
originaltrainingrecord and checkpoints are untouched. The existing evaluator
still verifies every source hash before execution. Its controller name explicitly
identifies matched_height_old_weights. This is a declared new inference experiment,
not a claim that oldweights were trained with the new rule.

Newpolicy result is reused only if completed with exactly matched checkpoint,
trainingcounts,common source/spec/window/seed,0resets/0switches. No redundant new
policy rerun. Fullphysics/control archives retained locally;weight files notpushed.

## Scores and interpretation

Main score:signed moving-cylinder-axis netrotation of the validprefix within120s.
Also30s window from the same uninterruptedtrajectory,validduration,peak,backward,
final30sadvance when observable,and drift/tilt/normalforce maxima WITHIN each
window. Eachwindow reports its own completion/stop reason and separately retains
the actual sourceepisode stop reason. Earlyfailures included;no pooledangles or
fixedturncount passgate. Reward'sfixedinitialaxis andmoving-axis metric remain
explicitly distinct as in the source port.

Outputs comparison.json/csv/md plus plan/status/checkpointidentities/execution
contract and rawold evaluation. Before execution,tablecells staypending. Historical
old15deg-score is excluded from commonranking. A change in taskdefinition alone
cannot be called learning improvement. This is onefixedinitialstate perpolicy,
not a multiple-seed benchmark,SOTA result,hardwarevalidation or indefiniterotation.

## Validation

13 focusedCPUunitchecks passed (comparisoncontracts,checkpointprovenance/count,
initialstatemismatch,dependencygating,migrationlabels,windowcensoring/diagnostics
and rotation scoring),compile anddiffcheck passed. Preparing the real manifest
verifies existing artifacts/source identity without loading Isaac or executing
physics. Active training/evaluator/protocol source files are not edited.
