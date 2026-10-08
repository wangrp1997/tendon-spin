# First-vs-third final teacher comparison under Boya workspace rules

2026-10-09 user explicitly requests enabling automatic evaluation of BOTH teachers
after the current training and asks about last checkpoint selection. Compare each
run's FINAL checkpoint after10002432actions; no best-reward or mid-run selection.
Final need not be best: this fixes equal training budgets. Preserve own normalizers.
Read experiment index, original matchedheight protocol, workspace training protocol.
This is a NEW comparison; the stopped v2 queue and its deletedweights stay historical.

## Common execution

Old: outputs/boya_hora1024_fresh10m_v1/training/teacher_final.pth.
New: outputs/boya_hora1024_workspace10m_v3/training/teacher_final.pth, afterbudget.
Original40x32mm/50g cylinder,grasp44/fullgravity,16fingeractions/heldwrists,
4passivecouplings,Isaac6.1/Lab3rc1,v3clippedPD,A=4*D_effective*dt,28excludes/
CADconvexselfcollision,TGS16/4,dt.0005/control20Hz/.35rad/s. Privileged96+9teacher,
noDR/student/hardwareclaim. Frozen deterministic actor with its OWN normalizer;
existing observationnoise seeded43. No resets, switches or training in evaluation.

Common boya_workspace spec from actual v3training record and pinnedgeometryJSON:
centerz>=.066801253194m;XYlower[-.459866091313,-.069149973487],
upper[-.259054954510,.076535408821]m in environment-local coordinates.
Two consecutive20Hzoutside samples confirm taskregion failure. Contactcount is
auxiliary only. Numericalfinite/speed100/mimic.05 checks perphysicsstep. This region
is a geometry-based approximation,not a proven boundary of controllability/drop.
Each evaluation120simulationseconds/firstfailure/1500swall including startup.
No20straining timeout duringevaluation. Camerasoff/headless,4threads/nice10.

## Reuse, identity and allowed source migration

Keep active training/evaluator/physics/protocol sources unchanged. The CPU waiter
waits for v3normalbudget AND its already queued final evaluation; reuses that new
policy rollout,then executes just ONE additional oldpolicy rollout. No newGPUscene
now. Stop on interrupteddependency/error/sourceidentity mismatch; noautoretry.

Reuse existing comparison script/module. Oldexecution descriptor explicitly retains
oldtrainingrecord/hash/controller/checkpoint/sources, alongside newruntime sources
and commonworkspace rule. Original records and weight artifacts remain untouched.
Both model/observation/action/dynamics implementations must match. The known
isaac_parallel.py change splits existing groupcontact computation and additionally
returns finger_contact_count/palm_contact. Compatibility permits ONLY these exact
diagnostic edits,then compares the ENTIRE module AST; any other physics change
fails. All other requiredcore hashes and archived_frame/observe/spin_increment ASTs
must match. Archive/source/asset pins checked before execution. No blanket physics
hash bypass. Finalartifact samplecounts/lineage/normalizer required.

Independently recordedinitialq/qdot/objectstate/commands must agree1e-6absolute,
rtol0. Source/contracts/spec/seed/window/counts/resets/switches validated. Duplicate
waiters prevented byexclusivequeueclaim. Oldheight queue remains stopped.

## Outputs

outputs/boya_hora_matched_workspace10m_v1/comparison.json,csv,md.
Main120s signed valid-prefix moving-cylinder-axis angle;30s supplementary,peak,
backward,duration,windowstop/sourceepisodestop,tailprogress andwindow-specific
valid drift/tilt/force maxima. Fullrawtraces stayignored. Oldhistorical63.63deg
strict-score is never inserted as a commonworkspace score. Singleseed/originalstate
pair,notSOTA/indefiniterotation/sim2realproof. Failed/earlystopped episodes retained.

## Necessary verification

Focused comparison CPU regressions plus realmanifest compatibility preparation;
no auxiliary GPU evaluation or training interruption. Queue launch/status recorded.
