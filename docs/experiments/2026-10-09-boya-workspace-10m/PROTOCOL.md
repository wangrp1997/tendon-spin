# Boya manipulation-region termination: fresh 1024 / 10M

User authorizes fixing the discussed drop definition and restarting promptly,
without extra experiments/check passes. New factor versus stopped height-v2:
replace nominal-minus-5mm height with a declared Boya CAD-derived task region.
Previous v2 stopped at495updates/4055040actions; its33weights deleted by user.
First strict10M80weights remain. No model is resumed or merged into this run.

Same cylinder diameter40mm/length32mm/mass50g, original grasp44/fullgravity,
16finger actions/heldwrists/4passive1:1couplings, Isaac6.1/Lab3rc1, v3 clippedPD,
uncalibrated A=4*D_effective*dt,28collisionexcludes/CADconvexselfcollision,
TGS16/4,dt.0005/control20Hz/.35rad/s, same28statecache. No floor/external support.
Same HoraPPO/reward/96proprio+9privileged/8latent, nominal physics/noDR/no student.
No new actor inputs. Not a full Hora reproduction or validated transfer method.

## Task region and failure

Profile boya_workspace loads assets/grasp/rotation_workspace.json, derived by
scripts/derive_boya_workspace.py using original contract body frames and CAD.
No new physics required for this derivation. Fixed environment-local boundaries:
- Palm mesh highest world-z point: .041188756244m.
- Cylinder circumsphere radius sqrt(.02^2+.016^2)=.025612496950m.
- Center-height lower bound=sum=.066801253194m. Nominal center=.106061099740m,
  so available descent is39.259846546mm (not initial-minus5mm).
- XY bounds from ALL FIVE fingers' middle/distal mesh envelope, expanded by that
  cylinder radius: lower[-.459866091313,-.069149973487],
  upper[-.259054954510,.076535408821]m.

At20Hz control boundaries, two consecutive samples outside either height or XY
region confirm failure. Returning inside clears the counter; episode reset clears
only corresponding counters. Report below-manipulation-region and lateral-escape
separately. Single transient outlier and fewer than2contact fingers do not reset.
Physical simulation contact grouped per finger and palm is recorded as diagnostics,
not tactile-array observations. Normal-force numerical floor1e-6N only identifies
reported contact; it is not a physical support/stability or task-done threshold.

The palm bound deliberately rejects a cylinder resting on the palm: its center
cannot exceed palm-top plus bounding radius. XY envelope is a practical static
approximation, NOT a proven full kinematic/controllability set. A crossing is a
DECLARED TASK-REGION failure, not proof of unrecoverability or true freefall.
No claim that all in-region poses remain manipulable. Geometry may need subsequent
evidence-driven revision; any revision requires a distinct experiment identity.

Training400controls=20s timeout. Nonfinite/>100rad/s/mimicerror>.05rad numerical
checks unchanged perphysicsstep. Olddrift5mm/tilt15deg/force12N remain diagnostics
and legacyshadow, not taskreset. Train/eval share exactspec and source/asset hashes.

## Budget and delivery

Fromscratch seed43/network/normalizers/Adam/RNG.1024headless/camerasoff,
8steps/update/minibatch512/5epochs, original HoraLR.005/KL.02/gamma.99/tau.95.
10000000requested/10002432rounded/1221updates. nice10/4threads, no watchdog,
resourcequota, GPUmemorycutoff or trainingwallbudget. Process-onlyEULA accepted.
Summarylogs/TensorBoard, paired resumable first/every16/finalcheckpoints.
Output outputs/boya_hora1024_workspace10m_v3. Stop afterbudget, noautoresume.

Existing supervisor performs ONE final frozen original-grasp44 120swindow
(firstfailure/1500swallbudget),0resets/0switches, fullphysicaltrace and oldstrict
shadow. Signed valid-prefix net/peak/backward/duration/stopreason, no720gate.
Task region differs from older profiles; historical scores cannot be directly
ranked. The former matchedheightqueue stays stopped; no oldpolicycomparison is
silently migrated to newcriteria. No automatic student/DR/hardware work.

## Necessary verification only

CPU regression of old5mm crossing, palm/lateral failure, shorttransient recovery,
partial-reset confirmation history; existing termination checks retained.
Then launch actual authorizedrun and verify firstupdate/checkpoint; no auxiliary
GPUexperiment, hyperparameterscan or rendercheck.
