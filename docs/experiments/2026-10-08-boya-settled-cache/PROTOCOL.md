# User-approved two-phase grasp initialization and screening

Read experiment_state, first parallel-cache README/protocol/results, sampling
failure diagnosis and nominal Hora pilot follow-up. User explicitly approved:
“采用分阶段筛选，再采样一次（推荐）”. This supersedes the pending-answer state.
User additionally notes that four-finger cylinder grasps may be scarce; neither
v1 nor this run requires four contacts. Contact count alone is not force closure.

New factor ONLY: separately declare .5s initialization settling followed by .5s
formal holding, same fixed controller/settings throughout; no reset or switch
between phases. Preserve historical v1 full-.5s screening result1/504 separately.
This is new physical execution, not offline re-scoring or repeated training.

64 independent Isaac environments,8batches,seed42,504 random candidates plus8
nominal anchors excluded from random cache counts. Same Hora±.25rad16-input samples
with source preload, original40×32mm/50g/grasp44 reference pose and full world−z
9.81gravity. All five fingers controllable, both wrists held,4distal1:1mimics.
V3 external clipped PD/declared uncalibrated armature/28collision excludes/CAD convex
self-collision/TGS16/4,dt.0005s,20Hz zero actions, no ground/external support spawned.
Simulation truth is used for selection; no deployed-policy or hardware claim.

Keep ALL initialization steps and their violations as diagnostics. Any nonfinite
initialization invalidates that candidate; it cannot be rehabilitated by a later
finite value. At .5s save each candidate's formal-start state. Check formal-start
and every subsequent physics state for drift>5mm from ORIGINAL cylinder center,
axis tilt>15deg from ORIGINAL axis,speed>100rad/s,link normal>12N,mimic error>.05rad
or nonfinite. These checks are identical numerical thresholds to v1. Never recenter
on a moved object. The .5s formal segment must pass in its entirety; final.1s must
observe at least2 hand contact groups in90% of steps (TH/FF/MF/RF/LF/palm grouped
as before). No pad-only or four-finger gate. Missing sensor-only contact is not
proof of physical contact loss. Diagnostic counts include actual group membership.

Each candidate is evaluated once. Planned16000 batched physics steps. Process
budget300s; cease collection at270s and save. Preserve source/config and each step's
actions/commands/q/v/efforts/poses/contact/phase/first failure; save accepted final
states only. Initialization validity is NOT relabeled as continuous manipulation
success. This cache screen does not establish full penetration certification,
sustained rotation,force closure against arbitrary perturbations or hardware safety.
No additional parameter sweep or extra simulator validation run in this stage.
