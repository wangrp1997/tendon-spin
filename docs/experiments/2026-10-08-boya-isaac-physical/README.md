# Physical-v1 failed: initial pose attribution corrected

Requested controller boya_source_pd_hold_v1: original40x32mm/50g/grasp44, full
world-z gravity,13 zero incremental position actions at20Hz,18 motors with
source kp10/40 and .3/1Nm caps, four intended passive mimics, PhysX TGS16/4,
.5ms steps, CAD convex hull/self-collision, source .05/.5 damping explicitly
injected as -D*qdot. Maximum1000steps/.5s and300s process budget.

Actually logged34steps/.017s; nonfinite joint state stopped execution. Process
exit0/5.555206s means the diagnostic finished recording failure. The offline
[audit](data/failure_diagnosis.json) confirms native WXYZ was passed to the Lab3
XYZW initializer, and output poses were read with the wrong convention. First-step
root orientation differs180deg; distal positions differ up to181.6mm. Therefore
**the raw original-state-restored marker is false; no original-grasp benchmark
prefix or success exists. Raw max-tilt0 is also invalid.** Retain all raw bytes.

Raw efforts are requested combined source motor/passive efforts, not verified
applied torque. Contact matrix allzero used30 incorrectly built paths while the
runtime has24 rigid bodies; sensorzero is not proof of contact loss. Runtime
explicitly disabled requestedCCD on GPU. Speeds reach1.137e22rad/s before NaNs.
[Unconstrained damping linearization](data/damping_linearization.json) gives
explicit stability dt<7.92microseconds versus500microseconds used, but excludes
contacts/equality and actual imported inertia: numerical-risk evidence, not an
isolated explanation of the full failure. No learned policy was run.

[Parent record](../../data/isaac_boya_physical.json) now links both exact raw phase
bytes and a strict JSON derivative. Eight nonfinite coupling values become null
only in the derivative, with field paths and original hash recorded; other raw
fields and historical mistaken marker remain visible with this correction.
Executed v1 script/helper snapshots stay under outputs and match original hashes;
current source is explicitly the separately evaluated v2 version.

The user approved coordinate/contact-path/solver-damping corrections and one new
bounded run. [Latest physical-v2 result](../2026-10-08-boya-isaac-physical-v2/README.md)
verified initialization but still failed dynamics; do not hide that follow-up or
call the v1 error a failure of Hora/AnyRotate/Sharpa or global hand infeasibility.
