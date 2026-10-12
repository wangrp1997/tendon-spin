# Five-finger grasp feasibility and one holding diagnostic

User asks to inspect/rework the initial grasp and permits changing object size
if needed for five-finger contact. No training, old-policy re-evaluation or
original cache replacement is authorized by this diagnostic.

Preserve all historical40x32mm/48x60mm profiles, grasp44, cache and weights.
The original grasp planner optimized13 joints/TH-FF-MF-RF; the current learning
interface already controls16 motors/TH4+FF3+MF3+RF3+LF3. Both wrists remain held
and all four distal mimics remain passive1:1. No new controller, gain or engine.

First use packaged CAD, MuJoCo3.11.0 FORWARD POSITION/convex distance only and
SciPy bounded least squares, with ZERO dynamics steps. This is a geometry
proposal, not a MuJoCo controller result or substitute execution backend.
Reuse botyard-inhand grasp/generate_grasp.py at204d197a9606fb7266e884f3b2e6110195be01cb
concepts (CAD contact distances, coupling, limits, nonpad/inter-finger clearance)
in a separately attributed five-finger proposal entry point; no original-repo writes.

Fixed ordered CPU budget: at most THREE local bounded optimizations, each
max180 least-squares evaluations. Stage1 changes only three LF joints, with
40x32mm and original object pose. Stage2 keeps40x32mm and original orientation,
allows all16 finger joints within existing limits and center translation±15mm.
Only if needed, stage3 additionally allows cylinder diameter36-56mm and axial
height32-70mm. Keep50g mass. No shape change or orientation scan. Stop early
on a usable geometric candidate; no indefinite/random restarts.

Require five distal CAD surfaces near contact (absolute gap<=.2mm), with each
contact on its pad-facing distal surface, no object/nonpad or inter-finger
penetration beyond.2mm. Geometric overlap used as preload proposal is explicitly
limited to.15mm. Record signed gaps, nearest points, all joints/pose/dimensions,
source/runtime identities, counts and any failed intermediate geometry. These
conditions do not prove force closure or stable dynamics.

If a candidate meets these geometry conditions, execute exactly ONE new native
Isaac GPUPhysX/original_tgs16_4 holding episode, one environment, same CAD convex
collisions/28 exclusions/friction.5/CCDoff/full gravity/v3 external clippedPD/
dt.0005s/20Hz. The newly declared grasp/size is its initial state; it is NOT the
original-grasp44 policy benchmark. Fixed initial position targets, zero policy
actions, no learned controller and no support floor/physical external forces.
Use inherited preload for unchanged motors and declared small geometry-derived
preload for newly repositioned fingers; freeze commands before the episode.

Bounded5s continuous episode: .5s initialization then4.5s holding, same commands,
no reset/controller switch. Nonfinite state, speed>100rad/s or mimic error>.05rad
stops immediately. After initialization, stop on drift>5mm from the NEW declared
initial center, tilt>15deg from its declared initial axis, or link normal>12N.
Keep all initialization violations as diagnostics. Save per-physics-step raw
actions/state/contact/effort/physical gates and raw provenance.

Pass requires all4.5s holding gates plus all FIVE finger normal-force groups
>=.05N simultaneously in>=90% of the final.5s. Report individual contact
fractions/min/max forces, drift/tilt/peak force, actual stop and phases. Also
report signed valid-prefix endpoint angle/peak/backward as incidental holding
motion, not a learned rotation score. No perturbation campaign or force-closure
claim from this one hold. A candidate that fails stops this diagnostic; present
concrete options rather than retuning/retrying or starting training.

If no geometric candidate is found, stop without a physical episode and report
the residual/active limits and options. Do not infer global impossibility from
local optimization failure. Record small evidence underdocs and milestonepush;
raw trajectories remainlocal. No optional audits/tests/video campaign.
