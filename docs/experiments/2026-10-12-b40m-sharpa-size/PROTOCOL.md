# B40M frozen-policy transfer to the Sharpa recommended cylinder

User authorization: select the best B checkpoint, try the Sharpa physical
cylinder dimensions, deliver a video, preserve the original size and provide
size/shape parameters. The user's “40000万” is interpreted from this conversation
as B40M, actual40,001,536 actions, not a nonexistent400M checkpoint.

Checkpoint: outputs/boya_hora1024_pose_ab_continue50m_v1/arm_b/actions_040000000/
training/teacher_final.pth, SHA256
2edabd00a4232b02398e3bb2d45a08fc931dc6949aca16c31ba36865101330fc.
Freeze this policy and its observation normalizer. No training or optimizer updates.

One new episode, requested30s or first existing physical/numerical stop;1500s
evaluation wall budget. No retries, alternative grasps, settling stage or scan.
Replace only the cylinder geometry: diameter48mm, axial height60mm. This is the
Sharpa README recommendation, not a measured object from its video. Keep50g mass
to isolate geometry; it is not a claim about the Sharpa hardware object's mass.
Retain the original40×32mm/50g profile, assets, cache, weights and recorded videos.

Keep original grasp44 hand joints/velocities/motor commands, object center
[-.365438990352,.018340188315,.106061099740]m and WXYZ
[.175062180804,-.901031779718,.346105303444,-.194180544127]. No center/orientation
adjustment for the larger object. Initial penetration or grasp failure is an
outcome of this direct transfer, not an invitation to silently regrasp.

Same IsaacSim6.1.0.0/IsaacLab3.0.0rc1/GPUPhysX/original_tgs16_4, CAD hand
convex collision, self-collision/28 excludes, friction.5, CCDoff, full gravity,
v3 external clipped PD. Object remains a native primitive collider, automatic
inertia for its changed geometry at fixed mass. No support floor.
dt.0005s,20Hz/100 physics steps per action. Five fingers/16 active motors,
both wrists held and four distal mimic joints passive.

Keep B pose-derived reward/terminal cost16 configuration, numerical checks and
the fixed ORIGINAL Boya workspace planes/confirmation unchanged. The workspace
was derived using the original object; it is deliberately not recomputed here,
so this is a geometry-transfer diagnostic under that same fixed spatial gate.
96 proprioceptive+9 privileged teacher observations and noise remain as trained;
shape/size privileged constants remain unchanged and no new shape input is added.
The checkpoint was not trained on this larger geometry. The28-state cache is
loaded only for construction and is never sampled in evaluation.

Reuse unchanged frozen evaluator source at dfad6a4 as a separately attributed
object-parameter entry point. Author primitive dimensions through the existing
scene_setup hook BEFORE physics initialization; keep pinned physics/task/train
sources and their original hash checks intact. Save the resolved geometry,
callback evidence, checkpoint/source identities, exact initial state and full
per-physics-step actions/state/contact/termination trace in a separate output.

Check only the necessary CPU geometry/CLI construction before the sole episode.
Then render its measured world poses with Isaac RTX, using the same resolved
geometry definition, original-speed20fps plus initial/terminal holds. Background
floor and marker are visual only. No optional render review, extra test campaigns
or additional policy episodes. A renderer run adds no physics or training.

Report valid-prefix signed moving-local-Z net, peak, backward motion, actual
time and stop reason. Use the existing finite-motion signal: full30s, positive
net, positive final10s. An early failure remains a delivered result. Never join
this new episode to the old40×32mm episode or claim hardware success.
