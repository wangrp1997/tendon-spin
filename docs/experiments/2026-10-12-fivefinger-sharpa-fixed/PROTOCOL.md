# Fixed Sharpa-recommended cylinder, new five-finger grasp

The user approved starting after discussing fixed48mm diameter /60mm axial
height and redesigned orientation, center and posture. This supersedes the
original-orientation restriction for THIS diagnostic only. These dimensions
are the official Sharpa README recommendation, not measured demo dimensions.
Mass remains50g. Preserve old objects, grasp44, cache, weights and failed plans.
No training, old-policy rollout, new engine or physical hardware commands.

Geometry uses packaged CAD and existing MuJoCo3.11.0 FORWARD POSITION only,
zero integration. Port the smooth CAD pad-surface/contact-coordinate and
static friction-force formulation from botyard-inhand grasp/generate_grasp.py
at204d197a9606fb7266e884f3b2e6110195be01cb to five fingers/16 joints. Use a
convex-hull halfspace ray for the same spline samples; use mj_fwdPosition for
all clearance measurements, avoiding stale mesh BVHs. This is a proposal
calculation, not a substitute execution backend or validated force closure.

Bounded CPU budget: at most TWO deterministic initializations, each one bounded
least-squares solve(max300nfev) followed by SLSQP(max200iterations). Seed1 is
the old posture with LF flexion initialized like RF; seed2 has the four fingers
in a common curled posture and a cylinder-axis seed derived from their pad
positions. Both may change all16 finger joints within original limits, center
within40mm of old center and cylinder orientation. Fixed48x60mm throughout.
No random restarts, size sweep, retry loop or controller tuning. Stop on first
feasible geometric plan; otherwise report residuals and stop before Isaac.

Contact coordinates remain on CAD pad surfaces(x±6mm,z11–27mm). Require normals
aligned>=.95 toward the cylinder, side contacts>=1mm inside the cylinder ends,
five distal signed CAD gaps within±.2mm(no overlap beyond.15mm), and nonpad /
cross-finger clearance>=-.2mm. Planning targets are positive.05mm distal gap
and .3mm other clearance. Static equilibrium force residual<=.01N and normalized
torque residual<=.01N, using conservative friction.4 and normal forces.7–2.5N.
These are geometry/preload screening conditions, not hardware certification.

For a valid plan only: exactly ONE native Isaac GPUPhysX holding episode,
one environment, original_tgs16_4, CAD convex collisions/28 exclusions,
friction.5/CCDoff/full gravity, existing v3 clipped externalPD and nominal
armature, dt.0005s/20Hz,16 active finger actions, wrists held, four passive
1:1 distal mimics. No extra LF lock or multiplier. Privileged full simulation
state/contact is used for diagnostics; there is no learned policy observation.

Before starting the episode, freeze finger motor targets as planned q plus
the static contact Jacobian-transpose torque/kp, reflecting passive-slave
contributions into masters, clipped to existing motor command bounds. Retain
original wrist targets. Record desired and clipped preload/forces. This fixed
kinematic preload is a proposal, not dynamic force control or engine-equivalent
gravity compensation. Zero policy actions throughout; no floor/support forces.

One uninterrupted5s episode: .5s initialization then4.5s hold. Stop immediately
for nonfinite state, speed>100rad/s or mimic error>.05rad. From.5s stop for
drift>5mm from NEW initial center, tilt>15deg from NEW initial axis or link normal
>12N. Keep initialization violations. No reset/switch/regrasp. Pass requires
completion plus ALL FIVE force groups>=.05N simultaneously in>=90% of final.5s.
Record full per-physics-step actions/state/body poses/forces/efforts/gates, source
identities and initial-state readback. Report individual force/contact statistics,
drift/tilt, signed valid-prefix net angle/peak/backward and actual stop reason.
Incidental holding motion is not learned rotation success. Stop after this episode
or any execution error; no retries/extensions/training/video campaign. Record
small results in docs and push; raw traces remain ignored under outputs/.
