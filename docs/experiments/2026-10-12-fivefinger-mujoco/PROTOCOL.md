# Parameterized MuJoCo five-finger grasp handoff

User approved fixing geometry/force planning then reproducing the old preload /
unsupported hold procedure, and requested a MuJoCo initial grasp for botyard-inhand
to reference. Latest steering requires selectable parameters and unchanged original
evaluation defaults. Write only TendonSpin. Preserve all old assets/cache/weights;
no policy learning, rotation-policy evaluation, hardware commands or original-repo
writes. A new grasp remains an explicitly selected configuration.

Target: Sharpa README recommended cylinder48mm diameter/60mm height,50g; same
packaged Boya CAD,16 fingers TH4+FF3+MF3+RF3+LF3, original wrists held, passive
distal1:1 mimics. Center/orientation/16-joint posture may be redesigned. No size,
friction, force-bound or gain scan. Original40x32mm/grasp44 stays the default.

Use existing isolated patched MuJoCo3.13.0/libSHA256
c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088,
the original native route. This is separate from Isaac evaluations and preceding
3.11 geometry-only attempts. No environment installation or engine-code change.
Keep packaged implicitfast/.5ms/gravity[0,0,-9.81]/elliptic cone/friction.5/
condim3/solref.003/noslip5/28collision exclusions/original motor gains and limits.
Privileged CAD/object/contact state is allowed only for initialization/diagnosis.

Geometry/force budget: first fixed-geometry force feasibility of the preceding
best candidate using analytic equilibrium/friction derivatives(SLSQP<=300it).
Then at most TWO deterministic geometry seeds: preceding best and its same
posture with old original-axis interpolation. Each geometry least-squares<=300nfev
plus constrained refinement<=400iterations, followed by fixed force solve<=300it.
No random restarts or physical trial pool. Use CAD surface formulation reused
from botyard-inhand204d197a9606fb7266e884f3b2e6110195be01cb. Correct near-zero
mesh/mesh clearance queries with a conservative separating-face-axis certificate
on the SAME convex CAD meshes; a nonpositive certificate does not prove overlap.
It cannot certify positive separation. Keep all engine query/certificate values.
Frozen-geometry force solves distinguish a force-feasibility failure from a
mixed nonlinear optimizer failure; no global impossibility claim.

Screen: five pad gaps within±.2mm/overlap<=.15mm, side contacts1mm inside ends,
pad normal alignment>=.95; nonpad and crossfinger positive separation>=.3mm;
force balance and radius-normalized torque residual<=.01N, conservative planning
friction.4, normal forces.7–2.5N(tolerance1e-5N). No criterion relaxation to pass.

For the FIRST screened candidate only, execute ONE native MuJoCo initialization:
supported object at the declared pose for<=10s, old100Hz normal-force-admittance
preload ported to all five fingers, ramp.5s, release after all5 pad contacts and
normal loads within.12N of goals for.3s. This temporary weld support is recorded
as initialization, excluded from unsupported-hold results. No resets or object
forces/pose feedback in the subsequent hold; freeze motor targets at release.
No controller substitution, preload coefficient scan or retry. Wrist commands
stay at original targets; no extra LF lock/scaling. Numerical/speed>100rad/s /
mimic>.05rad hard failures stop initialization. Save full-step initialization.

Then one uninterrupted5s unsupported holding episode, full-step state/actions /
command/forces/contact gates. Require each pad compression>.08N at EVERY hold
step, each group force<12N, drift<5mm, full orientation change<10deg, object/self
penetration<1.5mm, nonpad/floor force<.05N, no support/external forces/nonfinite /
solver warning/speed>100rad/s/mimic>.05rad. This matches the old four-pad holding
criteria with a fifth pad and explicit numerical/coupling checks. Stop on first
failure. Report signed valid-prefix net/peak/backward as incidental motion and
actual stop. No perturbation or rotation-success claim.

If5s passes, save frozen full state plus portable named-joint/object/command JSON
and model identity, then independently reload exactly that state once for30s
using the same frozen targets/gates. This conditional reload is part of verifying
the transferable initial-state artifact, not a second optimization/controller.
Report the5s and30s separately, never sum their durations or angles. If reload
fails preserve both and mark the artifact unvalidated. No further evaluation.

Expose explicit object/grasp parameters; original evaluation code/defaults remain
unchanged. A validated new grasp may be shipped as a separate profile; an
unsuccessful candidate stays labeled proposal only. No new training cache or
Isaac-transfer claim. Stop after declared budget/failure, record small evidence /
portable assets and push; raw traces/runtime remain ignored. No optional tests,
video or broad audit campaign. Ask before any subsequent configuration/route change.
