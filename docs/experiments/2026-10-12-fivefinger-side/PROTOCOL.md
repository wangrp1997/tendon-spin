# Bounded original-pose side-contact proposal

User requests preserving the old four-finger posture, allowing LF lateral pulp,
and spreading contacts without spending long. One geometry solve, <=120 objective
evaluations /120s wall; stop and report, no retry or size/backend scan.
Fixed48x60mm cylinder50g; original40x32/grasp44 reference; same native patched
MuJoCo3.13.0/CAD/collisions,16finger motors/held wrists/passive1:1distal mimics.
Only forward-position geometry calls;0physics steps/training/actions/controllers.
Privileged CAD state for planning only. Physical engine/gains/timing unchanged.

Reuse existing CAD nearest-point/convex separation solver, allowing distal palmar
and lateral surfaces (signed local y>=-0.5mm,z6–32mm), no central x coordinate
or frontal-normal target. Original13 joints within0.35rad; center within12mm per
coordinate; axis-angle departure<=12deg. Geometry screen: pad gaps±0.2mm,
overlap<=0.15mm, nonpad/interfinger gap>=0.3mm, contacts1mm inside cylinder ends,
pairwise contact spacing>=12mm and axial span>=20mm. These planning goals do not
change old physical-cache admission. No per-finger force gate. Existing static
force solve is initializer diagnostic only, conditional on geometry passing.
Render original and candidate for user to see; candidate stays unvalidated.
No physical hold/reload/sampling/training; no extra checks or render review.
Keep prior assets/caches/weights and evaluators/defaults. Save small result/source
identity, commit/push; large outputs local. Ask on a blocker before route changes.
