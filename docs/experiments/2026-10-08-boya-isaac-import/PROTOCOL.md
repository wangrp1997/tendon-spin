# Boya Isaac structural import diagnostic

Read docs/experiment_state.md, the official headless probe protocol/results,
backend_decision.md and hora_boya_port.md before this separate diagnostic.
The new factor is importing the actual packaged Boya CAD/URDF rather than an
empty stage. Use the already verified official headless entrypoint and existing
Isaac Sim6.1/Lab3 environment. Process-only license acceptance is authorized.

Input: assets/hand/boya.urdf, original self-contained24 CAD meshes. Fixed base,
no fixed-link merging, no mesh merging, collision from original URDF collision
meshes, Convex Hull, self-collision enabled, retain mimic joints. No material,
object size/grasp or contact gates are changed. Structural import only:
neutral drive overrides (target none, zero drive gains) are explicit, NOT the
13-input kp10 task controller. No cylinder or physical control/hold episode.
Geometry/potential collision schemas and four J1=J2 coupling schemas are inspected
in generated USD; original22 movable joints/13 action names retained as a contract.
Gravity, timing and observation privileges do not constitute an executed task in
this conversion-only diagnostic. No dynamic physical steps or training actions.

Acceptance: successful local USD conversion/opening, all22 expected revolute
joint names and all4 mimic slave schemas present, collision geometry present.
Report joint schema, references, units, self-collision and collision approximation
attributes; passive-drive/coupling physics and original unsupported grasp still
require later executed validation. USD schema presence is not proof of coupling
accuracy or stable holding.

Keep large generated USD/mesh layers in ignored outputs. Save source/asset hashes,
flushed phase record, official conversion configuration, full log and process exit.
Wall cap300s including at most2s termination grace. If required mimic/geometry
support cannot be preserved, stop dependent training and ask the user before
changing mechanics, backend, grasp, criteria or algorithm. No reinstall/global edits.
