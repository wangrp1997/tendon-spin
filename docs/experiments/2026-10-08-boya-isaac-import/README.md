# Boya USD structural import: passed, no executed grasp

Read current experiment_state, official-headless probe and Hora port checklist.
This import used packaged Boya URDF/24 CAD meshes, fixed base, no fixed/mesh merging,
convex hull and enabled self-collision; retained original22 revolute names and
four passive mimic slave schemas, neutral zero drive overrides, no cylinder,
controller action, physics integration or training.

Local conversion/opening completed in4.440463s, exit0. All22 expected names,
four NewtonMimicAPI relations and collision geometry were present. Unit scale1m,
Z-up. The current importer intentionally authors NewtonMimicAPI rather than the
legacy PhysxMimicJointAPI; schema presence alone proves neither physical coupling
nor stable holding. The subsequent physical-v1 failure and physical-v2 follow-up
are separate original-state diagnostics, not a structural-import success score.

See [protocol](PROTOCOL.md), [parent evidence](../../data/isaac_boya_import.json),
[phase/USD audit](../../data/isaac_boya_import_phases.json). USD, source snapshot and
launch log are in ignored outputs with hashes. Model/assets and native source
repository were unchanged. Latest physical outcome is recorded in
[physical-v2](../2026-10-08-boya-isaac-physical-v2/README.md).
