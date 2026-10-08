# Imported source identity

Boya meshes/source MJCF/URDF and saved grasp are copied from the user's local
robot and botyard-inhand204d197a9606fb7266e884f3b2e6110195be01cb; file SHA256 and
source paths are recorded in docs/data/import_manifest.json. Asset scene changes
only mesh paths. Native runner and spin/PPO modules are adapted from that project.

Hora code is copied from the official HaozhiQi/hora tagv0.0.1 archive. Preserve its
LICENSE and README; retrieval hash/version recorded in references/source_manifest.json.
Sharpa reference snapshot comes from the user's local fork5accf024d376685eaa17da7aa4614498217eab4d;
Sharpa LICENSE/NOTICE and component notices are retained. References are not
automatically runnable Boya ports or validated paper-result reproductions.

Copied patched MuJoCo binaries/headers are local ignored runtime assets, retained
under their original Apache license. Existing NVIDIA Isaac installations are reused
read-only; no NVIDIA software is vendored or EULA accepted by this repository.
