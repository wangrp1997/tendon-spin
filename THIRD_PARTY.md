# Imported source identity

Boya meshes/source MJCF/URDF and saved grasp originate from
botyard-inhand revision `204d197a9606fb7266e884f3b2e6110195be01cb`.
Asset scene changes only mesh paths. Native runner and spin/PPO modules are
adapted from that project. The public model contract is in `assets/config/`.
The right-hand V3.2 source comes from `botyard.git` branch
`fix/mujoco-v3.2` at `28f4ba5649714c74c13f881891d579bfceb766e2`.
The simulation-left asset is generated from that right source by reflecting
meshes, body/joint frames and inertia while retaining right-hand masses and
control parameters. Do not substitute the separately authored vendor-left CAD
when exact bilateral policy transfer is required.

Hora code is copied from the official HaozhiQi/hora tagv0.0.1 archive. Preserve its
LICENSE and README; retrieval hash/version recorded in references/source_manifest.json.
Sharpa reference snapshot comes from fork revision5accf024d376685eaa17da7aa4614498217eab4d;
Sharpa LICENSE/NOTICE and component notices are retained. References are not
automatically runnable Boya ports or validated paper-result reproductions.

Copied patched MuJoCo binaries/headers are local ignored runtime assets, retained
under their original Apache license. Existing NVIDIA Isaac installations are reused
read-only; no NVIDIA software is vendored or EULA accepted by this repository.
