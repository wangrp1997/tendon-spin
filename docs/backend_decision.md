# Backend decision

Start learning in packaged native MuJoCo, reuse the existing GPU Torch environment.
The entire original u64 executed episode, complete integration state and95-feature
observations replayed exactly after mesh relocation; see data/migration.json.
This preserves the verified 13-position kp10/saturation/coupling/contact contract.
The old131072-action pilot took325.23s including evaluations, so a bounded native
learning trial is currently practical. Learning does not require Isaac Sim.

Existing Isaac Sim6.1.0.0/Isaac Lab3.0.0rc1 are found in env_isaaclab. No environment
was rebuilt or package installed. The minimal headless launch reported that the
Omniverse EULA is not accepted for this runtime; no acceptance flag was supplied.
Record this as launch-unverified, not a defective physics engine. Package versions
and the local probe are retained under data and outputs respectively.

Even after launch is available, a Boya Isaac task needs independently verified:
the packaged CAD/URDF collision and self-contact geometry, passive finger coupling,
13 active position command units/preload/gains/limits, original unsupported grasp,
full gravity and object dimensions, contact sensors and per-step validity gates.
GPU batching is useful once that contract matches. Isaac performance/scores must
remain separate from MuJoCo, and a Sharpa/Allegro policy is not a Boya demo.

This is a first-stage backend decision, not a permanent rejection of Isaac.
