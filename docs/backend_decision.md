# Backend decision

Start learning in packaged native MuJoCo, reuse the existing GPU Torch environment.
The entire original u64 executed episode, complete integration state and95-feature
observations replayed exactly after mesh relocation; see data/migration.json.
This preserves the verified 13-position kp10/saturation/coupling/contact contract.
The old131072-action pilot took325.23s including evaluations, so a bounded native
learning trial is currently practical. Learning does not require Isaac Sim.

Existing Isaac Sim6.1.0.0/Isaac Lab3.0.0rc1 are found in env_isaaclab. No environment
was rebuilt or package installed. The historical first minimal launch stopped at the unaccepted Omniverse EULA;
its record remains in data/isaac_environment.json. The user then explicitly
accepted the license for this process. A direct SimulationApp follow-up timed
out at150s without returning the constructor; preserve that separate evidence
in data/isaac_direct_startup_timeout.json.

The user authorized one official-headless follow-up with a5-minute ceiling.
Installed AppLauncher + isaaclab.python.headless.kit returned, reset and3
nonrendering physics steps completed; process exited0 after3.951799s. Default
SimulationApp.close fast shutdown exits the process, so the post-close marker
expected by the protocol was not observed; this is stated explicitly in
experiments/2026-10-08-isaaclab-probe/README.md. Startup/stepping are verified;
Boya model, batched dynamics and training remain unverified. No package/global
environment was rebuilt or changed. Process-only EULA confirmation is authorized
and must not be requested again. Keep direct/full-app timeout and official
headless success separate; root cause of the full-app hang is not isolated.

Even after launch is available, a Boya Isaac task needs independently verified:
the packaged CAD/URDF collision and self-contact geometry, passive finger coupling,
13 active position command units/preload/gains/limits, original unsupported grasp,
full gravity and object dimensions, contact sensors and per-step validity gates.
GPU batching is useful once that contract matches. Isaac performance/scores must
remain separate from MuJoCo, and a Sharpa/Allegro policy is not a Boya demo.

This is a first-stage backend decision, not a permanent rejection of Isaac.
