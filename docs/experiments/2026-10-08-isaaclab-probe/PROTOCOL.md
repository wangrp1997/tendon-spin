# One bounded Isaac Lab startup diagnostic

Read AGENTS.md, docs/experiment_state.md, docs/backend_decision.md and the
teacher-v2 protocol/results before this diagnostic. The user explicitly accepted
the Omniverse license for this process and authorized this one official-headless
follow-up with a maximum 5-minute wall budget after the direct SimulationApp
probe timed out at 150 seconds. No repeated launch or backend substitution is
implicitly authorized by a timeout.

New factor: installed Isaac Lab AppLauncher with its official
isaaclab.python.headless.kit experience, no cameras or rendering. Reuse existing
env_isaaclab (Isaac Sim 6.1.0.0, Isaac Lab 3.0.0rc1); install/change no packages.
OMNI_KIT_ACCEPT_EULA=YES is supplied only to this process. Device cuda:0;
SimulationCfg uses default PhysX, dt=.005s, gravity (0,0,-9.81).

No object, Boya hand, grasp, contacts, controller or observations in this empty
scene. Call reset and exactly 3 nonrendering physics steps. Acceptance requires
AppLauncher construction, reset, all 3 steps and close to return successfully;
this establishes only empty-scene startup/stepping, not Boya dynamics, training,
GPU batch performance, baseline reproduction or rotation.

External timeout: 298s plus at most 2s termination grace (300s total). Save the
full launch log and incrementally flushed phase record, source/experience hashes,
actual stop reason and parent process return code. If blocked, stop dependent
model/training work and ask the user with evidence. Independent literature and
metric audits continue.
