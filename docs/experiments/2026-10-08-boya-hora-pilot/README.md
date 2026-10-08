# Original Hora teacher PPO now executes on Boya Isaac

64 environments,16 PPO updates,8192 executed control actions and819200 total
physics steps in121.031s wall. Stop: predeclared update budget. All rollout rewards
and optimizer losses finite; saved final network differs from initialization.
No warm start from teacher-v2, no controller substitution, no continuous-rotation score.

The directly reused Hora v0.0.1 PPO/ActorCritic/8-dimensional teacher encoder and
reward function execute against the new16-input Boya task. Vendored code stays
unchanged: loader removes an unused gym import and uses installed torch TensorBoard
in place of tensorboardX; no global environment changes or package install.
[Protocol](PROTOCOL.md) and [resolved config](config.yaml) declare all task differences.

760 training episodes ended: drift>5mm488,axis tilt>15deg198,link force>12N74.
No first termination from nonfinite physics,velocity>100rad/s or mimic error>.05rad.
These are exploratory training episodes from the single cached grasp, with resets.
Their angles are never pooled or used as an original-state frozen-policy score.
Remaining live episodes stop with the training budget; per-step histories retained.

Final losses: actor−.0183674,critic.204903,bounds43.7980,entropy22.4826,KL.0233387.
These establish finite optimizer execution only, not convergence or useful rotation.
Checkpoint outputs/boya_hora_nominal_v1/teacher_final.pth; intermediate u4/u8/u12/u16
also retained. [Result](result.json) records checkpoint SHA256 and exact source copies.
Raw all-env per-physics-step actions/commands/pre-post q/v/effort/body/object/contact
and prefix flags are in rollout_001.npz through rollout_016.npz (ignored outputs).
Control inputs/reward terms/dones and episode termination records are also saved.

Still incomplete: diverse cache(current1 state),property/size/PD randomization,
30-frame student adaptation,original-state frozen120s evaluation,complete physical
penetration certification and real hardware. Fixed privileged feature constants
use declared Boya scale1/friction[0,1] convention; upstream uses scale[.6,.9] and
friction[0,1.5],so this nominal feature port is not a numerically identical baseline.
Hardware-scaled action increment/PD and Boya failure gates also differ as declared.
This is a training integration milestone,not full Hora reproduction or SOTA.

First cache failure diagnosis from existing data only: most force violations happen
at the first physics step after random joint placement.26/504 candidates satisfy
only the final.1s gates,but historical full-prefix selection remains1/504. No
re-scoring into success, cache replacement or physical rerun was made. The proposed
separate initialization and hold-screening criterion is pending user confirmation
under AGENTS.md; original-grasp nominal pilot above is already completed.

No additional testing/audit or video work; local milestone commit, no push.

Latest follow-up: user approved the separate initialization criterion and the
[new cache execution](../2026-10-08-boya-settled-cache/README.md) produced28 states.
That resolves the pending-answer statement above. This pilot still used its original
one-state cache; no further PPO actions are attributed to the new cache.
