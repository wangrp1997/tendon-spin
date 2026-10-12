# One physical hold after the user's gap correction

Latest user explicitly accepts geometric gaps if the grasp actually holds and
can be used for later manipulation. This supersedes the geometry-veto/no-physics
boundary of PROTOCOL.md for this single episode. Retain the failed geometry screen
as history; do not claim it passed. Fixed candidate from outputs/boya_fivefinger_side_v1,
48x60mm50g/original-pose reference/native patchedMuJoCo3.13.0/same CAD and collision
settings/dt0.5ms/original gains and caps/16finger motors/held wrists/passive mimics.
Use the SAME static-force initializer diagnostic and frozen native target preload;
no force-feasibility, per-finger-bearing, gap or all-five admission gate.

Exactly ONE unsupported0.5s initialization+30s formal hold, no resets/support/
external forces/controller switch/retry. Same old physical cache formal rules:
finite, drift<=5mm, axis tilt<=15deg, speed<=100rad/s, object-to-link normal<=12N,
mimic error<=.05rad; >=2hand groups>1e-6N in>=90% of last0.1s. Initial nonfinite
stops; formal-start and every formal step checked. Stop first failure or budget.
Report actual per-finger/all-five touch, signed valid-prefix endpoint/peak/backward
and actual stop. Privileged simulation state/contact only; no learned policy.
Save full physics trace, small summary and passing named state separately.
No cache collection, independent reload, training, parameter scan or extension.
Preserve all old assets/cache/evaluators/weights; render this same trajectory only.
