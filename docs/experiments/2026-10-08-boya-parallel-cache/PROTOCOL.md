# First Boya 16-input parallel grasp cache

Read current experiment_state, Hora port checklist and v3 hold record before execution.
New factor: 64 independent Isaac environments and the user-authorized 16-input layout,
then Hora v0.0.1 allegro_hand_grasp.py sampling at canonical q ±0.25 rad.
This is a small cache integration run, not a reproduced paper or rotation result.

Original 40×32mm/50g cylinder, grasp44 object/hand orientation, gravity (0,0,-9.81).
TH4+FF3+MF3+RF3+LF3, both wrists held, four passive 1:1 distal mimics.
Same v3 external clipped PD, uncalibrated armature A=4 D_effective dt, 28 source
collision exclusions, CAD convex collision/self-collision, TGS16/4, dt .5ms,
20Hz zero actions while holding each sampled target. CCD false matches the prior
GPU-effective setting. Simulation truth for cache selection, no policy training.
No ground or other external support is spawned. Full self-penetration not certified.

Seed 42; 8 batches ×64 instances ×0.5s; process budget 300s, stopping collection at
270s to save. Each batch env0 is an original nominal anchor, explicitly excluded
from random cache counts. Thus 504 random candidates and 8 anchor episodes planned.
Each candidate starts independently, with zero q velocity and original object state;
clamp active joints to source bounds, set passive q to master q. Targets preserve
original preload offset from q (Boya adaptation vs Hora sampled-q targets).

Reject candidate at first nonfinite state, drift >5mm, axis tilt >15deg, joint speed
>100rad/s, single-link normal force >12N, or mimic error >.05rad during the entire
0.5s. Store per-step records even after failure; never restore a failed prefix into
a success. For cache eligibility additionally require at least 2 observed hand
contact groups in >=90% of the final0.1s. A sensor rejection means insufficient
observed support, not independently proven contact loss. Any real hand region may
support; no all-fingertips distance gate (different morphology and user task).
These gates are a declared Boya screening adaptation, not the original Hora gates.

Save successful final q/v, commands, object state relative to environment origin,
source candidate identities, initial samples, per-step pre/post q/v/efforts/body and
object poses/contact/gates. Small result and failure counts committed; raw output
ignored. A completed finite hold only licenses provisional training reset data;
no continuous rotation, full physical certification, SOTA or hardware claim.
