# Baseline reuse and virtual touch

We first reuse complete reference mechanisms and validate Boya adaptations;
PPO-v2 remains a separately identified diagnostic, not an original Hora baseline.
The current trial sources are frozen; the added reference modules are not used
in its policy or evaluation.

`baseline/reference_models.py` (under `tendonspin/baselines`) loads the pinned
Hora ActorCritic, privileged encoder and RunningMeanStd directly. Boya changes
only the declared channel/action dimensions. The 30-frame TCN is imported from
Sharpa's width-parameterized version. At the original32-channel width, the two
TCNs must agree exactly after copying weights. Stage-two initialization copies
the teacher and optimizes only the history encoder, following Hora padapt.
These modules are wired and tensor-tested; task observations, original rewards,
randomization, teacher training and student execution still need full porting.
They are not scored reproductions or a new hybrid task controller.

AnyRotate Appendix F constructs virtual fingertip sensing from simulator contact
force/position, applies a contact threshold, EMA, saturation/scaling and masking.
It does not render an entire real tactile image for policy training. On hardware
its CNN extracts contact pose and force from tactile images. Sharpa similarly
uses net contact forces, filtering and local-frame contact positions; contact
positions are disabled by its current default configuration.

`baselines/touch.py` implements AnyRotate's explicitly cited equations. Its input
is fingertip-calibrated sensing, not all-link support force or object-state truth.
Pose absence is explicitly masked; paper-default ranges are not Boya calibration.
The real Boya point-force array may supply weighted contact-location/distribution
features once geometry, units, zero point and coordinates are calibrated. Current
repo work does not synthesize a116-point grid or claim these features are measured.

The read-only `/home/rw/botyard_ws` ROS messages define5 finger slots, each
`TactilePoint[116]`, with point_id and raw int8 fx/fy/fz. **116 is declared message
capacity, not a verified active hardware point count.** Publication sets the
ROS header to current publish time and uses a10ms timer; acquisition timestamp,
actual freshness, point geometry and Newton conversion are not proven by this.
The teacher's existing95 truth features do not become deployment observations.

Next complete baseline: Hora teacher/history adaptation on13 real inputs, then
tactile-feature comparison based on AnyRotate/Sharpa. Use native MuJoCo for
contract verification; prefer the existing Isaac Lab for large batching after
passive coupling, actuator control and unsupported-grasp contact transfer pass
separate checks. Do not rewrite array physics as a prerequisite.
