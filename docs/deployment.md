# Deployment contract

Hardware deployment is a required project outcome, not an inference from nominal
teacher reward. Source interface notes are copied at docs/history/real_robot_plan.md.
Boya has18 active actuator commands; learning controls TH/FF/MF/RF13 while wrist
and LF retain calibrated hold references. Passive finger joints are not directly
commanded. Confirm measured motor/estimated-joint meanings and named order.

The deployed student consumes20 frames of13 measured motor positions,13 already
executed position commands, four pad-frame3D forces and13 previous actions
(51 features/frame). Object pose, velocity, true contact geometry, passive joint
truth and aggregate hand-link forces are absent. More sensing can be added only
through a declared revised schema and separately calibrated student. Teacher95
features and physically valid task monitoring remain diagnostic privileges.

tendonspin/deploy/runtime.py loads only a student-role checkpoint with this schema,
actuator-name mapping, calibration identity,13 measured command bounds/rates and
sensor timing limits. It rejects teacher checkpoints, malformed/stale/nonadvancing
measurements and returns position targets with explicit published_to_hardware=False.
This is an offline inference/command adapter, not a robot driver or trained student.
No ROS command has been sent. Tests use a random student only for schema/timing/
bounded-command checks and do not count as learning or hardware rotation.

Required stages: stable nominal teacher; identify motor gains/delay/deadzone/
backlash/coupling and tactile zero/frame/noise/time alignment; fit training
randomization ranges from measured data; train teacher across these errors and
distill observation-history/tactile student; export the frozen student and test
recorded-input inference timing; connect the existing ROS position driver after
validated stop/hold semantics, then execute and score uninterrupted hardware runs.
Source notes identify a driver fallback that changes near-zero speed to3.14;
zero-speed writes therefore cannot be assumed to mean stop. Simulation12N is not
automatically a hardware force limit. Store measured limits in the deployment bundle.

The current host training environment is reused. Student inference requires Torch/
NumPy only and does not import Isaac or MuJoCo. At20Hz record p50/p95/p99 compute,
communication delay and dropped frames on the actual host. Hardware robustness,
student accuracy and continuous deployment are all pending physical evidence.
