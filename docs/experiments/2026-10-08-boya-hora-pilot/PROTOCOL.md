# First original Hora PPO integration on Boya

Read current experiment_state, parallel-cache README/result and Hora source/config.
New factor: actually train pinned Hora v0.0.1 PPO/ActorCritic/attribute encoder
against the executed 64-environment Boya interface. This is a nominal integration
pilot on the ONE accepted cache state, not full Hora reproduction or robust training.
Prior teacher-v2 remains a different algorithm/configuration and is not resumed.

Original40×32mm/50g/full gravity/grasp44 reference orientation; v3 Isaac physics,
16 independent finger actions, wrists held, four passive mimics. Cache reset restores
saved q/object pose and zero velocity as Hora does; preserve saved Boya preloaded
motor targets. Physics dt=.0005s, control20Hz, source rate .35rad/s (max .0175rad
per action) and target interpolation retained. Hora uses1/24rad and different motor
PD: these morphology/hardware port differences are declared, not silently equated.

Actor: three32-channel frames (scaled measured joint q plus raw motor targets),
±.02rad joint observation noise, 30-frame history retained. Teacher priv9: object
position relative to environment origin, scale1, mass.05 normalized[0,.2],
friction.5 normalized[0,1], COM0. Encoder[256,128,8], actor[512,256,128].
No tactile student yet. No property/size/PD randomization in THIS nominal pilot;
those full reproduction mechanisms are explicitly pending, not claimed complete.

Reuse upstream pure reward: clipped signed angular speed[-.5,.5] ×1, object
linear L1 speed ×−.3, pose squared error ×−.3, motor torque squared ×−.1, squared
net mechanical power ×−2. Rotate about negative initial cylinder long axis rather
than Allegro's world−z. Actual signed angle separately integrates moving local axis.
Torque/work are evaluated on16 active motors at the last physics substep. All
18 motor commands/22 joint states are logged. No new reward sweep.

Each control action checks every physics step for nonfinite/drift>5mm/tilt>15deg/
joint speed>100rad/s/link normal>12N/mimic error>.05rad. The first failed prefix
is retained; reset occurs at the end of that control interval and post-failure
steps remain marked invalid. At400 actions(20s) truncate as upstream; timeout
bootstrapping retained. These are stricter Boya terminations than Hora height-only.
Missing sensor-only support is diagnostic, not a training termination rule.
No full self-penetration certification; no benchmark/hardware claim.

Seed43;64envs,horizon8,minibatch512,5 PPO epochs per update,Adam lr.005,KL target.02,
upstream normalization/GAE/coefficients. 16 updates=8192 actions planned, wall budget
300s, finish current update then save at240s. Intermediate local checkpoint every4
updates. Preserve original source algorithm; only replace unused legacy gym import
and tensorboardX writer dependency at load time using installed torch TensorBoard.
Store per-step state/actions/contact/flags, input observations, actual sample count,
losses/checkpoint identity. Upstream .train() counter offset is avoided by calling
.train_epoch() from actual count0. No warm start, no mixed-controller scoring.

The milestone requires finite rollout/reward/loss and a saved updated checkpoint.
It does not require or establish continuous rotation. The first frozen original-state
120s evaluation remains a separate next stage; training resets never form its score.
