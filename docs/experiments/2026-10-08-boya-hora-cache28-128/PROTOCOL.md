# Hora Boya teacher: user-requested128 environments,28-grasp stage

Read experiment_state, nominal Hora pilot README/config and settled-cache README
before this stage. User asks whether to restart and begin actual rotation training,
after authorizing continuation with the28-state cache. Fresh network/normalizers/
optimizer,seed43; do not resume the8192-action single-grasp integration checkpoint.
New factors:28 uniform-sampled resets instead of1 and a larger sample budget instead of the16-update pilot.
Keep the same algorithm/reward/physical settings. No parameter search.

Isaac Sim6.1/Lab3 existing env,headless=True,enable_cameras=False,128environments.
Original40×32mm/50g/grasp44 reference orientation/full−9.81gravity,16finger actions,
wrists held,four passive1:1mimics. Same v3 declared nominal uncalibrated armature,
external clippedPD,28 source collision exclusions,convex CAD/self-collision,
TGS16/4,dt.0005s,control20Hz,command rate.35rad/s. At training reset pick one of
28accepted states,restore q/object pose and preloaded command,zero velocities as
previous pilot. No additional random joint placement or initialization exemption
inside training. Each step uses previous drift/tilt/force/speed/mimic/finite gates.
No four-finger contact condition. Record cache identity for every episode.

Direct Hora v0.0.1 PPO,ActorCritic,priv9-to8encoder,pure reward,GAE,normalizers,
Adam and hyperparameters as prior protocol. 96-frame-stacked proprioceptive input,
30-frame history,±.02rad joint noise,teacher truth attributes. Rotate about original
cylinder−axis. Morphology/action/reward-axis/physical-failure adaptations and feature
scaling differences remain as pilot. No property/size/PD randomization yet, no
student training. This nominal learning stage is not complete Hora reproduction.

128×horizon8×64updates=65536 training actions requested;minibatch512,5epochs,
initial learning rate.005,clip.2,gamma.99,tau.95,KL.02. Save weights plus optimizer
and RNG every16updates;save actual progress per update. Training wall cap1800s
(finishes current update); external hard cap1860s. Source/config/cache hashes and
all per-physics-step actions/commands/poses/contact/validity/episode/cache IDs stored.
No success implied by optimizer progress; active training episodes are budget-truncated.

After training, automatically evaluate the final available frozen checkpoint in a
SEPARATE Isaac process/one environment, restored ORIGINAL nominal q/v/object state
and original commands,not a cached starting pose. Zero resets/controller switches,
deterministic mean action,retained±.02rad observation noise with fixed seed43 and
frozen normalization. Control/gains/criteria unchanged throughout. Stop at the first
physical gate or120s simulated time; evaluator wall cap1500s/hard cap1560s.
Report signed valid-prefix net/peak/backward rotation,actual duration,30s and120s
windows and tail motion,physical/wall/time-limit reason. Never add training angles
or compare cached-start exploration as an original-state result. Full penetration
certification still absent,benchmark_validated=false. No hardware claims.

The supervisor records a running status and child PIDs; the stage may outlive the
interactive reply. Do not label queued evaluation as executed. On hard timeout or
process failure preserve evidence and stop; no automatic algorithm/backend changes.

User steering after the64-env stage started: modestly increase GPU use while
keeping desktop responsive. That64-env process was explicitly interrupted and
its partial data retained separately (no evaluation, no pooled training count).
This run restarts from seed43 with128envs;same65536-action stage budget,64updates,
rollout1024,minibatch512(two minibatches per epoch),5epochs. It does not pretend
the new batch geometry is identical to64envs. Checkpoints every16updates retain
16384-action cadence. nice+10 and4 Torch/OMP/MKL threads; no global GPU clock/power
changes. Record observed whole-device memory every8updates and first update;
stop after an update if sampled use exceeds12GiB of16GiB. This is headroom monitoring,
not a hard reservation or guarantee against desktop lag.
