# User-requested512-environment throughput trial

Read experiment_state,128-env protocol/README and latest completed result plus its
frozen evaluation.128env finished65536actions in635.98s process,104.283actions/s
training-loop throughput,max sampled3838MiB GPU memory. Final original-state frozen
policy valid prefix .484s/+34.5814deg,stopped at drift>5mm;separate control result,
not evidence of sustained rotation. No old jobs remain during this trial.

User: “那你先尝试下汇报给我”,referring to512envs with desktop responsiveness.
New factor:512 parallel environments.8updates×horizon8×512=32768requested actions,
minibatch512/5epochs,seed43 fresh policy/normalizers/optimizer,28-state grasp cache.
Compare actual actions/time to first32updates of the existing128-env run (same
32768-action budget) and its full-run104.283actions/s. Different rollout batch
sizes/episode realizations affect overhead; one bounded run is a throughput sample,
not a statistical algorithm comparison. No 1024 scan is authorized in this stage.

All physics/task/privilege/reward/observations/action rates unchanged from128-env:
40×32mm/50g,original orientation/fullgravity,16finger actions,heldwrists,4passive
mimics,v3 nominal uncalibrated armature/external clippedPD,TGS16/4,selfcollisions/
28excludes,dt.0005s/control20Hz,.35rad/s position rate. Original center/axis gates:
5mm/15deg/12N/100rad/s/.05radmimic/finite. No four-finger rule. No DR/student yet.

Existing Isaac environment,headless/no cameras,nice+10,Torch/OMP/MKL4threads.
Training collection wall cap240s;external hard cap300s. Sample whole-device memory,
GPU utilization and child RSS once per second. If sampled GPU memory>10240MiB,
request stop at an update boundary;hard cap still applies. This leaves about6GiB
nominal VRAM headroom but does not guarantee desktop responsiveness or zero spikes.
No global GPU power/clock/settings changes. No competing training/evaluation task.

Preserve all per-physics-step traces,source/config/cache identities,normalizers,
updated checkpoints and actual sample count. Save checkpoint at4and8updates.
Unfinished-control trace export is made tolerant of absent reward fields with
explicit recorded masks;normal complete control records are unchanged. This fixes
the previously observed interrupt-only exporter failure and changes no physics/PPO.
No extra simulation test for that diagnostics-only edit.

Report startup/time-inclusive and training-loop actions/s,peak observed GPU memory,
GPU utilization/RSS samples,stop reason and whether512 improved practical throughput.
This trial does not evaluate a frozen policy or establish improved rotation/SOTA.
Retain existing128env checkpoint and evaluation; no training-angle/episode pooling.
