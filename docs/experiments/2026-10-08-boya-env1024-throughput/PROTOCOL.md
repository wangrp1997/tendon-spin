#1024-environment throughput follow-up

User asks whether the measured350actions/s can be faster. Read current experiment
state,512-env README/result/comparison and resource summary.512 achieved349.97
training-loop actions/s at4008MiB max whole-device memory;GPU median63%/peak78%.
The old512-stage restriction against an unrequested1024scan applied to that completed
stage; this new user request authorizes a separately documented throughput follow-up.

New factor only:1024envs instead of512.4PPO updates×horizon8×1024=32768actions,
same total action budget as5128updates. Same minibatch512/5epochs,seed43 fresh
network/normalizers/optimizer,28cached starts,HORA teacher source and rewards.
Different rollout batch size is explicit;compare throughput,not policy performance.
No warm start,aggregated episode angle or claim of convergence.

All physics/observation/task criteria retained:original40×32mm50g/fullgravity and
reference orientation,16finger inputs,heldwrists,four passive1:1mimics,v3 declared
nominal armature/external clippedPD,28collision excludes,selfcollisions,TGS16/4,
physics.0005s,control20Hz/rate.35rad/s,privileged teacher96+9 observations,±.02qnoise,
previous5mm/15deg/12N/100rad/s/.05radmimic/finite gates. No four-finger criterion.
No property/size/PD randomization/student/complete baseline/hardware result yet.

Headless/no cameras,nice+10,Torch/OMP/MKL4threads. No competing simulator jobs.
Monitor whole-device GPUmemory/utilization and childRSS1Hz. Request graceful stop
above10GiB GPU use;hard process cap300s,training collection cap240s. No global GPU
power/clock or OS settings. No GUI latency claim from memory/utilization alone.
Keep all per-physics-step records,checkpoint/source/config/cache hashes. This is a
resource/speed test; no new frozen-policy evaluation is claimed. Report identical
sample-count comparison with128/512 and actual startup-inclusive time as well.
