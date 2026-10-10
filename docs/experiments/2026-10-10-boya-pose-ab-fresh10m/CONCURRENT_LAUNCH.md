# Concurrent B authorization and execution note

2026-10-10 user reviewed A's actual memory and explicitly says to start B in
parallel. This supersedes the earlier A-first/B-pending launch status, while
retaining the shared PROTOCOL.md unchanged: A has pinned its source hash for
training/evaluation/resume. No algorithm, task, engine or budget change.

B starts fresh seed43, zero counts, own new policy/critic/normalizers/Adam/LR;
1024envs, hora_pose_delta_drop16, rawterminalcost16 only on existingworkspace
failurecodes9/10, ordinarytimeoutcost0. A remains hora_pose_delta. Both requested
10M/rounded10,002,432actions/1221updates, with1/3/5/10M frozen30s evaluations.
Full declared object/grasp/actions/observations/physics/collision/timing setup
remains the shared protocol. All historical weights and runs are preserved.

Two independent outer/segment/learner process trees share one RTX5080; neither
arm loads the other's learner. B launched2026-10-10 16:57:27Shanghai,9min28s after
A. Runtime/training/analysis/wrapper source files and shared protocol are identical
and remain frozen. Independent outputs, paired checkpoints and TensorBoard runs.
Verify the actual initial policy/critic and both normalizers are exactly equal;
paired contract differs only in reward_configuration. First B checkpoint must
show8192actions/update1/Adamstep80 and no inherited lineage/migration.

Each arm independently triggers evaluation when its own declared node completes;
there is no common wall-clock barrier. Evaluations may overlap the other arm's
training. Keep the existing1500s evaluation wall cap, fulltrace andactualstop
reporting; never count an incomplete window as30s. No change to simulationdt,
environmentcount or physical criteria to improve throughput. GPUallocation and
throughput observations are measurements only, not a watchdog or stop threshold.
No cgroup, memory quota, trainingwalllimit,automaticretry/extension or extraGPUtrial.

Update ETA from a short interval of actual concurrent training counters, rather
than A's cumulative average (which includes its earlier solo interval). Report
each training PID, actual memory, interval and rates; estimates can change with
learning/episode lengths/evaluation scheduling. Launch evidence and provenance
are pushed; weights/rawlogs/fulltraces/TensorBoard remain local.

Actual startup verification completed: Binitial policy/critic and bothnormalizers
exactly match A; sources identical; pairedcontractdiffonlyreward_configuration.
Concurrent interval16:59:47–17:01:46Shanghai gives A426.898236/B428.072820actions/s,
total854.971055. SharedGPU7279/16303MiB,bothprocesses3252MiB. At17:01:46A548,864/
B98,304actions. Initial ETA:firstmatched1Mresults17:45–18:00;both10M+evaluations
2026-10-11 00:00–01:00Shanghai,conditionalonfuturethroughput/evaluationduration.
Exact samples and forecasts:concurrent_launch_measurement.json.
