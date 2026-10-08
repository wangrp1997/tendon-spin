# Learner continuation and logging

The user approved a first10M-action budget,then requested a video review before
starting it. Only the short65536-actionpreview is currently authorized to execute
before that review. Do not launch the10Mcontinuation from these notes automatically.

`scripts/train_boya_hora.py --resume <paired_checkpoint> --total-actions 10000000`
continues the checkpoint's learner to a cumulative target (not10Madditionalactions).
With1024envs/horizon8 this rounds to10002432actions/1221updates. If the preview final
checkpoint is used,its65536actions are already included. Use the same1024env/task/
cache/PPO configuration;the task/source contract rejects silent incompatibilities.
Old weights-only pilot/throughput checkpoints are not accepted as full resume states.

The same file contains model,observation AND value normalizers,Adam moments,LR/
adaptive scheduler,RNG,and completed action/update counts. It is atomically replaced;
`latest_checkpoint.json` points at a complete file with itsSHA256. On resume the
simulation starts new episodes from the declared28-statecache;there is no claim of
exact simulator replay. Partial rollouts are discarded. Diagnostic weights saved on
an interrupted rollout are separate from the last complete resumable checkpoint.

Use `--trace-mode summary` for compact per-update statistics and TensorBoard,with
bounded episode data in RAM. Physics/gate computations remain active. Full standalone
evaluation retains per-step evidence. `--wall-s 0 --max-gpu-memory-mib 0` disables the
optional training wall limit and GPU polling/cutoff;do not use the guarded launcher
for the user-declared1024route. Keep headless/nice+10/4threads as before.

Focused validation: `PYTHONPATH=. /home/rw/miniconda3/envs/env_isaaclab/bin/python
-m unittest tests.test_hora_resume -v` passed1test in0.555s onCPU,withoutrobotphysics.
It trains originalHoraPPO on a tiny deterministic test environment,compares the next
update between preserved learner state and a loaded checkpoint after the same reset,
and requires exact equality of weights,both normalizers,Adam state,LR/actioncounts.
Mismatched contracts are refused. No simulated manipulation result comes from this test.
