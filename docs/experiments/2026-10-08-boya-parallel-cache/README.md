# First 16-input Isaac parallel cache executed

64 environments, 8 batches, 8000 parallel physics steps, 82.539 s wall.
504 random candidate episodes at ±.25rad: **1 accepted** (0.1984%).
8 independent nominal-anchor episodes all completed .5s and passed screening.
These anchors are excluded from random cache size. No resets within each candidate,
no training actions, no rotation result. Stop: requested batch budget.

First rejection counts: link normal >12N:400; drift >5mm:100; tilt >15deg:3.
No candidate first rejected for nonfinite state, excessive joint speed or mimic error.
Thus nominal holding survived batching; this does not establish diverse reset coverage.
One surviving grasp is enough for connecting the PPO interfaces, not a paper-scale cache.

[Protocol](PROTOCOL.md) declares original40×32mm/50g/grasp44/full gravity,
16 finger inputs and held wrists, v3 dynamics, all criteria and Hora sampling differences.
[Result](result.json) includes configuration/source identity and per-batch evidence.
Raw initial samples, each physics step and failure indices: outputs/boya_parallel_cache_v1/.
Cache: outputs/boya_parallel_cache_v1/grasp_cache.npz (one state).
Observed GPU memory during collection approximately3802MiB of16303MiB;
Torch's reported8.84MB excludes PhysX/Kit and is not total GPU memory.
No full penetration certification or hardware result; benchmark_validated=false.

Execution itself is the required integration validation; no extra test/audit pass.
Local milestone commit only, remote not created. Next: original Hora PPO nominal pilot
using this explicitly limited cache, then resolve cache coverage before large training.

随后接通原Hora PPO并完成16次更新，见[训练阶段](../2026-10-08-boya-hora-pilot/README.md)。
已有缓存数据诊断：400个力门槛首拒中首失败步的25/50/75分位均为第1步；
26个随机候选仅在末.1s满足各项门槛。后者只说明初始化暂态影响筛选，
不修改历史1/504通过数，也不证明这26个都能从缓存状态稳定重启。
更改为初始化/正式保持两阶段筛选已询问用户，尚未执行。

最新后续：用户已批准并完成[独立两阶段v2](../2026-10-08-boya-settled-cache/README.md)，
28/504通过，保留本页旧1/504全程标准及所有原始数据。
