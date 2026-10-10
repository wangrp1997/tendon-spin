# Fresh A/B 10M final results

Both fixed-budget runs completed and stopped: A at2026-10-10 23:06:16, B at23:19:14 (Asia/Shanghai). Each executed10,002,432 actions/1221 PPO updates from zero. All four training/evaluation/analysis stages per arm completed with exit0; training stopped at its update budget. No extension or new experiment was started.

A uses measured-pose rotation reward; B adds the single raw terminal failure cost16. The frozen object/grasp/orientation,16 motor actions, privileged teacher observations, original4 Isaac engine, task, timing, cache and PPO are declared in [PROTOCOL.md](PROTOCOL.md). Same initial policy/normalizers were verified at launch. Later segments restored each arm's own full learner while resetting simulation from the same28-state cache; training is not one uninterrupted physical episode.

Each row below is exactly one frozen checkpoint/own-normalizer/originalgrasp44 episode, requested30s. Initial-state errors are zero; no resets, policy switches or learning occurred inside evaluation. All eight stopped because the object fell below the Boya manipulation-region lower bound. Net/peak/backward angles describe the signed moving-cylinder-axis valid prefix; no angles are summed across episodes.

| Arm | Requested training actions | Actual training actions | Actual stop s | Valid s | Signed net deg | Peak deg | Backward deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1,000,000 | 1,007,616 | 1.2500 | 1.2495 | +19.302969 | 59.347702 | 40.058890 |
| A | 3,000,000 | 3,006,464 | 1.3500 | 1.3495 | +103.011349 | 149.320902 | 46.309554 |
| A | 5,000,000 | 5,005,312 | 1.5500 | 1.5495 | +94.096571 | 101.783974 | 7.805334 |
| A | 10,000,000 | 10,002,432 | 2.5500 | 2.5495 | +149.587268 | 156.025928 | 6.731557 |
| B | 1,000,000 | 1,007,616 | 2.3000 | 2.2995 | -57.080269 | 29.256714 | 93.909558 |
| B | 3,000,000 | 3,006,464 | 1.7000 | 1.6995 | +68.697964 | 68.697964 | 18.344471 |
| B | 5,000,000 | 5,005,312 | 1.3500 | 1.3495 | +16.078711 | 27.396844 | 11.487115 |
| B | 10,000,000 | 10,002,432 | 8.9500 | 8.9495 | +47.259685 | 302.394167 | 281.898073 |

At matched10M, A retains more signed net turning (+149.58726779625277deg) with little backward motion, but falls at2.55s. Compared with its5M snapshot, valid duration rises from1.5495 to2.5495s and net from+94.096571 to+149.587268deg.

B falls later at8.95s versus1.35s at5M, but its302.394167deg peak is followed by large backward motion (281.898073deg total), leaving only+47.259685deg net. Existing offline analysis reports+205.943724deg over0–5s and−158.684082deg over5–8.9495s. Its final0.2s net−233.364393deg is a posthoc endpoint diagnostic, not a filtered score. Longer valid duration is observed in this episode; stable holding or a proven causal penalty benefit is not established.

Neither arm observed the full30s, so both fail the protocol's preliminary full-window signal. The20–30s metrics remain null. A improves short turning and B delays failure in these final snapshots, but stable sustained rotation remains unresolved. One training seed and one original state per node support a trend only, not statistical superiority, convergence, invalidity of the reward method, SOTA or hardware readiness.

The original fixed-axis reward and moving-cylinder-axis evaluation metric remain distinct: B's observed5–8.9495s fixed-axis pose mean is+1.634224rad/s while moving-axis mean is−.701243rad/s. The existing clipped rotation reward mean is+.331745. Pose-derived reward does not imply that its mean equals the primary signed net angle, particularly during large tilt. No reward or metric change was made.

[Small result and provenance files](data/manifest.json) retain all node evaluation results/stages, both final analyses and training summaries, exact checkpoint hashes and original local source hashes. Weights, full physics traces, full update history and TensorBoard stay local under ignored outputs/. All previous history is preserved. This delivery only records already completed work; no additional checking passes, tests, physics episodes or parameter changes were performed.
