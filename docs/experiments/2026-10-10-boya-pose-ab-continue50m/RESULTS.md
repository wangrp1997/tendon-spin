# A/B50M completed: final checkpoint regression; preserve B40M

Both fixed-budget chains completed and stopped without execution errors. A finished2026-10-12 00:22:19, B00:54:32 (Asia/Shanghai). Each executed50,003,968 cumulative actions/6104 PPO updates, including40,001,536 new actions/4883 updates after its own10M parent. At08:55:56 the recorded outer and latest child processes were no longer alive. All four continuation train/evaluation/analysis stages per arm completed normally with exit0; training stopped at the update budget. No follow-on training or evaluation was launched.

Reuse the [fixed continuation protocol](PROTOCOL.md): A=hora_pose_delta; B=the same plus one raw terminal failure cost16 at existing workspace codes9/10, timeoutcost0. Each arm retained its OWN full learner/normalizers/Adam/LR/scheduler/RNG/lineage. Later segment resumes reset simulation from the same28-state cache; training is not uninterrupted physics. Object/grasp/orientation,16 active actions/held wrists/passive mimics, privileged observations, original Isaac engine/collision/task/timing/PPO remain exactly as declared. Both protocols and runtime code are unchanged.

Exactly eight NEW frozen episodes were executed at20/30/40/50M, plus two existing10M result rows reused below. Every row is one independent frozen final-policy/own-normalizer/originalgrasp44/requested30s episode, zero resets/switches/learning and zero initial-state differences. Signed net/peak/backward are valid-prefix moving-cylinder-axis angles. All incomplete rows stop below the Boya manipulation-region lower bound; B30M and B40M stop normally at the requested30s time limit. No angles are combined across episodes.

| Arm | Requested cumulative actions | Actual cumulative actions | Actual stop s | Valid s | Signed net deg | Peak deg | Backward deg |20–30s net deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 10,000,000 | 10,002,432 | 2.5500 | 2.5495 | +149.587268 | 156.025928 | 6.731557 | unobserved |
| A | 20,000,000 | 20,004,864 | 1.1500 | 1.1495 | +43.045273 | 49.158806 | 6.133280 | unobserved |
| A | 30,000,000 | 30,007,296 | 1.8000 | 1.7995 | +14.628842 | 40.716825 | 27.184674 | unobserved |
| A | 40,000,000 | 40,001,536 | 1.5000 | 1.4995 | +9.579396 | 57.906078 | 50.935933 | unobserved |
| A | 50,000,000 | 50,003,968 | 3.3500 | 3.3495 | +124.265787 | 124.265787 | 15.046098 | unobserved |
| B | 10,000,000 | 10,002,432 | 8.9500 | 8.9495 | +47.259685 | 302.394167 | 281.898073 | unobserved |
| B | 20,000,000 | 20,004,864 | 22.7000 | 22.6995 | +649.225988 | 654.896947 | 46.169490 | unobserved |
| B | 30,000,000 | 30,007,296 | 30.0000 | 30.0000 | +994.812913 | 994.812913 | 12.423861 | +343.901530 |
| B | 40,000,000 | 40,001,536 | 30.0000 | 30.0000 | +1155.299400 | 1155.299400 | 11.469393 | +384.291421 |
| B | 50,000,000 | 50,003,968 | 7.2500 | 7.2495 | +258.927618 | 258.927618 | 59.004045 | unobserved |

Final50M: A falls at3.35s (valid3.3495s/net+124.265787deg/peak124.265787deg/backward15.046098deg); B falls at7.25s (valid7.2495s/net+258.927618deg/peak258.927618deg/backward59.004045deg). Neither final checkpoint completes30s, so neither satisfies the preliminary full30s/positive net/positive last10s signal. Both final20–30s metrics remain null.

B30M and B40M met that signal in their respective executed episodes. Among evaluated nodes, B40M remains the best observed candidate: full30s/net+1155.299400deg/peak1155.299400deg/backward11.469393deg/final10s+384.291421deg, normal30s stop. B50M regresses relative to this recorded episode despite more training; the latest checkpoint is not the strongest observed one. B is better than A in the matched20/30/40/50M episodes, while the50M final endpoint is itself below the full-window criterion.

This shows useful finite-motion behavior was learned at intermediate B checkpoints and was not retained in the final recorded50M episode. It does not identify the cause as forgetting, PPO instability or any other mechanism. One seed and one original state per checkpoint establish an observed trend, not independent statistical validation, indefinite rotation, original-baseline reproduction, SOTA or hardware readiness. Selection of B40M uses these same milestone metrics; preserve it as a candidate, not an independently confirmed winner.

Preserve all parents and intermediate/final checkpoints, especially B30M and B40M. B40M SHA256 2edabd00a4232b02398e3bb2d45a08fc931dc6949aca16c31ba36865101330fc. Final A SHA256 cff1976562394b666039d34e09fbc784344ba00216c9617dec118e229873f4a8; final B 7fdcedcebbcaa16ff492c2c2fccbd5a5e61f410bef3a161b866f0453f959023f. No weight deletion, overwrite, coefficient scan, criterion change, retry or budget extension occurred.

[Final small evidence and original-source hashes](data/manifest_final.json) include final status, per-arm CSVs,50M stages/evaluations/analyses and training summaries. Prior [30M](PROGRESS_30M.md) and [40M](PROGRESS_40M.md) evidence remains. Full traces, weights, update history and TensorBoard remain local under ignored outputs/. Final recording only; no extra tests, physics episodes, previews or audit campaign.
