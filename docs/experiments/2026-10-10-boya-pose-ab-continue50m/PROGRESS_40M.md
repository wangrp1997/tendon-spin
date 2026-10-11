# A/B40M: B retains and improves the30s finite-motion signal

2026-10-11 19:24Shanghai read: A41,934,848actions/5119updates; B40,484,864/4942. Both chains have completed40M training/evaluation/analysis and normally train toward the final50M cumulative budget, with no reported errors. Same [fixed protocol](PROTOCOL.md), algorithms, rewards, physics, task, cache and observations.

Each40M checkpoint has40,001,536actions/4883updates. Each result below is one separate frozen own-checkpoint/normalizer/originalgrasp44 episode requested30s. Zero initial-state errors, evaluation resets, policy switches or training actions. Training/evaluation/analysis exit0. No new rollout or checking campaign was added for this record.

| Arm | Actual stop s | Valid s | Signed net deg | Peak deg | Backward deg |20–30s net deg | Actual stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 1.5000 | 1.4995 | +9.579396 | 57.906078 | 50.935933 | unobserved | object below Boya manipulation region |
| B | 30.0000 | 30.0000 | +1155.299400 | 1155.299400 | 11.469393 | +384.291421 | time limit |

B40M again meets the predeclared finite-motion signal: full30s, positive net and positive final10s net. Relative to [B30M](PROGRESS_30M.md), net increases+994.812913->+1155.299400deg, final10s+343.901530->+384.291421deg and backward motion decreases12.423861->11.469393deg. It stops normally at the requested30s time limit. A drops below the manipulation region at1.5s, with valid1.4995s/net+9.579396deg/peak57.906078deg/backward50.935933deg.

B is better in this matched40M pair and has met the30s criterion at two independently executed training nodes. These are separate episodes and checkpoints, not a continuous60s result. One seed/state per node supports an observed learning trend, not statistical superiority, indefinite rotation, original Hora reproduction or hardware readiness. Preserve both B30M/B40M and all earlier weights. Continue the existing50M budget; no extension or algorithm/task change.

B40M checkpoint SHA256 2edabd00a4232b02398e3bb2d45a08fc931dc6949aca16c31ba36865101330fc; A40M 6d14c38bf8fcb2de46691a21afb9e9ef8b9c906b613fb92d2c24630c288067d6. [Small archived results and provenance](data/manifest_through40m.json) retain per-arm CSVs,40M evaluation/stage/analysis and original source hashes. Full traces, TensorBoard, weights and update history remain local. Both pinned protocols and all runtime code remain unchanged.

At the19:24 read, current segment rates A496.98/B483.58actions/s predict pure training finishes23:55/00:52Shanghai (Oct11/12), before final evaluation/analysis. Tentative delivery of both final50M results is Oct12 01:00–02:00, subject to rates and evaluation duration; this is not a learning guarantee.
