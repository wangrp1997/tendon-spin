# A/B through30M: first complete30s rotation signal in B

2026-10-11 14:42Shanghai read: both chains continue normally toward40M (A33,972,224/B33,054,720actions), without execution errors. Existing completed20M/30M stages all contain training/evaluation/offline analysis; no new episode or code change is introduced by this record. [Fixed continuation protocol](PROTOCOL.md).

Each entry is one frozen checkpoint/own-normalizer/originalgrasp44 episode, requested30s, zero resets/switches/learning and zero initial-state differences. Angles are signed moving-cylinder-axis valid-prefix metrics. All early stops below are object below Boya manipulation-region lower bound; B30M ends normally at the requested30s time limit.

| Arm | Requested cumulative actions | Actual cumulative actions | Actual stop s | Valid s | Signed net deg | Peak deg | Backward deg |20–30s net deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 10,000,000 | 10,002,432 | 2.5500 | 2.5495 | +149.587268 | 156.025928 | 6.731557 | unobserved |
| A | 20,000,000 | 20,004,864 | 1.1500 | 1.1495 | +43.045273 | 49.158806 | 6.133280 | unobserved |
| A | 30,000,000 | 30,007,296 | 1.8000 | 1.7995 | +14.628842 | 40.716825 | 27.184674 | unobserved |
| B | 10,000,000 | 10,002,432 | 8.9500 | 8.9495 | +47.259685 | 302.394167 | 281.898073 | unobserved |
| B | 20,000,000 | 20,004,864 | 22.7000 | 22.6995 | +649.225988 | 654.896947 | 46.169490 | unobserved |
| B | 30,000,000 | 30,007,296 | 30.0000 | 30.0000 | +994.812913 | 994.812913 | 12.423861 | +343.901530 |

B30M completes30s, net+994.812913deg, peak994.812913deg, backward12.423861deg and final20–30s net+343.901530deg. It passes all three predeclared preliminary finite-motion conditions: full30s, positive whole-window signed net and positive last10s net. It improves the observed20M result (valid22.6995s/net+649.225988deg/backward46.169490deg, below-region stop).

A30M drops at1.8s (valid1.7995s/net+14.628842deg/peak40.716825deg/backward27.184674deg); its20–30s metric is unobserved. B is better in this matched-budget pair of executed episodes. One seed and one state per checkpoint support this observed trend; they do not establish statistical superiority, indefinite rotation, original Hora reproduction or hardware readiness. Continue the unchanged authorized40/50M schedule. Preserve B30M and all earlier checkpoints even if later performance changes.

B30M checkpoint SHA256 fe881d605f8fc898144aab6a0ff2df8194650bbb5e06840e407f0e9ecff5864f; A30M a1c6a180a3928f866e28e5d4bab7bcce788a4e5f7b16b021d8f0a8a652a6d759. Each contains30,007,296actions/3663updates. B's existing independent SciPy pose analysis agrees: net+994.812657deg and20–30s+343.901505deg; primary scores remain unchanged.

[Small archived results and provenance](data/manifest_through30m.json) include20/30M evaluation/stage records, both30M analyses and per-arm CSVs. Full traces, weights, TensorBoard and update history remain local. Source code and both pinned protocols are unchanged. No added tests, GPU previews, physics rollouts, coefficient scans or task changes.
