# Required learned-policy comparisons

Target: beat the matched Boya ports of Hora, AnyRotate and SharpaWave on signed
rotation, sustained valid execution and deployment robustness. No current claim
of superiority to the original papers. Their hand/action/observations differ.

|Baseline|Available evidence/source|Boya evaluation status|Deployment comparison|
|---|---|---|---|
|Hora CoRL2022|arXiv2210.04887v1, downloaded v0.0.1 official source under third_party/hora|Reference only; needs13-action hand/task port and retraining|Proprioception-history extrinsics adaptation, no object truth at deployment|
|AnyRotate CoRL2024|arXiv2405.07391v3 and official project saved under references; project source-code availability not established|Paper-defined reproduction pending; not label a generic tactile PPO as original AnyRotate|Dense tactile feature calibration/student distillation on available Boya sensing|
|SharpaWave RL Lab|Local source snapshot5accf024d376685eaa17da7aa4614498217eab4d,22DoF/PPO/proprio-adaptation/deploy source under third_party/sharpa; original license/NOTICE retained|Reference only;13-action tendon-coupling/motor calibration port pending|Motor history + tactile observations and deployment command interface|
|Historical Boya PPO-v1|Frozen teacher pilot imported with SHA;2.506s/94.622083deg|Historical starting reference, privileged95 features|Not deployable; no student exists|
|TendonSpin teacher-v2|Bounded128-update reward diagnostic, explicit historical-weight initialization|u128:120s/18.397394deg; last30s−.200283deg; repeated rotation unresolved|Privileged teacher only; no trained student|

Sharpa is a related implementation rather than automatically a distinct published
algorithm; its YAML explicitly includes Hora privileged-embedding settings.
The local repo is a user's fork. GitHub API lookup was rate-limited, so upstream
revision equivalence remains unverified. Do not call this a reproduced original
Sharpa result or treat local non-RL adaptations as the original learned baseline.

For matched comparison fix40x32mm/50g/grasp44/world-z/.5 friction/13 kp10/fixed
wrist and LF, engine/collision configuration,120s evaluation window and the
original strict gates. Keep backward motion in signed endpoint angle, peak and
stop reason separate. Report input modalities, student/teacher role, training
samples/wall budget and actual inference/communication timing. Report all declared
checkpoints; a finite burst can rank highly while failing sustained evaluation.

After ports are verified, use at least5 training seeds and predeclared held-out
grasp/object/dynamics conditions with confidence intervals. Match training compute
budgets within a comparison tier; do not compare our tiny pilot to an undertrained
external port and claim SOTA. Original-hand paper metrics stay in a separate table.
No fixed turn-count target; a sustained120s demo must retain physical validity and
positive net progress in its final30s. This is a finite-duration requirement and
does not establish indefinite rotation.

Real deployment uses a shared measured-input tier, calibrated driver limits,
identical object/initialization/time budget and separately measured rotation truth.
Compare speed/net angle, valid duration/drop rate, tactile outages, latency,
position/axis drift and unseen material/mass effects with frozen parameters.
Teacher truth cannot be hidden inside a deployment result.

Executed teacher-v2 checkpoints and all source/replay evidence are at
[the trial record](experiments/2026-10-08-teacher-v2/README.md). Reference network
adapters are tensor-tested only and are not added as scored baseline episodes.

Original-hand paper scores, training scale and grasp generation are separately
verified in [the evidence audit](baseline_evidence.md); they are not Boya scores.
