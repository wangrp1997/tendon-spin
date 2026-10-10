# Proposed Boya pose-reward plus terminal-failure cost trial

Status: PROPOSAL ONLY, not approved or launched.2026-10-10. User requests a
decision on the next plan after the pose-reward20M+1M trial regressed. No training
code/reward/checkpoint has been changed in this planning stage. Budget so far:
0newphysics,0trainingactions,0policyepisodes; existing reward arrays only.

## Decision and evidence

Recommend ONE new Boya reward adaptation: retain pose-derived rotation and add
a one-time raw reward cost16 to the existing manipulation-region terminal failures.
This is an explicit extension beyond original Hora v0.0.1, not an original baseline.
Original Hora has linear-velocity/pose/torque/work costs and drop termination, but
no separate drop cost. Termination cuts future value; its incentive depends on
the rewards available by continuing. Positive prefix reward alone proves no exploit.

Existing final policy actually drops at2.0s. Its40recorded control rewards,
INCLUDING the terminal control (already negative-.565460), have discounted
return+6.696855 at gamma.99. The original20M teacher's recorded first20s/400controls,
offline rescored with pose reward, gives-3.763435. These are two finite recorded
paths with different lengths, not a matched-state action intervention or the true
PPO value function. Parent20s endpoint is truncated without timeout critic bootstrap.
They only calibrate ONE candidate; they do not prove an optimal policy prefers dropping.

For this observed2s terminal, discount gamma^39=.675729. A raw terminal cost
15.480006 ties these two recorded sums. Select the smallest integer that reverses
their order:16, making observed drop return-4.114809. Upstream PPO retains its
.01 reward multiplier, so terminal cost in stored training-reward units is.16.
Do not tune a penalty sweep or infer that16 generalizes to other paths/drop times.
See existing_trace_return.json for exact values and source identities.

## Proposed frozen configuration and budget

Restore the preserved ORIGINAL20M paired learner, SHA256
9781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e,
20,004,864actions/2442updates; explicitly migrate to a separate reward branch.
Keep policy/critic, bothnormalizers,Adam/LR/scheduler,RNG,counters intact;
reset simulation from same28cache. Do not start from the failed21M final model.
Existing migration provenance/strict contract rules remain; extend identity to
include the terminal-cost configuration. No silent legacy relabeling.

Inherit the full fixed setup from
[the preceding protocol](../2026-10-10-boya-pose-reward-continue1m/PROTOCOL.md):
40x32mm/50g/fullgravity/grasp44, center[-.365438990352,.018340188315,.106061099740]m,
WXYZ[.175062180804,-.901031779718,.346105303444,-.194180544127];16finger motor
actions TH4+FF3+MF3+RF3+LF3,heldwrists/4passivemimics;privileged96+9/8latent teacher,
noDR/student/tactile;v3external clippedPD/A=4*D_effective*dt,CADconvex/selfcollision/
28excludes/friction.5/CCDoff,existingIsaac6.1/Lab3rc1/GPU PhysX/original_tgs16_4
scene16/255position/4/255velocity,articulation16/4,rigid16/1;
dt.0005/control20Hz/100physicssteps,targetrate.35;cacheSHA256
7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca.
Same1024envs/horizon8/minibatch512/5epochs/HoraPPO/gamma.99/tau.95.

Retain boya_workspace height/XY bounds andtwo-control confirmation, lowcontact
diagnostic only,400control/20s training timeout, existing numerical checks and
oldstrict shadow. Charge cost16 ONCE per terminated control only for existing
below-region/lateral-region failure codes9/10. Ordinary timeout gets no terminal
cost. Do not introduce contact/tilt gates, change terminations, reset optimizer,
add alive/pose rewards, modify other weights, or change engine/grasp/observations.

Requested new200,000actions, rounded204,800/25updates; proposed final cumulative
20,209,664actions/2467updates. Separate output/lineage/reward configuration.
Summarytraining/headless/TensorBoard/first-every16-finalpairedcheckpoints,
noresourcewatchdog/cgroup/memory stop/trainingwallcutoff. Rough training5min.

## Evaluation and stop decision

After normal budget completion, ONE frozenfinal/originalgrasp44/ownnormalizer
requested30s-or-firstfailure/1500swall evaluation, same exactinitialreference,
full per-physics actions/state/contact/gates plus reward components; no reset,
switch,learning,video/auxiliarytrial. Rough additional9min if30s completes.
Preserve signed valid-prefix endpoint/peak/backward,valid duration andstop reason.
Compare with reused20M30s reference:valid30s/net+30.426747deg/peak35.822706/
backward37.240682,normal30sstop. Do not repeat the reference episode.

Completing30s and net>30.426747deg is preliminary retention-plus-rotation
improvement; report backward change explicitly. Holding alone is insufficient.
Earlydrop or no net increase fails this bounded goal. Regardless of outcome,
STOP after this budget/evaluation; no automatic retries/extensions or new choices.
If useful, discuss later validation/budget; if ineffective, record failure and
discuss the route before changing another factor. No optional broad test/audit run.

This20M+204,800 branch differs in training budget from the completed20M+1,007,616
no-cost branch. It tests feasibility against the saved parent, not a causal,
matched-budget, multi-seed superiority claim. No hardware/indefinite rotation claim.
User approval is required before implementing/executing this new reward stage.
