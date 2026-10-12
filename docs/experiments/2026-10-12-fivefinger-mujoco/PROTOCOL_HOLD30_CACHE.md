# Approved30s hold, then one bounded native cache collection

User approved: first30s unsupported hold of the existing new grasp, then one
bounded batch with the old cache rules. No training. This supersedes the prior
no-long-hold/no-sampling boundary for THESE declared stages only. Keep original
evaluation defaults, all original assets/28-state Isaac cache and weights intact.
Do not write botyard-inhand or sharpa-rl-lab. Existing named optional profile is
fivefinger_sharpa_cache28_v1:48x60mm cylinder50g, new pose/posture/frozen preload.

Same native patchedMuJoCo3.13.0 SHA
c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088,
sceneSHAa2a368b6d998c2b5037cd15cb57d77e64d66b6ede2aa04067d6a66bcb27ac1e7.
Unchanged full gravity/implicitfast/.0005s/elliptic friction.5/condim3/noslip5/
native CAD collisions/28 exclusions/native kp10finger40wrist/damping.05/.5/
caps.3finger1wrist/passive1:1mimics.16 finger motors; wrists held; no LF multiplier.
Privileged simulated state/contact used for diagnosis. No learned-policy control,
support weld, object forces, controller switch, gain/size/criterion/engine scan.

Stage1: ONE exact full-state restore of the saved optional profile, frozen motor
targets, uninterrupted requested30.0s/60000steps. No initialization grace period
or reference recenter. Center/axis remain the declared geometry-plan reference.
Check initial state and every physics step: finite, drift<=5mm, axis tilt<=15deg,
speed<=100rad/s, link normal<=12N, mimic error<=.05rad. Same old cache contact
admission: >=2 observed hand groups(TH/FF/MF/RF/LF/PALM),threshold>1e-6N,in>=90%
of final.1s. Report per-finger and all-five contact throughout; no minimum load
per finger and no five-finger veto. Keep floor/support/external forces, warnings,
nonpad contacts and penetrations as diagnostics. Stop first physical/numerical
gate failure; no retries or controller changes. Save full-step native integration
state/q/v/commands/effort/contact/gates and signed valid-prefix endpoint/peak/
backward.30s holding is nominal finite holding, not rotation/robustness success.

Stage2 ONLY IF30s passes: exactly512 candidate episodes in8groups of64,seed42,
504 random plus8 repeated nominal anchors excluded from random counts. Four CPU
worker processes execute independent native episodes; no GPU or new engine.
Each random candidate uniformly perturbs all16 motor joints by±.25rad, clipped
to original motor command bounds. Distal mimics initialized to masters, hand
velocities and solver warm start zeroed. Object pose/velocity retained from the
same saved profile. Retain source preload bias(commands minus nominal joint q)
and original wrist commands; commands clipped to original bounds. No resampling
failed candidates. Each nominal anchor restores the exact full saved state.

Each episode .5s initialization +.5s formal hold, same fixed controller, no reset
between phases. Record initialization violations; any nonfinite initialization
invalidates the candidate. Explicitly check formal-start state at.5s plus every
subsequent formal state against the SAME six gates above, same fixed center/axis.
Last.1s requires>=2 hand groups in>=90% of steps. No per-finger load or all-five
admission condition. Report accepted-state finger counts/all-five fractions.
Ordinary physical rejection advances the fixed candidate list; execution/engine/
source/manual error stops dependent collection without retry. Failed episodes may
stop at first formal physical failure, whereas passed episodes must complete1s.

Save each candidate's initial/formal-start state, full native physics trace and
selection. Cache contains accepted RANDOM final states only, full integration
state plus named-joint q/v/object/commands and native model/source identity.
No target minimum count or automatic second sampling batch; report actual count
and contact distribution. Accepted states are local variations around one grasp,
not independent grasp topologies,30s validations or Isaac-ready training cache.
No zero-velocity-reload or per-candidate30s follow-on run is added.

Stop after collection. No training, extra rollout/video/test/audit campaign.
Raw traces/full cache stay ignored underoutputs; commit selectable code, small
execution summaries/provenance and push. Mark the existing profile's30s result
separately; do not overwrite its original1s evidence or saved initial state.
