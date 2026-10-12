# User correction: original cache admission, selectable new geometry

Latest user explicitly rejects requiring every finger to bear load and requests
the SAME rules that generated the old28-state cache. This supersedes the force /
five-loaded-pad /5s+30s holding admission sections in PROTOCOL.md for this branch.
The previous static-force-gated solve was interrupted on this correction and
retained separately;0physics steps. No previous result or protocol is rewritten.

Fixed48x60mm cylinder50g, new five-finger geometric posture/orientation, same
packaged CAD and original native patchedMuJoCo3.13.0 identity described in
PROTOCOL.md. This is an explicit MuJoCo branch for handoff, not an Isaac result.
Same16 finger motors/wrists held/4passive1:1 mimics. Original evaluation entry
points/default object/grasp/cache stay byte-for-byte unchanged. New tools expose
--object-profile /--grasp-plan /--acceptance-profile. No new training or hardware.

Planning: reuse preceding seed and conservative CAD separation fix; at most TWO
deterministic geometry-only solves(max300nfev each), stop on first geometrically
usable proposal. Keep static-force calculation as labeled initializer diagnostic,
NOT an admission gate. No minimum force per finger, no all-five-load rejection,
no force/friction/gain/size scan. Geometric proximity of5 pads is a planning goal,
not proof of actual5-finger touch. Report actual touch independently.

ONE nominal MuJoCo candidate episode first, no physical retry. Initialize q at
the plan with passive mimics matched and object at declared pose, zero velocities.
Use fixed original native position actuators(kp10fingers/40wrists, damping.05/.5,
caps.3/1Nm), with kinematic Jacobian-transpose contact preload plus native static
joint-gravity compensation, slave torque reflected to master, clipped to original
command ranges. Wrists keep original0targets. All finger motors use the SAME
preload formula; no special LF factor. Freeze all targets before physics. No weld,
floor, object force, support during initialization or hold; original floor remains
below the hand and any observed floor contact is recorded.

Continuous .5s initialization +.5s formal hold,dt.0005/control20Hz. Same formal
admission as old cache: every formal step finite, drift<=5mm from DECLARED plan
center, cylinder-axis tilt<=15deg from declared axis, joint speed<=100rad/s,
maximum object-to-link normal force<=12N, passive mimic error<=.05rad. At least
2 observed hand groups(TH/FF/MF/RF/LF/PALM), threshold>1e-6N, in>=90% of last.1s.
No all-five requirement or per-finger minimum load. Keep initialization violations
as diagnostics; nonfinite initialization is unrecoverable. Five-finger contact
fractions, all-five fraction and forces are reported, never relabeled as required
acceptance. Full per-step native state/contact/commands/zero-policy-actions/gates
recorded; no physical stop/validity inference from static-force optimization.

If nominal passes, save its final full integration state plus portable named
q/qvel/object/command JSON as a separately labeled cache candidate. Independently
reload it exactly ONCE for the SAME .5init+.5hold to check the handoff artifact;
report separately, no summed durations/angles. No extra5/30s trial or bulk cache
sampling in this turn. Report signed valid-prefix net/peak/backward, actual stop
and old-rule pass versus observed five-finger contact. A one-second screen is not
long-term or perturbed stability. A passing state with only2–4 contacts is admitted
by old rules, but is NOT a verified five-finger-contact solution.

Stop after nominal/reload or execution failure, no automatic retries, scans or
training. Preserve all old evidence and the interrupted stricter attempt. Deliver
parameterized code, small results/portable assets and milestonepush promptly.
