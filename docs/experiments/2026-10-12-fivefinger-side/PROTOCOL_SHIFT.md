# User-directed small LF-side shift and flexion

User approves moving the object slightly toward LF and curling LF slightly, then
requests only a top-down picture. One candidate: settled side-v1 native state,
48x60mm50g cylinder, unchanged orientation/TH-FF-MF-RF/LFJ4; translate0.15mm
toward the nearest LF surface radially and add1.72deg to LFJ3 and LFJ2, keeping
LFJ1 passive1:1.16finger motors, wrists held; no LF amplitude multiplier.
Same patchedMuJoCo3.13.0/CAD/collision/gains/caps/friction/dt0.5ms. Privileged
state/contact for initialization/diagnosis only. Reuse fixed static-force preload,
no per-finger force or geometric-gap veto. No size/backend/controller substitution.
Exactly one0.5s initialization+30s formal unsupported hold under the SAME old
cache physical gates/contact-tail rule of PROTOCOL_PHYSICAL.md. Full trace,
valid-prefix endpoint/peak/backward/actual stop and each finger's touch fraction.
All-five>=90%formal is a reported task outcome, not a new cache admission gate.
Stop on first formal failure or budget; no retry/scan/cache/training/reload.
Render the saved final state from above;0render-stage physics, no audit/video.
Preserve all previous assets/states/cache/weights/defaults. Small evidence/push.
