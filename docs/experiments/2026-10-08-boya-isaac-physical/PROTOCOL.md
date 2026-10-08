# Original-grasp Isaac physical interface diagnostic

Read docs/experiment_state.md, original teacher-v2/grasp provenance, the official
headless and structural import follow-ups and hora_boya_port.md. New factor:
actual single-scene PhysX dynamics with the imported Boya hand and original
40x32mm/50g cylinder/physical q/velocity initial condition, rather than USD schema.

No learned controller or new reward/training. Controller boya_source_pd_hold_v1:
13 source active position inputs,18 original position motors including held wrist
and LF,4 passive mimic joints. Zero actions at20Hz; interpolate at.0005s and
preserve source kp10 fingers/kp40 wrists and .3/1Nm motor effort caps, software
command limits, separate source .05/.5 viscous damping. Passive effort contains
only source viscous damping; mimic is retained, never an independent policy action.

Original CAD collision/self-collision, fixed hand base/original root quaternion,
full world-z9.81 gravity and original unsupported object pose/velocities. Source
physical coordinates are cloned, not native solver warm-start buffers: this is an
independent new-engine initialization. PhysX TGS16position/4velocity/CCD, .1mm
contact offset/zero rest offset, rigid static/dynamic friction.5/restitution0.
PhysX friction/compliance/solver are not MuJoCo's3-component friction/soft equality;
no cross-engine angle/history pooling or identical-dynamics claim.

Budget: exactly1000 logged physical steps=.5s,300s maximum total process wall.
Record every step commands/efforts/q/qdot/object pose/body poses/contact matrix,
coupling residuals and actual scene/source configuration. The diagnostic stops
on nonfinite state or5mm object drift/15deg tilt; preserve the first failing frame.
No manual recovery, support, pose snapping or resets during the logged prefix.

This tests interface/mimic behavior and initial support. Contact-force reporting,
penetration and all physical acceptance gates are not yet certified on Isaac;
therefore even a full.5s run is not a physically validated benchmark rotation,
stable120s grasp or learned-policy result. Record actual limitations explicitly.
A large task/cache/PPO run is blocked until coupling, grasp and all physical gates
are verified. If transfer fails, stop dependent work and ask the user before
changing size/grasp, observation, engine, mechanics or criteria.
