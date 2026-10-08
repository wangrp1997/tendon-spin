# Boya Isaac physical transfer v2: bounded approved correction

Read experiment_state, physical-v1 protocol/failure_diagnosis/damping_linearization,
structural import protocol/evidence and Hora port checklist before this execution.
The user explicitly selected the quaternion/contact-filter/damping repair followed
by one single-scene verification capped at 300 seconds. No repeated launch scan.

New factors relative to boya_source_pd_hold_v1: native WXYZ is explicitly converted
at the Isaac Lab 3 XYZW boundary, including output interpretation; viscous damping
is assigned to solver-side joint friction with unchanged .05/.5 Nm s/rad; contact
filters are real rigid prim paths in the nested hierarchy and must match runtime
articulation links. Source motor PD alone supplies clipped external effort.
Controller name boya_source_pd_hold_v2; fresh original-state execution, no splice.

Unchanged: original 40x32mm/50g cylinder, grasp44 root/joint/object pose and velocity,
CAD convex-hull/self-collision, full -z 9.81 gravity, four original passive mimic
relations (no independent actions), 13 zero position actions at 20Hz, .35rad/s
increment interface and .5ms interpolation,18 source motors, kp10/40 and .3/1Nm
caps, fixed base/wrist/LF motor hold. PhysX TGS16/4, .1mm contact offset, zero rest
and .5/.5 friction/zero restitution. CCD requested but GPU runtime may disable it;
record warning/effective setting without claiming identical MuJoCo physics.

Before logged stepping, restore physical state and refresh FK without integration.
Save actual body poses/q/qdot/object/commands. Verify each body and object within
.05mm/.05deg of source, joint state/control within 1e-6 and solver damping readback
within1e-6. These are numerical initialization identity tolerances, not relaxed
physical task gates. Any mismatch blocks logged execution and is recorded.

Exactly1000 requested physics steps=.5s, stop on first nonfinite state/force/effort,
object drift>5mm or axis tilt>15deg. Save every zero action, command, pre-action
q/qdot, requested source motor effort, forwarded actuator effort and PhysX
actuation-force readback (all exclude total passive/contact reaction), resulting
joint/body/object states, normal/friction force matrices and contact positions.
Absent-contact positions may be NaN and are not themselves physical failure.
Save coupling errors and source/protocol/runtime/asset identity. No rescue, resets
or controller switches inside the episode; reset setup is followed by exact source
physical restoration, not restoration of native solver warm-start buffers.

Completing .5s only diagnoses local transfer/holding. No rotation benchmark score,
training, stable120s result or hardware robustness claim. Contact/support/penetration
and remaining physical gates still require certification. Preserve v1 failure and
correction explicitly; if physical transfer remains blocked, stop dependent large
training and ask as required by the user before changing task/backend/mechanics.
