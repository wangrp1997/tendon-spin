# White preview correction

The user reported an all-white video. Inspection of the saved final frame confirms
the original preview contains only background and labels; the advertised visual
preview is not usable. Preserve that file and original physics result as history.

Installed Isaac Lab3 IsaacRtxRenderer defaults enable_scene_partitioning=True and
prepare_stage assigns env_0 geometry to its scene partition. Our spectator camera
is /World/Camera, outside /World/envs/env_0, and has no matching partition token.
The prior nonparallel single-scene hold does not use partitioned environment roots
and its same-camera view shows the hand. Renderer documentation explicitly describes
partitionless cameras excluding partitioned content unless spectator mode is enabled.

Correction: use the documented per-process override
ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 for this SINGLE-env preview.
No installed package/global setting, training/physics source, checkpoint, grasp,
contact threshold, camera pose or controller changes. Existing evaluator still
checks all source hashes against the original training record. Wrapper records the
render override, command, source/checkpoint identities and process result.

One new frozen original-state episode is authorized by the requested video fix:
same65536-action checkpoint,seed43,5swindow/firstphysicalfailure stop,full step logs,
nativeRTX camera. Keep it separate from the white-video execution even if numerical
results match. Inspect actual returned video frames before re-delivery, as required
to fix this reported defect. Long10M training remains pending user video review.

Execution completed; correction results below supersede this initial protocol.


The partition override rerecord completed16.38s/exit0,with the same .356sactual,
.3555svalid,+24.05925deg,drift>5mm result. Actual final image was viewed and shows
hand and object;white-video fault is corrected. User then requests Sharpa-style
background/floor. Sharpa5accf024::_setup_scene has GroundPlane and DomeLight.

A separate nativeRTX replay now adds a slate checker reference floor,dome/keylight,
orange cylinder and a thin painted orientation marker. Render measured world-link
poses from the completed visible_v2episode;no controller or physics stepping,new
training0. No mesh/pose interpolation;floor has no CollisionAPI and is purelyvisual.
Keep old videos and trajectory files. Label archived motion source and nativeRTX
renderer in video/manifest. Style changes do not revise physical rotation metrics.


Completed: visible_v2 rerecord has same65536-action checkpoint,.3560s actual,
.3555s valid,+24.05925deg net,drift>5mmstop,0reset/switch. Source hash checks passed;
no controller/physics changes. This remains a separate execution fromwhite_v1.

Studio_v3 nativeIsaacRTX replay completed with no new physics/controller/training.
Initial and terminal images were actually viewed: hand,cylinder,checker floor and
shadows visible. Floor is a Mesh with NO CollisionAPI; body world matrices are set
from recorded poses. Orange material and a cap marker are visual decoration only;
marker visibility depends on viewpoint. No pose interpolation or claimed extra motion.

Current deliverable:outputs/boya_hora1024_studio_v3/isaac_native.mp4.
[Rendered-frame provenance](studio_render.json),[visible rerecord](visible_evaluation_result.json),
[render visibility override](visible_preview_manifest.json),[correction summary](video_correction.json).
Original white video and original execution are retained. Long10M remains unstarted.
