# Small-finger side contact

The bounded CAD proposal reduced orientation departure to7.73deg but failed its geometric screen; retained under `outputs/boya_fivefinger_side_v1`. After the user accepted gaps, ONE physical0.5s initialization+30s hold passed old rules, but only four fingers touched (actual orientation departure7.51deg). No random cache was generated.

User then requested shifting toward LF and curling LF. ONE new candidate: same48x60mm/50g, cylinder translated0.15mm toward LF, LFJ3 and LFJ2 each+1.72deg, other motor posture/orientation unchanged before the shared fixed preload.

ONE0.5s initialization+30s formal native hold passed. Each finger and all-five contact fraction100%; max formal drift0.049620mm, axis tilt0.393813deg; mean LF normal0.100682N (diagnostic, no bearing-force gate). Valid endpoint+0.013094deg/peak0.021782/backward0.021034; normal time-limit stop. This is holding, not learned rotation.

Optional profile `fivefinger_sharpa_side_v2` contains the settled native handoff state and model. Original defaults/assets/cache/weights retained. No independent reload, Isaac transfer, bulk cache or training. Small evidence is `assets/grasp/fivefinger_sharpa_side_v2/validation.json`; raw traces and top-view image stay local under `outputs/boya_fivefinger_side_shift_v2/`.

Network reference consulted: [DexGraspNet contact candidates](https://github.com/PKU-EPIC/DexGraspNet/blob/bd1b13d7248729af117e1d46aaa6266b147a3c7b/grasp_generation/utils/hand_model.py), [energy terms](https://github.com/PKU-EPIC/DexGraspNet/blob/bd1b13d7248729af117e1d46aaa6266b147a3c7b/grasp_generation/utils/energy.py). Upstream HEAD observed at this revision; no DexGraspNet code imported or experiment executed.
