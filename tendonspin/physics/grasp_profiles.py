# Sources: TendonSpin native.py original-state restoration at6089025;
# botyard-inhand204d197a9606fb7266e884f3b2e6110195be01cb saved integration state.
# New optional named profiles; no existing evaluator/controller imports this module.
"""Select an original or separately validated MuJoCo initial grasp by parameter."""
import hashlib
import json
from pathlib import Path


def grasp_profile(root, name='boya_original'):
    root=Path(root)
    catalog=json.loads((root/'assets/grasp/profiles.json').read_text())
    if name not in catalog['profiles']:
        raise ValueError(f'Unknown grasp profile {name!r}; choose {tuple(catalog["profiles"])}')
    return dict(catalog['profiles'][name],name=name)


def load_native_grasp(root, name='boya_original'):
    """Return model,data,metadata with exact full state. Does not integrate physics.

    Default is the preserved original grasp44. Other engines must use the named
    joints/object/commands as a proposal and validate their own transfer.
    """
    import mujoco
    import numpy as np
    root=Path(root);spec=grasp_profile(root,name)
    engine=Path(mujoco.__file__).parent/f'libmujoco.so.{mujoco.__version__}'
    if mujoco.__version__!=spec['mujoco_version'] or hashlib.sha256(engine.read_bytes()).hexdigest()!=spec['engine_sha256']:
        raise RuntimeError('Grasp profile requires its declared native MuJoCo engine')
    scene=root/spec['scene'];state_path=root/spec['state']
    if hashlib.sha256(scene.read_bytes()).hexdigest()!=spec['scene_sha256']:
        raise RuntimeError('Grasp profile scene identity changed')
    if spec['state_format']=='npz':
        with np.load(state_path,allow_pickle=False) as saved:
            if str(saved['model_sha256'])!=spec['state_source_scene_sha256']:
                raise RuntimeError('Original saved-state identity changed')
            signature=int(saved['signature']);state=saved['state'].copy()
    else:
        saved=json.loads(state_path.read_text())
        if saved['scene_sha256']!=spec['scene_sha256'] or not saved['validated']:
            raise RuntimeError('Custom state is unvalidated or belongs to another scene')
        signature=saved['signature'];state=np.array(saved['integration_state'])
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    signature=mujoco.mjtState(signature)
    if len(state)!=mujoco.mj_stateSize(model,signature):raise ValueError('State size differs from profile model')
    mujoco.mj_setState(model,data,state,signature)
    mujoco.mj_forward(model,data)
    # Populate derived kinematics/contact buffers without changing saved warm start.
    mujoco.mj_setState(model,data,state,signature)
    actual=np.empty_like(state);mujoco.mj_getState(model,data,actual,signature)
    if not np.array_equal(actual,state):raise RuntimeError('Exact grasp-state restore failed')
    if data.eq_active[model.equality('cylinder_support').id] or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
        raise RuntimeError('Exported grasp unexpectedly uses external support')
    return model,data,spec
