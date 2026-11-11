"""Exercise shared integration and conservative clearance without starting Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def methods():
    path = ROOT/'tendonspin/physics/bilateral.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = {'step', 'cross_clearance', 'canonical_state', 'world_force'}
    ns = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(path), 'exec'), ns)
    return ns


class BilateralPhysicsTests(unittest.TestCase):
    def test_both_controls_precede_one_shared_physics_step(self):
        events=[]
        def side(name):
            return SimpleNamespace(adapter=SimpleNamespace(apply=lambda:events.append(name+'_control')),
                object=SimpleNamespace(instantaneous_wrench_composer=SimpleNamespace(
                    set_forces_and_torques_index=lambda **kw:events.append(name+'_force')),
                    write_data_to_sim=lambda:events.append(name+'_write')),
                contact=SimpleNamespace(update=lambda *a,**kw:events.append(name+'_contact')))
        p=SimpleNamespace(sides={'left':side('left'),'right':side('right')},dt=.0005,physics_steps=0,
            world_force=lambda side,force:force,sim=SimpleNamespace(step=lambda **kw:events.append('step')),
            scene=SimpleNamespace(update=lambda dt:events.append('scene_update')))
        methods()['step'](p,torch.zeros(1,3))
        self.assertEqual(events.count('step'),1)
        self.assertLess(events.index('left_write'),events.index('step'))
        self.assertLess(events.index('right_write'),events.index('step'))
        self.assertGreater(events.index('scene_update'),events.index('step'))
        self.assertEqual(p.physics_steps,1)

    def test_placement_inverse_preserves_task_position(self):
        yaw=Rotation.from_euler('z',180,degrees=True)
        pivot=np.array([-.36,-.02,.1]);target=np.array([-.12,0,.1])
        actual=np.array([-.35,-.025,.11])
        state=torch.zeros((1,13),dtype=torch.float64)
        state[0,:3]=torch.tensor(target+yaw.apply(actual-pivot))
        p=SimpleNamespace(layout={'left':(pivot,target,yaw)},device='cpu')
        result=methods()['canonical_state'](p,'left',state)
        np.testing.assert_allclose(result[0,:3],actual,atol=1e-12)
        self.assertFalse(torch.equal(result,state))

    def test_separating_gap_detects_body_or_object_overlap(self):
        corners=torch.tensor([[[-.01,-.01,-.01],[.01,.01,.01]]])
        p=SimpleNamespace(geometry={'diameter_m':.02,'height_m':.02},
            sides={side:SimpleNamespace(bounds_corners=corners) for side in ('left','right')})
        measured={}
        for side,x in (('left',-.05),('right',.05)):
            body=torch.tensor([[[x,0,0,0,0,0,1.]]])
            obj=torch.zeros(1,13);obj[0,0]=x
            measured[side]={'body_pose':body,'object_state':obj}
        clearance=methods()['cross_clearance']
        self.assertGreater(clearance(p,measured),.07)
        measured['right']['body_pose'][0,0,0]=-.04
        self.assertLess(clearance(p,measured),0)
        measured['right']['body_pose'][0,0,0]=.05
        measured['right']['object_state'][0,0]=-.04
        self.assertLess(clearance(p,measured),0)
