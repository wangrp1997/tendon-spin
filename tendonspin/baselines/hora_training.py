"""Load pinned Hora v0.0.1 PPO/reward with dependency-only Isaac Lab adaptations.

Reference: HaozhiQi/hora tag v0.0.1, MIT, In-Hand Object Rotation via Rapid
Motor Adaptation (arXiv:2210.04887). Vendored files and licenses stay unchanged.
Only remove experience.py's UNUSED gym import and substitute torch's installed
TensorBoard SummaryWriter for missing tensorboardX. PPO math remains upstream.
The pure reward function is extracted without importing legacy IsaacGym.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
HORA = ROOT/'third_party/hora'


def load_ppo():
    if str(HORA) not in sys.path:
        sys.path.insert(0,str(HORA))
    for name,relative in (('hora.algo.ppo.experience','hora/algo/ppo/experience.py'),
                          ('hora.algo.ppo.ppo','hora/algo/ppo/ppo.py')):
        path = HORA/relative
        tree = ast.parse(path.read_text(),filename=str(path))
        tree.body = [node for node in tree.body if not
            (isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='gym')]
        for node in tree.body:
            if isinstance(node,ast.ImportFrom) and node.module=='tensorboardX':
                node.module='torch.utils.tensorboard'
        module=types.ModuleType(name)
        module.__file__=str(path)
        module.__package__=name.rpartition('.')[0]
        sys.modules[name]=module
        exec(compile(tree,str(path),'exec'),module.__dict__)
    return module.PPO


def load_reward():
    import torch
    path=HORA/'hora/tasks/allegro_hand_hora.py'
    tree=ast.parse(path.read_text(),filename=str(path))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_hand_reward')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['compute_hand_reward']
