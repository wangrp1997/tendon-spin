"""Direct reuse of reference networks, with a declared Boya channel adapter.

Sources:
- HaozhiQi/hora, tag v0.0.1, hora/algo/models/models.py and running_mean_std.py
  https://github.com/HaozhiQi/hora (MIT, copyright 2022 Haozhi Qi).
  Paper: In-Hand Object Rotation via Rapid Motor Adaptation, arXiv:2210.04887.
- wangrp1997/sharpa-rl-lab, commit 5accf024d376685eaa17da7aa4614498217eab4d,
  rl_isaaclab/algo/models/models.py; retains Hora's MIT header and Sharpa notices.

Reuse: load the pinned ActorCritic and RunningMeanStd implementations directly.
Boya adaptation: 13 actions, 26 measured-position/target channels per frame;
use Sharpa's parameterized version of the same 30-frame TCN for variable width.
This module is a network port, not a complete Hora/Sharpa/AnyRotate reproduction.
It is not used by teacher-v2, whose frozen source remains separate.
"""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'hora_models': 'third_party/hora/hora/algo/models/models.py',
    'hora_normalizer': 'third_party/hora/hora/algo/models/running_mean_std.py',
    'sharpa_models': 'third_party/sharpa/rl_isaaclab/algo/models/models.py',
}


@lru_cache(maxsize=3)
def reference_module(name):
    relative = SOURCES[name]
    records = json.loads((ROOT / 'references/source_manifest.json').read_text())['files']
    expected = next(r['sha256'] for r in records if r['path'] == relative)
    path = ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Reference source identity changed: ' + relative)
    spec = importlib.util.spec_from_file_location('_tendonspin_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(*, frame_features=26, actions=13, privileged_features=9, student=False):
    """Original Hora actor/critic/embedding; width-adjusted original TCN."""
    if frame_features <= 0 or actions <= 0 or privileged_features <= 0:
        raise ValueError('Positive channel counts required')
    model = reference_module('hora_models').ActorCritic(dict(
        actor_units=[512, 256, 128], priv_mlp_units=[256, 128, 8],
        actions_num=actions, input_shape=(3 * frame_features,),
        priv_info=True, proprio_adapt=student, priv_info_dim=privileged_features,
    ))
    if student:
        model.adapt_tconv = reference_module('sharpa_models').ProprioAdaptTConv(frame_features)
    return model


def build_normalizers(frame_features=26):
    """Use Hora's observation and history normalization without reimplementation."""
    cls = reference_module('hora_normalizer').RunningMeanStd
    return cls((3 * frame_features,)), cls((30, frame_features))


def student_from_teacher(teacher, frame_features=26, actions=13, privileged_features=9):
    """Hora stage two: copy/freeze the teacher; optimize only the history TCN."""
    student = build_model(frame_features=frame_features, actions=actions,
                          privileged_features=privileged_features, student=True)
    missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)
    if unexpected or not missing or any(not k.startswith('adapt_tconv.') for k in missing):
        raise ValueError('Teacher and declared student architecture do not match')
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith('adapt_tconv.'))
    return student


def adaptation_loss(student, observations):
    """Exact latent MSE used by Hora's padapt.py (teacher target detached)."""
    _, _, _, prediction, target = student._actor_critic(observations)
    return ((prediction - target.detach()) ** 2).mean()
