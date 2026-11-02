"""Frozen C action head plus a wider sensor-history adaptation encoder.

The actor remains 96+8 wide. New J1/tactile channels only enter the TCN.
Only training-side TeacherLabels can access privileged object information.
"""
from copy import deepcopy
from pathlib import Path
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from tendonspin.baselines.reference_models import build_model, reference_module
from .observations import FRAME_DIM, HISTORY, SCHEMA


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TeacherLabels(nn.Module):
    def __init__(self, checkpoint, device='cpu'):
        super().__init__()
        self.checkpoint_sha256 = sha256(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved.get('schema') != 'tendonspin-hora-resume-v1':
            raise ValueError('Require a paired C teacher checkpoint')
        self.contract = saved['contract']
        if self.contract.get('arm') != 'C':
            raise ValueError('Require the selected C teacher')
        self.model = build_model()
        self.model.load_state_dict(saved['model'], strict=True)
        self.normalizer = reference_module('hora_normalizer').RunningMeanStd((96,))
        self.normalizer.load_state_dict(saved['running_mean_std'], strict=True)
        for module in (self.model, self.normalizer):
            if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
                raise ValueError('Teacher checkpoint contains nonfinite tensors')
        self.to(device).requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, observation):
        latent = torch.tanh(self.model.env_mlp(observation['priv_info']))
        normalized = self.normalizer(observation['obs'])
        action = self.model.mu(self.model.actor_mlp(torch.cat((normalized, latent), -1)))
        return action.clamp(-1, 1), latent


class FrozenActionHead(nn.Module):
    def __init__(self, teacher=None):
        super().__init__()
        if teacher is None:
            self.normalizer = reference_module('hora_normalizer').RunningMeanStd((96,))
            self.actor = reference_module('hora_models').MLP([512, 256, 128], 104)
            self.mu = nn.Linear(128, 16)
        else:
            self.normalizer = deepcopy(teacher.normalizer)
            self.actor = deepcopy(teacher.model.actor_mlp)
            self.mu = deepcopy(teacher.model.mu)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, history, latent):
        observation = history[:, -3:, :32].reshape(-1, 96).clamp(-5, 5)
        x = torch.cat((self.normalizer(observation), latent), dim=-1)
        return self.mu(self.actor(x)).clamp(-1, 1)


class TouchStudent(nn.Module):
    def __init__(self, teacher=None, modality='touch'):
        super().__init__()
        if modality not in ('touch', 'proprio'):
            raise ValueError('Unknown student modality')
        self.modality = modality
        self.action_head = FrozenActionHead(teacher)
        self.history_encoder = reference_module('sharpa_models').ProprioAdaptTConv(FRAME_DIM)

    def forward(self, history):
        if not isinstance(history, torch.Tensor) or history.shape[1:] != (HISTORY, FRAME_DIM):
            raise ValueError('Student accepts only its sensor-history tensor')
        if self.modality == 'proprio':
            history = history.clone()
            history[:, :, 36:56] = 0  # No tactile signal, mask or age leakage.
        latent = torch.tanh(self.history_encoder(history))
        return dict(action=self.action_head(history, latent), latent=latent)


def distillation_loss(prediction, teacher_action, teacher_latent):
    action = F.mse_loss(prediction['action'], teacher_action.detach())
    latent = F.mse_loss(prediction['latent'], teacher_latent.detach())
    return dict(total=action+latent, action=action, latent=latent)


def save_student(path, model, observation_contract, teacher_sha256, progress):
    path = Path(path)
    payload = dict(schema=SCHEMA, model=model.state_dict(), modality=model.modality,
                   observation_contract=observation_contract, teacher_sha256=teacher_sha256,
                   progress=progress, hardware_validated=False)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_student(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('schema') != SCHEMA:
        raise ValueError('Incompatible student checkpoint; legacy/botyard weights are not accepted')
    model = TouchStudent(modality=saved['modality']).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False).eval()
    return model, saved
