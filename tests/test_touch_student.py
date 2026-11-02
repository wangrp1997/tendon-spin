"""Meaningful student boundaries: teacher parity, frozen weights and real inputs."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tendonspin.baselines.reference_models import build_model, reference_module
from tendonspin.students.model import (TeacherLabels, TouchStudent, distillation_loss,
                                      load_student, save_student)
from tendonspin.students.observations import (FRAME_DIM, SensorConfig, SensorHistory,
    joint_limits, sensor_contract)
from tendonspin.students.runtime import TouchStudentRuntime
from tendonspin.students.training import Replay, split_environments
from tendonspin.students.evaluation import summarize

ROOT = Path(__file__).resolve().parents[1]


class TouchStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.contract = json.loads((ROOT/'assets/config/boya_native_contract.json').read_text())
        cls.limits = joint_limits(cls.contract)

    def setUp(self):
        torch.manual_seed(51)

    def readings(self, n=1):
        return (torch.zeros(n, 20), torch.zeros(n, 16), torch.ones(n, 5),
                torch.ones(n, 5, dtype=torch.bool), torch.zeros(n, 5))

    def teacher(self, folder):
        model = build_model()
        norm = reference_module('hora_normalizer').RunningMeanStd((96,))
        path = Path(folder)/'teacher.pth'
        torch.save(dict(schema='tendonspin-hora-resume-v1', model=model.state_dict(),
                        running_mean_std=norm.state_dict(), contract=dict(arm='C')), path)
        return TeacherLabels(path)

    def test_passive_j1_is_independent_of_active_j2(self):
        processor = SensorHistory(self.limits)
        fields = self.readings()
        initial = processor.push(*fields).clone()
        fields[0][0, 16] = .2
        changed = processor.push(*fields)
        torch.testing.assert_close(initial[0, -1, :32], changed[0, -1, :32])
        self.assertNotEqual(float(initial[0, -1, 32]), float(changed[0, -1, 32]))

    def test_sensor_reset_does_not_cross_episodes_or_environments(self):
        processor = SensorHistory(self.limits, num_envs=2)
        first = self.readings(2)
        processor.push(*first)
        first[0][:] = .3
        processor.push(*first)
        untouched = processor.frames[1].clone()
        processor.reset(torch.tensor([0]))
        first[0][0] = .7
        history = processor.push(*first)
        self.assertTrue((history[0, :-1, -1] == 0).all())
        self.assertEqual(float(history[0, -1, -1]), 1)
        torch.testing.assert_close(history[0, 0, :36], history[0, -1, :36])
        torch.testing.assert_close(history[1, :-1], untouched[1:])

    def test_force_amplitude_missing_and_stale_have_distinct_features(self):
        processor = SensorHistory(self.limits, config=SensorConfig(force_alpha=1))
        fields = self.readings()
        fields[2][0, :2] = torch.tensor([1., 2.])
        history = processor.push(*fields)
        self.assertGreater(float(history[0, -1, 42]), float(history[0, -1, 41]))
        fields[2][:] = 0
        fields[3][0, 1] = False
        fields[4][0, 2] = .2
        history = processor.push(*fields)
        self.assertTrue((history[0, -1, 36:46] == 0).all())
        torch.testing.assert_close(history[0, -1, 46:49], torch.tensor([1., 0., 0.]))
        with self.assertRaises(ValueError):
            processor.push(*fields[:2], -torch.ones(1, 5), *fields[3:])

    def test_copied_head_matches_teacher_and_only_encoder_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            teacher = self.teacher(folder)
            student = TouchStudent(teacher)
            history = torch.randn(4, 30, FRAME_DIM)
            observation = dict(obs=history[:, -3:, :32].reshape(4, 96).clamp(-5, 5),
                               priv_info=torch.randn(4, 9))
            action, latent = teacher(observation)
            torch.testing.assert_close(student.action_head(history, latent), action, atol=0, rtol=0)
            before = {k: v.clone() for k, v in student.state_dict().items()}
            student.train()
            self.assertFalse(student.action_head.normalizer.training)
            loss = distillation_loss(student(history), action, latent)['total']
            optimizer = torch.optim.Adam(student.history_encoder.parameters(), lr=3e-4)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            changed = [key for key, value in student.state_dict().items()
                       if not torch.equal(value, before[key])]
            self.assertTrue(changed)
            self.assertTrue(all(key.startswith('history_encoder.') for key in changed))
            self.assertFalse(any('env_mlp' in key or 'critic' in key for key in student.state_dict()))
            with self.assertRaises(ValueError):
                student(dict(history=history, object_position=torch.ones(4, 3)))

    def test_proprio_ablation_cannot_read_tactile_values_masks_or_ages(self):
        student = TouchStudent(modality='proprio').eval()
        history = torch.randn(2, 30, FRAME_DIM)
        original = student(history)['action']
        history[:, :, 36:56] = torch.randn(2, 30, 20)*50
        torch.testing.assert_close(original, student(history)['action'], atol=0, rtol=0)

    def test_checkpoint_and_runtime_load_without_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            teacher = self.teacher(folder)
            model = TouchStudent(teacher).eval()
            path = Path(folder)/'student.pth'
            save_student(path, model, sensor_contract(self.limits), teacher.checkpoint_sha256, {})
            loaded, metadata = load_student(path)
            fields = self.readings()
            history = SensorHistory(self.limits).push(*fields)
            torch.testing.assert_close(model(history)['action'], loaded(history)['action'], atol=0, rtol=0)
            (Path(folder)/'teacher.pth').unlink()
            runtime = TouchStudentRuntime(path)
            values = dict(joint_pos_rad=fields[0][0], issued_targets_rad=fields[1][0],
                force_magnitude_n=fields[2][0], tactile_valid=fields[3][0],
                tactile_age_s=fields[4][0], timestamp_s=1.)
            action = runtime.step(**values)
            self.assertEqual(action.shape, (16,))
            with self.assertRaises(ValueError):
                runtime.step(**values)
            values['timestamp_s'] = 2.
            runtime.step(**values)
            self.assertTrue((runtime.history.frames[0, :-1, -1] == 0).all())

    def test_replay_wrap_and_scale_balanced_heldout_split(self):
        replay = Replay(3, 'cpu')
        replay.add(torch.ones(2, 30, FRAME_DIM), torch.ones(2, 16), torch.ones(2, 8))
        replay.add(torch.full((2, 30, FRAME_DIM), 2.), torch.full((2, 16), 2.), torch.full((2, 8), 2.))
        self.assertEqual(replay.size, 3)
        self.assertEqual(sorted(replay.history[:, 0, 0].tolist()), [1., 2., 2.])
        train, validation = split_environments(252)
        self.assertEqual(len(train), 216)
        self.assertEqual(len(validation), 36)
        self.assertFalse(set(train.tolist()) & set(validation.tolist()))
        self.assertEqual(torch.bincount(validation % 9).tolist(), [4]*9)

    def test_closed_loop_summary_retains_failed_prefixes(self):
        result = summarize([100., 20., -10.], [30., 4., 1.], [0, 9, 6], 30.)
        self.assertAlmostEqual(result['mean_net_deg'], 110/3)
        self.assertEqual(result['retained'], 1)
        self.assertAlmostEqual(result['retention_fraction'], 1/3)
        self.assertEqual(result['failure_codes'], {'0': 1, '9': 1, '6': 1})


if __name__ == '__main__':
    unittest.main()
