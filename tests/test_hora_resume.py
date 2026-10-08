"""A restored original-Hora learner must match the next update after the same reset."""
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from tendonspin.baselines.hora_training import ROOT,load_ppo
from tendonspin.rl.checkpoint import save_checkpoint,restore_checkpoint


class ToyEnv:
    action_space=Box(-1.,1.,shape=(16,),dtype=np.float32)
    observation_space=Box(-5.,5.,shape=(96,),dtype=np.float32)
    def reset(self):
        self.t=0
        return self.observe()
    def observe(self):return dict(obs=torch.randn(4,96)*.1,priv_info=torch.randn(4,9)*.1)
    def step(self,action):
        self.t+=1
        return self.observe(),-(action**2).mean(-1),torch.full((4,),self.t%4==0,dtype=torch.uint8),{}


class HoraResumeTests(unittest.TestCase):
    def test_next_update_matches_after_reset(self):
        torch.set_num_threads(2);torch.manual_seed(43);np.random.seed(43);random.seed(43)
        cfg=OmegaConf.create(dict(seed=43,rl_device='cpu',test=False,checkpoint=None,
            task=dict(env=dict(numEnvs=4)),train=OmegaConf.load(ROOT/'third_party/hora/configs/train/AllegroHandHora.yaml')))
        cfg.train.ppo.priv_info=True;cfg.train.ppo.minibatch_size=16
        PPO=load_ppo();contract={'test':'CPU original PPO; not robot physics'}
        with tempfile.TemporaryDirectory() as directory:
            a=PPO(ToyEnv(),directory+'/a',cfg);b=PPO(ToyEnv(),directory+'/b',cfg)
            try:
                a.obs=a.env.reset();a.epoch_num=1;a.train_epoch();a.storage.data_dict=None
                progress=dict(actions_executed=32,completed_updates=1,completed_episodes=8,
                              episode_stop_counts={},lineage_id='test')
                file=Path(directory)/'state.pth';save_checkpoint(a,file,contract=contract,progress=progress)
                saved=torch.load(file,weights_only=False)
                # Control arm retains learner state but follows the declared reset protocol.
                a.current_rewards.zero_();a.current_lengths.zero_();a.dones.fill_(1)
                a.obs=a.env.reset();a.epoch_num=2;a.train_epoch();a.storage.data_dict=None
                expected={k:v.clone() for k,v in a.model.state_dict().items()}
                returned=restore_checkpoint(b,file,contract=contract)
                self.assertEqual(returned,progress)
                self.assertEqual(b.agent_steps,32);self.assertEqual(b.epoch_num,1)
                self.assertEqual(b.last_lr,saved['learning_rate'])
                torch.testing.assert_close(b.value_mean_std.state_dict(),saved['value_mean_std'],rtol=0,atol=0)
                self.assertTrue(b.optimizer.state_dict()['state'])
                b.obs=b.env.reset();b.epoch_num=2;b.train_epoch()
                torch.testing.assert_close(b.model.state_dict(),expected,rtol=0,atol=0)
                torch.testing.assert_close(b.running_mean_std.state_dict(),a.running_mean_std.state_dict(),rtol=0,atol=0)
                torch.testing.assert_close(b.value_mean_std.state_dict(),a.value_mean_std.state_dict(),rtol=0,atol=0)
                torch.testing.assert_close(b.optimizer.state_dict(),a.optimizer.state_dict(),rtol=0,atol=0)
                self.assertEqual(b.agent_steps,64);self.assertEqual(b.last_lr,a.last_lr)
                with self.assertRaisesRegex(ValueError,'contract mismatch'):
                    restore_checkpoint(b,file,contract={'different':'task'})
            finally:a.writer.close();b.writer.close()


if __name__=='__main__':unittest.main()
