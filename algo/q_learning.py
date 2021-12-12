import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import base
from . import tools


class DQN(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, memory_size=2**10, batch_size=64, update_every=5, use_mf=False, learning_rate=0.0001, tau=0.005, gamma=0.95):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf, learning_rate=learning_rate, tau=tau, gamma=gamma)
        
        self.replay_buffer = tools.MemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)
        
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
        
    def train(self, cuda):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feat, obs_next, feat_next, dones, rewards, acts, masks = self.replay_buffer.sample()
            
            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat).cuda() if cuda else torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next).cuda() if cuda else torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts).cuda() if cuda else torch.LongTensor(acts)
            rewards = torch.FloatTensor(rewards).cuda() if cuda else torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones).cuda() if cuda else torch.FloatTensor(dones)
            masks = torch.FloatTensor(masks).cuda() if cuda else torch.FloatTensor(masks)
            
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, q = super().train(obs=obs, feature=feat, target_q=target_q, acts=acts, mask=masks)
            
            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)
    
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        eval_file_path = os.path.join(dir_path, "dqn_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "dqn_target_{}".format(step))
        torch.save(self.eval_net.state_dict(), eval_file_path)
        torch.save(self.target_net.state_dict(), target_file_path)
        print("[*] Model saved")
        
    def load(self, dir_path, step=0):
        eval_file_path = os.path.join(dir_path, "dqn_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "dqn_target_{}".format(step))

        self.target_net.load_state_dict(torch.load(target_file_path))
        self.eval_net.load_state_dict(torch.load(eval_file_path))
        print("[*] Loaded model")
        
        

class MFQ(base.ValueNet):
    def __init__(self, env, name, handle, sub_len, eps=1.0, memory_size=2**10, batch_size=64, update_every=5, use_mf=True, learning_rate=0.0001, tau=0.005, gamma=0.95):
        super().__init__(env, name, handle, update_every=update_every, use_mf=use_mf, learning_rate=learning_rate, tau=tau, gamma=gamma)
        
        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)
        self.update_every = update_every
        
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self, cuda):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            
            obs = torch.FloatTensor(obs).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs).permute([0, 3, 1, 2])
            obs_next = torch.FloatTensor(obs_next).permute([0, 3, 1, 2]).cuda() if cuda else torch.FloatTensor(obs_next).permute([0, 3, 1, 2])
            feat = torch.FloatTensor(feat).cuda() if cuda else torch.FloatTensor(feat)
            feat_next = torch.FloatTensor(feat_next).cuda() if cuda else torch.FloatTensor(feat_next)
            acts = torch.LongTensor(acts).cuda() if cuda else torch.LongTensor(acts)
            act_prob = torch.FloatTensor(act_prob).cuda() if cuda else torch.FloatTensor(act_prob)
            act_prob_next = torch.FloatTensor(act_prob_next).cuda() if cuda else torch.FloatTensor(act_prob_next)
            rewards = torch.FloatTensor(rewards).cuda() if cuda else torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones).cuda() if cuda else torch.FloatTensor(dones)
            masks = torch.FloatTensor(masks).cuda() if cuda else torch.FloatTensor(masks)
            
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(obs=obs, feature=feat, target_q=target_q, prob=act_prob, acts=acts, mask=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        eval_file_path = os.path.join(dir_path, "mfq_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "mfq_target_{}".format(step))
        torch.save(self.eval_net.state_dict(), eval_file_path)
        torch.save(self.target_net.state_dict(), target_file_path)
        print("[*] Model saved")
        
    def load(self, dir_path, step=0):
        eval_file_path = os.path.join(dir_path, "mfq_eval_{}".format(step))
        target_file_path = os.path.join(dir_path, "mfq_target_{}".format(step))

        self.target_net.load_state_dict(torch.load(target_file_path))
        self.eval_net.load_state_dict(torch.load(eval_file_path))
        print("[*] Loaded model")