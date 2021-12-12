import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueNet(nn.Module):
    def __init__(self, env, name, handle, update_every=5, use_mf=False, learning_rate=1e-4, tau=0.005, gamma=0.95):
        super(ValueNet, self).__init__()
        self.env = env
        self.name = name
        self._saver = None
        
        self.view_space = env.unwrapped.env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.unwrapped.env.get_feature_space(handle)[0]
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        
        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.temperature = 0.1
        
        self.lr= learning_rate
        self.tau = tau
        self.gamma = gamma
        
        self.eval_net = self._construct_net()
        self.target_net = self._construct_net()
        
        self.optim = torch.optim.Adam(lr=self.lr, params=self.get_params(self.eval_net))
        
    def _construct_net(self):
        temp_dict = nn.ModuleDict()
        temp_dict['conv1'] = nn.Conv2d(in_channels=self.view_space[2], out_channels=32, kernel_size=3)
        temp_dict['conv2'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        temp_dict['obs_linear'] = nn.Linear(self.get_flatten_dim(temp_dict), 256)
        temp_dict['emb_linear'] = nn.Linear(self.feature_space, 32)
        if self.use_mf:
            temp_dict['prob_emb_linear'] = nn.Sequential(
                nn.Linear(self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        temp_dict['final_linear'] = nn.Sequential(
            nn.Linear(320 if self.use_mf else 288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        return temp_dict
        
    def get_flatten_dim(self, dict):
        return dict['conv2'](dict['conv1'](torch.zeros(1, self.view_space[2], self.view_space[0], self.view_space[1]))).flatten().size()[0]
    
    def get_params(self, dict):
        params = []
        for k, v in dict.items():
            params += list(v.parameters())
        return params
    
    def get_all_params(self):
        params = []
        eval_params = self.get_params(self.eval_net)
        target_params = self.get_params(self.target_net)
        params += eval_params
        params += target_params
        return params
    
    def calc_target_q(self, obs, feature, dones, rewards, prob=None):
        t_h = F.relu(self.target_net['conv2'](F.relu(self.target_net['conv1'](obs)))).flatten(start_dim=1)
        t_h = torch.cat([self.target_net['obs_linear'](t_h), self.target_net['emb_linear'](feature)], -1)
        if self.use_mf:
            t_h = torch.cat([t_h, self.target_net['prob_emb_linear'](prob)], -1)
        t_q = self.target_net['final_linear'](t_h)

        e_h = F.relu(self.eval_net['conv2'](F.relu(self.eval_net['conv1'](obs)))).flatten(start_dim=1)
        e_h = torch.cat([self.eval_net['obs_linear'](e_h), self.eval_net['emb_linear'](feature)], -1)
        if self.use_mf:
            e_h = torch.cat([e_h, self.eval_net['prob_emb_linear'](prob)], -1)
        e_q = self.eval_net['final_linear'](e_h)
        
        act_idx = e_q.max(1)[1]
        q_values = torch.gather(t_q, 1, act_idx.unsqueeze(-1))
        target_q_value = rewards + (1. - dones) * q_values.reshape(-1) * self.gamma
        return target_q_value
    
    def update(self):
        for k, v in self.target_net.items():
            for param, target_param in zip(self.eval_net[k].parameters(), self.target_net[k].parameters()):
                target_param.detach().copy_(self.tau * param.detach() + (1. - self.tau) * target_param.detach())
                
    def act(self, obs, feature, prob=None, eps=None):
        if eps is not None:
            self.temperature = eps
            
        e_h = F.relu(self.eval_net['conv2'](F.relu(self.eval_net['conv1'](obs)))).flatten(start_dim=1)
        e_h = torch.cat([self.eval_net['obs_linear'](e_h), self.eval_net['emb_linear'](feature)], -1)
        if self.use_mf:
            e_h = torch.cat([e_h, self.eval_net['prob_emb_linear'](prob)], -1)
        e_q = self.eval_net['final_linear'](e_h)
        predict = F.softmax(e_q / self.temperature, dim=-1)
        #distribution =  torch.distributions.Categorical(predict)
        #actions = distribution.sample().detach().cpu().numpy()
        actions = predict.max(1)[1].detach().cpu().numpy()
        return actions
    
    def train(self, obs, feature, target_q, acts, prob=None, mask=None):
        e_h = F.relu(self.eval_net['conv2'](F.relu(self.eval_net['conv1'](obs)))).flatten(start_dim=1)
        e_h = torch.cat([self.eval_net['obs_linear'](e_h), self.eval_net['emb_linear'](feature)], -1)
        if self.use_mf:
            e_h = torch.cat([e_h, self.eval_net['prob_emb_linear'](prob)], -1)
        e_q = self.eval_net['final_linear'](e_h)
        
        e_q = torch.gather(e_q, 1, acts.unsqueeze(-1)).squeeze()
        if mask is not None:
            loss = ((e_q - target_q.detach()).pow(2) * mask).sum() / mask.sum()
        else:
            loss = (e_q - target_q.detach()).pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), {'Eval-Q': np.round(np.mean(e_q.detach().cpu().numpy()), 6), 'Target-Q': np.round(np.mean(target_q.detach().cpu().numpy()), 6)}
        