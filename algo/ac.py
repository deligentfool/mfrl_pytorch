import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import tools


class ActorCritic(nn.Module):
    def __init__(self, env, name, handle, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4, use_cuda=False):
        super(ActorCritic, self).__init__()
        
        self.env = env
        self.name = name
        self.view_space = env.unwrapped.env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.unwrapped.env.get_feature_space(handle)[0]
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        self.gamma = gamma
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss
        
        # init training buffers
        self.view_buf = np.empty([1,] + list(self.view_space))
        self.feature_buf = np.empty([1,] + [self.feature_space])
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.replay_buffer = tools.EpisodesBuffer()
        
        self.net = self._construct_net()
        self.optim = torch.optim.Adam(lr=self.learning_rate, params=self.get_all_params())
        self.use_cuda = use_cuda
        
    def get_all_params(self):
        params = []
        for k, v in self.net.items():
            params += list(v.parameters())
        return params
        
    def _construct_net(self):
        temp_dict = nn.ModuleDict()
        temp_dict['obs_linear'] = nn.Linear(np.prod(self.view_space), 256)
        temp_dict['emb_linear'] = nn.Linear(self.feature_space, 256)
        temp_dict['cat_linear'] = nn.Linear(256 * 2, 256 * 2)
        temp_dict['policy_linear'] = nn.Linear(256 * 2, self.num_actions)
        temp_dict['value_linear'] = nn.Linear(256 * 2, 1)
        return temp_dict
    
    def _calc_value(self, **kwargs):
        if self.use_cuda:
            obs = torch.FloatTensor(kwargs['obs']).cuda().unsqueeze(0)
            feature = torch.FloatTensor(kwargs['feature']).cuda().unsqueeze(0)
        else:
            obs = torch.FloatTensor(kwargs['obs']).unsqueeze(0)
            feature = torch.FloatTensor(kwargs['feature']).unsqueeze(0)
        flatten_view = obs.reshape(obs.size()[0], -1)
        h_view = F.relu(self.net['obs_linear'](flatten_view))
        h_emb = F.relu(self.net['emb_linear'](feature))
        dense = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net['cat_linear'](dense))
        value = self.net['value_linear'](dense)
        value = value.flatten()
        return value.detach().cpu().numpy()
    
    def train(self, cuda):
        # calc buffer size
        n = 0
        # batch_data = sample_buffer.episodes()
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer()

        for episode in batch_data:
            n += len(episode.rewards)

        self.view_buf.resize([n,] + list(self.view_space), refcheck=False)
        self.feature_buf.resize([n,] + [self.feature_space], refcheck=False)
        self.action_buf.resize(n, refcheck=False)
        self.reward_buf.resize(n, refcheck=False)
        view, feature = self.view_buf, self.feature_buf
        action, reward = self.action_buf, self.reward_buf
        
        ct = 0
        gamma = self.gamma
        # collect episodes from multiple separate buffers to a continuous buffer
        for episode in batch_data:
            v, f, a, r = episode.views, episode.features, episode.actions, episode.rewards
            m = len(episode.rewards)

            r = np.array(r)

            keep = self._calc_value(obs=v[-1], feature=f[-1])

            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            reward[ct:ct + m] = r
            ct += m

        assert n == ct
        
        if self.use_cuda:
            view = torch.FloatTensor(view).cuda()
            feature = torch.FloatTensor(feature).cuda()
            action = torch.LongTensor(action).cuda()
            reward = torch.FloatTensor(reward).cuda()
            action_mask = torch.zeros([action.size(0), self.num_actions]).cuda().scatter_(1, action.unsqueeze(-1), 1).float()
        else:
            view = torch.FloatTensor(view)
            feature = torch.FloatTensor(feature)
            action = torch.LongTensor(action)
            reward = torch.FloatTensor(reward)
            action_mask = torch.zeros([action.size(0), self.num_actions]).scatter_(1, action.unsqueeze(-1), 1).float()

        # train
        flatten_view = view.flatten(1)
        h_view = F.relu(self.net['obs_linear'](flatten_view))
        h_emb = F.relu(self.net['emb_linear'](feature))
        dense = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net['cat_linear'](dense))
        policy = F.softmax(self.net['policy_linear'](dense / 0.1), dim=-1)
        policy = torch.clamp(policy, 1e-10, 1-1e-10)
        value = self.net['value_linear'](dense)
        value = value.flatten()
        
        advantage = (reward - value).detach()
        log_policy = (policy + 1e-6).log()
        log_prob = (log_policy * action_mask).sum(1)
        
        pg_loss = -(advantage * log_prob).mean()
        vf_loss = self.value_coef * (reward - value).pow(2).mean()
        neg_entropy = self.ent_coef * (policy * log_policy).sum(1).mean()
        total_loss = pg_loss + vf_loss + neg_entropy
        
        # train op (clip gradient)
        self.optim.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.get_all_params(), 5.0)
        self.optim.step()
        
        print('[*] PG_LOSS:', np.round(pg_loss.detach().cpu().item(), 6), '/ VF_LOSS:', np.round(vf_loss.detach().cpu().item(), 6), '/ ENT_LOSS:', np.round(neg_entropy.detach().cpu().item()), '/ Value:', np.mean(value.detach().cpu().numpy()))
        
    def act(self, **kwargs):
        flatten_view = kwargs['obs'].reshape(kwargs['obs'].size()[0], -1)
        h_view = F.relu(self.net['obs_linear'](flatten_view))
        h_emb = F.relu(self.net['emb_linear'](kwargs['feature']))
        dense = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net['cat_linear'](dense))
        policy = F.softmax(self.net['policy_linear'](dense / 0.1), dim=-1)
        policy = torch.clamp(policy, 1e-10, 1-1e-10)
        distribution = torch.distributions.Categorical(policy)
        action = distribution.sample().detach().cpu().numpy()
        return action.astype(np.int32).reshape((-1,))
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
        
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, "ac_{}".format(step))
        torch.save(self.net.state_dict(), file_path)
        print("[*] Model saved")
        
    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "ac_{}".format(step))

        self.net.load_state_dict(torch.load(file_path))
        print("[*] Loaded model")
        
        
        
class MFAC(nn.Module):
    def __init__(self, env, name, handle, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4, use_cuda=False):
        super(MFAC, self).__init__()
        
        self.env = env
        self.name = name
        self.view_space = env.unwrapped.env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.unwrapped.env.get_feature_space(handle)[0]
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        self.gamma = gamma
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss
        
        # init training buffers
        self.view_buf = np.empty([1,] + list(self.view_space))
        self.feature_buf = np.empty([1,] + [self.feature_space])
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)
        
        self.net = self._construct_net()
        self.optim = torch.optim.Adam(lr=self.learning_rate, params=self.get_all_params())
        self.use_cuda = use_cuda
        
    def get_all_params(self):
        params = []
        for k, v in self.net.items():
            params += list(v.parameters())
        return params
        
    def _construct_net(self):
        temp_dict = nn.ModuleDict()
        temp_dict['obs_linear'] = nn.Linear(np.prod(self.view_space), 256)
        temp_dict['emb_linear'] = nn.Linear(self.feature_space, 256)
        # * use the action_prob
        temp_dict['action_linear_1'] = nn.Linear(self.num_actions, 64)
        temp_dict['action_linear_2'] = nn.Linear(64, 32)
        temp_dict['act_obs_emb_linear'] = nn.Linear(32 + 256 * 2, 256)
        temp_dict['value_linear'] = nn.Linear(256, 1)

        temp_dict['cat_linear'] = nn.Linear(256 * 2, 256 * 2)
        temp_dict['policy_linear'] = nn.Linear(256 * 2, self.num_actions)
        return temp_dict
    
    def _calc_value(self, **kwargs):
        if self.use_cuda:
            obs = torch.FloatTensor(kwargs['obs']).cuda().unsqueeze(0)
            feature = torch.FloatTensor(kwargs['feature']).cuda().unsqueeze(0)
            input_act_prob = torch.FloatTensor(kwargs['prob']).cuda().unsqueeze(0)
        else:
            obs = torch.FloatTensor(kwargs['obs']).unsqueeze(0)
            feature = torch.FloatTensor(kwargs['feature']).unsqueeze(0)
            input_act_prob = torch.FloatTensor(kwargs['prob']).unsqueeze(0)
        flatten_view = obs.flatten(1)
        h_view = F.relu(self.net['obs_linear'](flatten_view))
        h_emb = F.relu(self.net['emb_linear'](feature))
        cat_layer = torch.cat([h_view, h_emb], dim=-1)
        action_dense = F.relu(self.net['action_linear_1'](input_act_prob))
        action_dense = F.relu(self.net['action_linear_2'](action_dense))
        cat_act_obs_emb = torch.cat([action_dense, cat_layer], dim=-1)
        dense_act_obs_emb = F.relu(self.net['act_obs_emb_linear'](cat_act_obs_emb))
        value = self.net['value_linear'](dense_act_obs_emb)
        value = value.flatten()
        return value.detach().cpu().numpy()
    
    def train(self, cuda):
        # calc buffer size
        n = 0
        # batch_data = sample_buffer.episodes()
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        for episode in batch_data:
            n += len(episode.rewards)

        self.view_buf.resize([n,] + list(self.view_space), refcheck=False)
        self.feature_buf.resize([n,] + [self.feature_space], refcheck=False)
        self.action_buf.resize(n, refcheck=False)
        self.reward_buf.resize(n, refcheck=False)
        view, feature = self.view_buf, self.feature_buf
        action, reward = self.action_buf, self.reward_buf
        act_prob_buff = np.zeros((n, self.num_actions), dtype=np.float32)
        
        ct = 0
        gamma = self.gamma
        # collect episodes from multiple separate buffers to a continuous buffer
        for episode in batch_data:
            v, f, a, r, prob = episode.views, episode.features, episode.actions, episode.rewards, episode.probs
            m = len(episode.rewards)
            
            assert len(prob) > 0

            r = np.array(r)

            keep = self._calc_value(obs=v[-1], feature=f[-1], prob=prob[-1])

            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            reward[ct:ct + m] = r
            act_prob_buff[ct:ct + m] = prob
            ct += m

        assert n == ct
        
        if self.use_cuda:
            view = torch.FloatTensor(view).cuda()
            feature = torch.FloatTensor(feature).cuda()
            action = torch.LongTensor(action).cuda()
            reward = torch.FloatTensor(reward).cuda()
            act_prob_buff = torch.FloatTensor(act_prob_buff).cuda()
            action_mask = torch.zeros([action.size(0), self.num_actions]).cuda().scatter_(1, action.unsqueeze(-1), 1).float()
        else:
            view = torch.FloatTensor(view)
            feature = torch.FloatTensor(feature)
            action = torch.LongTensor(action)
            reward = torch.FloatTensor(reward)
            act_prob_buff = torch.FloatTensor(act_prob_buff)
            action_mask = torch.zeros([action.size(0), self.num_actions]).scatter_(1, action.unsqueeze(-1), 1).float()

        # train
        flatten_view = view.flatten(1)
        h_view = F.relu(self.net['obs_linear'](flatten_view))
        h_emb = F.relu(self.net['emb_linear'](feature))
        cat_layer = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net['cat_linear'](cat_layer))
        policy = F.softmax(self.net['policy_linear'](dense / 0.1), dim=-1)
        policy = torch.clamp(policy, 1e-10, 1-1e-10)
        action_dense = F.relu(self.net['action_linear_1'](act_prob_buff))
        action_dense = F.relu(self.net['action_linear_2'](action_dense))
        cat_act_obs_emb = torch.cat([action_dense, cat_layer], dim=-1)
        dense_act_obs_emb = F.relu(self.net['act_obs_emb_linear'](cat_act_obs_emb))
        value = self.net['value_linear'](dense_act_obs_emb)
        value = value.flatten()
        
        advantage = (reward - value).detach()
        log_policy = (policy + 1e-6).log()
        log_prob = (log_policy * action_mask).sum(1)
        
        pg_loss = -(advantage * log_prob).mean()
        vf_loss = self.value_coef * (reward - value).pow(2).mean()
        neg_entropy = self.ent_coef * (policy * log_policy).sum(1).mean()
        total_loss = pg_loss + vf_loss + neg_entropy
        
        # train op (clip gradient)
        self.optim.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.get_all_params(), 5.0)
        self.optim.step()
        
        print('[*] PG_LOSS:', np.round(pg_loss.detach().cpu().item(), 6), '/ VF_LOSS:', np.round(vf_loss.detach().cpu().item(), 6), '/ ENT_LOSS:', np.round(neg_entropy.detach().cpu().item()), '/ Value:', np.mean(value.detach().cpu().numpy()))
        
    def act(self, **kwargs):
        flatten_view = kwargs['obs'].reshape(kwargs['obs'].size()[0], -1)
        h_view = F.relu(self.net['obs_linear'](flatten_view))
        h_emb = F.relu(self.net['emb_linear'](kwargs['feature']))
        cat_layer = torch.cat([h_view, h_emb], dim=-1)
        dense = F.relu(self.net['cat_linear'](cat_layer))
        policy = F.softmax(self.net['policy_linear'](dense / 0.1), dim=-1)
        policy = torch.clamp(policy, 1e-10, 1-1e-10)
        distribution = torch.distributions.Categorical(policy)
        action = distribution.sample().detach().cpu().numpy()
        return action.astype(np.int32).reshape((-1,))
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
        
    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, "mfac_{}".format(step))
        torch.save(self.net.state_dict(), file_path)
        print("[*] Model saved")
        
    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "mfac_{}".format(step))

        self.net.load_state_dict(torch.load(file_path))
        print("[*] Loaded model")
