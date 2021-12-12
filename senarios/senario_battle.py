import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def play(env, n_round, handles, models, print_every, eps=1.0, render=False, train=False, cuda=True):
    """play a ground and train"""
    env.reset()

    max_steps = env.unwrapped.max_cycles
    step_ct = 0
    done = False
    
    obs_list = []
    if render:
        obs_list.append(env.render(mode='rgb_array'))
    
    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.unwrapped.env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.unwrapped.env.get_action_space(handles[0])[0], env.unwrapped.env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.unwrapped.env.get_action_space(handles[0])[0])), np.zeros((1, env.unwrapped.env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.unwrapped.env.get_observation(handles[i]))
            ids[i] = env.unwrapped.env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            if cuda:
                acts[i] = models[i].act(obs=torch.FloatTensor(state[i][0]).permute([0, 3, 1, 2]).cuda(), feature=torch.FloatTensor(state[i][1]).cuda(), prob=torch.FloatTensor(former_act_prob[i]).cuda(), eps=eps)
            else:
                acts[i] = models[i].act(obs=torch.FloatTensor(state[i][0]).permute([0, 3, 1, 2]), feature=torch.FloatTensor(state[i][1]), prob=torch.FloatTensor(former_act_prob[i]), eps=eps)
                
                
        for i in range(n_group):
            env.unwrapped.env.set_action(handles[i], acts[i].astype(np.int32))

        # simulate one step
        done = env.unwrapped.env.step()

        for i in range(n_group):
            rewards[i] = env.unwrapped.env.get_reward(handles[i])
            alives[i] = env.unwrapped.env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        if train:
            models[0].flush_buffer(**buffer)

        # stat info
        nums = [env.unwrapped.env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            obs_list.append(env.render(mode='rgb_array'))
            
        # clear dead agents
        env.unwrapped.env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train(cuda)

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards, obs_list


def battle(env, n_round, handles, models, print_every, eps=1.0, render=False, train=False, cuda=True):
    """play a ground and train"""
    env.reset()

    max_steps = env.unwrapped.max_cycles
    step_ct = 0
    done = False

    obs_list = []
    if render:
        obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)))

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.unwrapped.env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.unwrapped.env.get_action_space(handles[0])[0], env.unwrapped.env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.unwrapped.env.get_action_space(handles[0])[0])), np.zeros((1, env.unwrapped.env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.unwrapped.env.get_observation(handles[i]))
            ids[i] = env.unwrapped.env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            if cuda:
                acts[i] = models[i].act(obs=torch.FloatTensor(state[i][0]).permute([0, 3, 1, 2]).cuda(), feature=torch.FloatTensor(state[i][1]).cuda(), prob=torch.FloatTensor(former_act_prob[i]).cuda(), eps=eps)
            else:
                acts[i] = models[i].act(obs=torch.FloatTensor(state[i][0]).permute([0, 3, 1, 2]), feature=torch.FloatTensor(state[i][1]), prob=torch.FloatTensor(former_act_prob[i]), eps=eps)
            
        for i in range(n_group):
            env.unwrapped.env.set_action(handles[i], acts[i].astype(np.int32))

        # simulate one step
        done = env.unwrapped.env.step()

        for i in range(n_group):
            rewards[i] = env.unwrapped.env.get_reward(handles[i])
            alives[i] = env.unwrapped.env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # stat info
        nums = [env.unwrapped.env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)))
            
        # clear dead agents
        env.unwrapped.env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards, obs_list
