"""Self Play
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pettingzoo.magent import battle_v3

from algo import spawn_ai
from algo import tools
from senarios.senario_battle import play


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs('./data', exist_ok=True)

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=5, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=2000, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--cuda', type=bool, default=True, help='use the cuda')

    args = parser.parse_args()

    # Initialize the environment
    env = battle_v3.env(
        map_size=args.map_size,
        minimap_mode=True,
        step_reward=-0.005,
        dead_penalty=-0.1,
        attack_penalty=-0.1,
        attack_opponent_reward=0.2,
        max_cycles=args.max_steps,
        extra_features=True
    )
    handles = env.unwrapped.env.get_handles()

    log_dir = os.path.join(BASE_DIR, 'data/tmp/{}'.format(args.algo))
    render_dir = os.path.join(BASE_DIR, 'data/render/{}'.format(args.algo))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.algo))

    start_from = 0

    models = [spawn_ai(args.algo, env, handles[0], args.algo + '-me', args.max_steps, args.cuda), spawn_ai(args.algo, env, handles[1], args.algo + '-opponent', args.max_steps, args.cuda)]
    runner = tools.Runner(env, handles, args.max_steps, models, play,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, render_dir=render_dir, train=True, cuda=args.cuda)

    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
