from . import q_learning

IQL = q_learning.DQN
MFQ = q_learning.MFQ


def spawn_ai(algo_name, env, handle, human_name, max_steps, cuda=True):
    if algo_name == 'mfq':
        model = MFQ(env, human_name, handle, max_steps, memory_size=80000)
    elif algo_name == 'iql':
        model = IQL(env, human_name, handle, max_steps, memory_size=80000)
    if cuda:
        model = model.cuda()
    return model