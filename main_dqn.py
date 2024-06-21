from env import *
from DDPG import *
from DQN import *
import yaml


if __name__ == '__main__':

    with open('./config.yaml', 'r',encoding="utf-8") as f:
        config = yaml.safe_load(f)

    memory_size = config['memory_size']
    batch_size = config['batch_size']
    hidden_dim = config['hidden_dim']
    hidden_layers = config['hidden_layers']
    num_frames = config['num_frames']
    epoch = config['epoch']

    env = Env(60)
    input_dim = env.state.size
    algo = 'dqn'

    agent = DQNAgent(env,memory_size,batch_size,)
    for i in range(epoch):#这里epoch就是迭代次数
        agent.train(576)

    











