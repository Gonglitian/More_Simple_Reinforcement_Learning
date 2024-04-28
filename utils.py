from IPython import display
import random
import gym
from typing import Tuple, Callable, Optional, Union, List, Any, NamedTuple
import pickle
import torch


Experience = Tuple[List[float], int, float, List[float], bool]


class Env(gym.Wrapper):
    def __init__(self, env_name: str):
        env = gym.make(env_name, render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.env_name = env_name
        # self.state_dim = self.env.observation_space.shape
        # self.action_dim = self.env.action_space.shape
        self.step_controler: Optional[Callable[[
            Tuple, float, bool], Tuple]] = None

    def reset(self):
        return self.env.reset()[0]

    def step(self, action) -> Tuple[Any, float, bool]:
        # modify env response through
        state, reward, terminated, truncated, info = self.env.step(action)
        over = terminated or truncated

        if self.step_controler is not None:
            state, reward, over = self.step_controler(state, reward, over)
        return state, reward, over

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        # todo validate
        return self.env.seed(seed)

    def sample_action(self):
        return self.env.action_space.sample()

    def show(self) -> None:
        """
        显示游戏图像。
        """
        from matplotlib import pyplot as plt
        plt.figure(figsize=(3, 3))
        plt.imshow(self.env.render())
        plt.show()

    def __str__(self):
        return self.env_name


class Pool:
    def __init__(self, capacity: int = 10000, sample_size: int = 32):
        self.capacity = capacity
        self.data: List[Experience] = []
        self.position: int = 0
        self.sample_size = sample_size

    def push(self, data: Tuple):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.data, self.sample_size)

    @property
    def len(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f'Pool({self.env}, {len(self)})'

    def __repr__(self):
        return self.__str__()


class Agent:
    def __init__(self, pool: Pool, model=None):
        self.env: Env = None
        self.pool = pool
        self.model_dict = {}
        self.optimizer_dict = {}
        self.loss_fn_dict = {}
        self.fit_times = 200

    def act(self, state):
        # must have a model named 'action'
        assert 'action_model' in self.model_dict
        # if action model is torch.nn.Module, turn it into eval mode and turn state to tensor
        if isinstance(self.model_dict['action_model'], (torch.nn.Module, torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
            self.model_dict['action_model'].eval()
            # print(state)
            state = torch.tensor(state, dtype=torch.float32)
            state = state.unsqueeze(0)

        # note choose biggest value for action
        action = self.model_dict['action_model'](state).argmax().item()
        return action

    def learn(self):
        for i in range(self.fit_times):
            state, action, reward, next_state, over = self.pool.sample()
            # 计算value
            value = model(state).gather(dim=1, index=action)

            # 计算target
            with torch.no_grad():
                target = model(next_state)
            target = target.max(dim=1)[0].reshape(-1, 1)
            target = target * 0.99 * (1 - over) + reward

            loss = self.loss_fn_dict['action_model'](value, target)
            loss.backward()
            self.optimizer_dict['action_model'].step()
            self.optimizer_dict['action_model'].zero_grad()

    def create_model(self, name: str, model: Union[torch.nn.Module, Callable], optimizer: Optional[torch.optim.Optimizer] = None, loss_fn=None):
        self.model_dict[name] = model
        if optimizer is not None:
            self.optimizer_dict[name] = optimizer
        else:
            self.optimizer_dict[name] = torch.optim.Adam(
                model.parameters(), lr=2e-4)
        if loss_fn is not None:
            self.loss_fn_dict[name] = loss_fn
        else:
            self.loss_fn_dict[name] = torch.nn.MSELoss()

    def collect_data(self, data: Experience):
        self.pool.push(data)

    def save(self, path):
        raise NotImplementedError


class Simulator:
    def __init__(self, agent: Agent, env: Env, show: bool = False):
        # bind elements to Simulator
        self.env = env
        self.agent = agent
        self.agent.env = env
        self.check()  # check if compatible or not
        # Simulator's state
        self.current_state = None
        self.current_step = 0
        self.show = show
        self.reward_sum = 0
        self.round_reward_list = []

        # Simulator's config
        self.epochs = 1000
        self.log_interval = 10

    def check(self):
        # todo
        # check if agent's model input dim is equal to env's state dim
        # check if agent's model output dim is equal to env's action dim
        # for each model in agent.model_dict:
        # assert self.agent.model_dict['model'].in_features == self.env.observation_space.shape[0]
        # assert self.agent.model_dict['model'].out_features == self.env.action_space.n

        # todo tell user how to change input dim and output dim
        pass

    def interact(self):
        state = self.env.reset()
        over = False
        while not over:
            action = self.agent.act(state)
            next_state, reward, over = self.env.step(action)
            self.agent.collect_data(
                (state, action, reward, next_state, over))
            state = next_state

            # display
            if self.show:
                display.clear_output(wait=True)
                self.env.show()

            # record
            self.reward_sum += reward
            self.current_state = state
            self.current_step += 1

        # record
        self.round_reward_list.append(self.reward_sum)

    def train(self):
        # set optimizers and loss functions
        for name, model in self.agent.model_dict.items():
            if isinstance(model, (torch.nn.Module, torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
                model.train()

        for epoch in range(self.epochs):
            self.interact()
            self.agent.learn()

            if epoch % self.log_interval == 0:
                print(epoch, self.reward_sum)

    def save(self, path):
        # save state and pool
        with open(path, 'wb') as f:
            pickle.dump((self.current_state, self.agent.pool), f)

    def load(self, path):
        # load state and pool
        with open(path, 'rb') as f:
            (self.current_state, self.agent.pool) = pickle.load(f)

    def __str__(self):
        return f'Simulator({self.env})'

    def __repr__(self):
        return self.__str__()

    def pause(self):
        # save current state,pool,model
        # stop interact
        pass

    def stop():
        # stop interact
        pass


if __name__ == '__main__':
    env = Env('CartPole-v1')
    agent = Agent(Pool(10000))
    ############################
    # config agent
    # set agent model
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    agent.create_model('action_model', model)

    # set agent train method
    # agent.learn = ...
    ############################
    Simulator = Simulator(agent, env, show=False)
    Simulator.train()
    print(Simulator.agent.pool.len)
