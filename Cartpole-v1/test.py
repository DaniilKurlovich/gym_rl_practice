import gym
import random
import importlib.util
import numpy as np
import heapq

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


spec_module = importlib.util.find_spec('perstep')
module = None

if not spec_module:
    import os
    os.error('module perstep is empty')
else:
    module = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(module)
    del spec_module

GAMMA = 10
EXPLORATION_BACKUP = 0.95


class PlayerAgent:
    def __init__(self, count_states: int, count_actions: int):
        self.model = None
        self.obs_space = count_states
        self.create_model(count_states)
        self.action_space = count_actions
        self.exploration_rate = 1.0

        self.batch_size = 500
        self.elit_batch_size = 1500
        self.elit_batch_size_start_learn = 70
        self.cd = []
        self.lower_bound_elite = 10
        self.possible_random_actions = 5000
        self.start_learn_bad = 250
        self.bad_size = 500
        self.enable_random_action = True
        self.average_step = 1
        self.k = 30
        self.d = 7
        self.epsilon = 2
        self.exc_flag = None

        self.bad_elite = []
        self.elit_states = []

    def create_model(self, count_states):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(count_states, ),
                             activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))

    def get_action(self, state):
        if random.random() < self.exploration_rate:
            if self.exploration_rate > 0.05:
                self.exploration_rate *= 0.9995
                return random.randrange(self.action_space)
        next_state_zero, _, _, _ = module.virtual_step(state[0], 0)
        next_state_one, _, _, _ = module.virtual_step(state[0], 1)
        v_values_zero = self.model.predict(np.reshape(next_state_zero, [1, self.obs_space]))
        v_values_one = self.model.predict(np.reshape(next_state_one, [1, self.obs_space]))
        if v_values_one[0][0] > v_values_zero[0][0]:
            return 1
        else:
            return 0

    def push_to_heap(self, state, reward, terminal, score):
        '''
        :param tuple: (score, (state, reward, terminal))
        :return: void
        '''
        self.exc_flag = None
        value_added = 0
        while not self.exc_flag:
            try:
                heapq.heappush(self.elit_states, (score + value_added,
                                                  (state, reward, terminal)))
                self.exc_flag = 'stop'
                value_added += 0.01
            except ValueError:
                self.exc_flag = None

    def remember(self, state, reward, terminal, score):
        if self.lower_bound_elite + self.epsilon < score:
            self.push_to_heap(state, reward, terminal, score)
            self.lower_bound_elite = float(sum(i[0] for i in
                                               self.elit_states)) / max(
                len(self.elit_states), 1) + 0.25
        else:
            if self.d < score < self.k:
                self.d += 0.01
                self.k += 0.01
                self.bad_elite.append((state, reward, terminal))
                if len(self.bad_elite) > self.bad_size:
                    del self.bad_elite[0]

        if len(self.elit_states) > self.elit_batch_size:
            heapq.heappop(self.elit_states)

    @staticmethod
    def get_max_reward_from_state(state):
        max_reward = None
        for _action in [0, 1]:
            _, _reward, _, _ = module.virtual_step(state, _action)
            if not max_reward:
                max_reward = _reward
            elif max_reward < _reward:
                max_reward = _reward
        return max_reward

    def learn_elite(self):
        if len(self.bad_elite) < self.start_learn_bad or \
                len(self.elit_states) < self.elit_batch_size_start_learn:
            return

        for item1, item2 in self.elit_states:
            state, reward, terminal = item2
            v_update = reward
            if not terminal:
                v_update = reward + GAMMA * self.get_max_reward_from_state(
                    state[0])
            self.model.fit(state, np.array([v_update]), verbose=0)

    def learn_bad(self):
        if len(self.bad_elite) < self.start_learn_bad or \
           len(self.elit_states) < self.elit_batch_size_start_learn:
            return

        count_batchs = min(25, self.start_learn_bad)
        batch = random.sample(self.bad_elite, count_batchs)
        for state, reward, terminal in batch:
            v_update = reward
            if not terminal:
                v_update = reward + GAMMA * self.get_max_reward_from_state(
                    state[0])
            self.model.fit(state, np.array([v_update]), verbose=0)

    def experience_replay(self):
        '''
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, reward, terminal in batch:
            v_update = reward
            if not terminal:
                v_update = reward + GAMMA * self.get_max_reward_from_state(
                    state[0])
            self.model.fit(state, np.array([v_update]), verbose=0)
        '''
        self.learn_bad()
        self.learn_elite()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    attempt = 0
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = PlayerAgent(obs_space, action_space)
    max_score = 0
    is_elite_st = False
    while attempt < 15000:
        cur_state = env.reset()
        cur_state = np.reshape(cur_state, [1, obs_space])
        attempt += 1
        gamma_reward = 0
        step = 0
        while True:
            step += 1
            if attempt > 5000:
                env.render()
            action = agent.get_action(cur_state)
            next_state, reward, terminal, info = env.step(action)
            gamma_reward += reward
            if gamma_reward > max_score:
                max_score = gamma_reward
                is_elite_st = True
            reward = reward if not terminal else -reward
            next_state = np.reshape(next_state, [1, obs_space])
            agent.remember(state=cur_state, reward=reward,
                           terminal=terminal,
                           score=gamma_reward)
            is_elite_st = False
            agent.experience_replay()
            cur_state = next_state
            if terminal:
                print("Attempt: " + str(attempt) + ", exploration: "
                      + str(agent.exploration_rate) + ", score: "
                      + str(step) + ', max score:', max_score, ' lb: ',
                      agent.lower_bound_elite, ' k: ', agent.k, ' d: ',
                      agent.d)
                break
