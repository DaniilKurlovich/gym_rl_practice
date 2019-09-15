import random
import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

GAMMA = 0.85
EXPLORATION_BACKUP = 0.95

class PlayerAgent:
    def __init__(self, count_states: int, count_actions: int):
        self.model = None
        self.create_model(count_states, count_actions)
        self.action_space = count_actions
        self.exploration_rate = 1.0
        self.memory = []

    def create_model(self, count_states, count_actions):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(count_states, ),
                             activation='relu'))
        self.model.add(Dense(count_actions, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))

    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, reward, action, next_state, terminal):
        self.memory.append((state, reward, action, next_state, terminal))

    def experience_replay(self):
        if len(self.memory) < 20:
            return
        batch = random.sample(self.memory, 20)
        for state, reward, action, next_state, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = reward + GAMMA * np.amax(
                    self.model.predict(next_state)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        if self.exploration_rate > 0.011:
            self.exploration_rate *= 0.998


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    attempt = 0
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = PlayerAgent(obs_space, action_space)
    while attempt < 500:
        cur_state = env.reset()
        cur_state = np.reshape(cur_state, [1, obs_space])
        attempt += 1
        step = 0
        shaping_reward = cur_state[0][1] + 0.5
        while True:
            step += 1
            if attempt > 100:
                env.render()
            action = agent.get_action(cur_state)
            next_state, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            next_state = np.reshape(next_state, [1, obs_space])
            agent.remember(next_state=next_state, reward=shaping_reward,
                           terminal=terminal, state=cur_state,
                           action=action)
            agent.experience_replay()
            cur_state = next_state
            shaping_reward = reward + 10 * abs(next_state[0][1])
            if terminal:
                print('Attempt: {}, State: Left{}, Right{}, Reward: {}'.format(
                    attempt, cur_state[0][0], cur_state[0][1], shaping_reward))
                break
