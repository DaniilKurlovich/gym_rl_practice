import gym
import random
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


np.random.seed(2)
Episodes = 5000


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 50
        self.learning_rate = 0.001
        self.Epsilon = epsilon
        self.Gamma = 0.9
        self.Epsilon_decay = epsilon_decay
        self.Epsilon_min = 0.1
        self.memory = deque(maxlen=2000)
        self.model = self.buildModel()

    def buildModel(self):
        model = Sequential()
        model.add(Dense(15, input_dim=self.state_size, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def chooseAction(self, state):
        if (np.random.uniform() <= self.Epsilon):
            return random.randrange(self.action_size)
        action = self.model.predict(state)
        return np.argmax(action)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        cost = 0
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.Gamma * np.amax(
                    self.model.predict(next_state))
            current = self.model.predict(state)
            cost += abs(target - current[0][action])
            current[0][action] = target
            self.model.fit(state, current, epochs=1, verbose=0)
        if (self.Epsilon > self.Epsilon_min):
            self.Epsilon *= self.Epsilon_decay

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, 1.0, 0.99)
    done = False
    for e in range(Episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        steps = 0
        for time in range(1000):
            if (e >= 2500):
                env.render()
            action = agent.chooseAction(state)
            if action != 1:
                steps += 1
            next_state, reward, done, _ = env.step(action)
            angle = math.atan2(next_state[1], next_state[0])
            reward = (-next_state[0] - next_state[2] + angle) * 10
            if reward > 10:
                reward += 50
            next_state = np.reshape(next_state, [1, state_size])
            agent.store(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.replay()
        if done:
            print("Episode: {}/{}, e: {:.2}, Time: {}, Steps: {}"
                  .format(e, Episodes, agent.Epsilon, time, steps), end='\n')
        else:
            print("Episode: {}/{}, e: {:.2}, Time: {}, Steps: {}"
                  .format(e, Episodes, agent.Epsilon, time, steps), end='\r')
    env.close()
