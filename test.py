import gym
import keras


env = gym.make('CartPole-v0')
env.reset()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    attempt = 0
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = keras.models.load_model('my_model.h5')
    while attempt < 100:
        cur_state = env.reset()
        attempt += 1
        step = 0
        while True:
            step += 1
            env.render()
            action = agent.predict(cur_state)
            next_state, reward, terminal, info = env.step(action)
            cur_state = next_state
            if terminal:
                print("Attempt: " + str(attempt) + ", exploration: " + str(agent.exploration_rate) + ", score: " + str(step))
                break

env.close()
