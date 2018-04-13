""" File for setting up and understanding the basics of the cart-pole environment in
    the open-ai gym
"""

import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()  # just to make sure everything starts from scratch

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    # close the environment
    env.close()
