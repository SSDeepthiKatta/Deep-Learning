import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()

EPISODES = 1000

for i_episode in range(EPISODES):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("episode: {}/{}, score: {}".format(i_episode, EPISODES, t))
            break
env.env.close()
