import gym
env = gym.make('MountainCarContinuous-v0')
print(env.action_space)

print(env.observation_space)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break
