# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

EPISODES = 100000
plt.ion()
plt.gca().set_color_cycle(['orange', 'blue'])

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.9   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            #if not done:
            target = reward + self.gamma * \
                                  np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)


            # target = self.model.predict(state)
            # if done:
            #     target[0][action] = reward
            # else:
            #     target[0][action] = reward + self.gamma * \
            #                                  np.max(self.model.predict(next_state)[0])
            # self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("./save/lunarlander.h5")
    done = False
    batch_size = 64

    rewards = deque(maxlen=200)
    qvalues = deque(maxlen=200)

    avrewards = []
    avqvalues = []

    # for e in range(1000):
    #     state = env.reset()
    #     state = np.reshape(state, [1, state_size])
    #     reward_total = 0
    #     for time in range(500):
    #         action = env.action_space.sample()
    #         next_state, reward, done, _ = env.step(action)
    #         reward_total += reward
    #         next_state = np.reshape(next_state, [1, state_size])
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             break
    #     # if len(agent.memory) > batch_size:
    #     #     agent.replay(batch_size)
    #     print("Observation: {}".format(e))


    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        reward_total = 0
        qvalue_total = 0
        reward = 0
        for time in range(1000):
            if(e%100 == 0):
                env.render()
            qval = agent.model.predict(state)
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            reward_total+=reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            qvalue_total = qval[0][action]
            print(qval)

            rewards.append(reward_total)
            qvalues.append(qval[0][action])

            qvalue_total = qval[0][action]

            if (done or time >= 999):
                avrewards.append(reward_total)
                avqvalues.append(qvalue_total)
                break
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e, EPISODES, reward_total, agent.epsilon))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 25 == 0:
            agent.save("./save/lunarlander.h5")
            plt.cla()
            plt.plot(avrewards, linewidth=1)
            plt.plot(avqvalues, linewidth=1)
            #plt.plot(np.array(qvalues)/len(rewards), linewidth=1)
          #  plt.plot(qvalues/ len(qvalues), linewidth=1)
            #plt.draw()
        plt.pause(0.00001)
