# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

EPISODES = 1000000
POPULATION_SIZE = 100
MUTATION_RATE = 0.5
MUTATION_SIZE = 0.5
POPULATION_WINNERS = 0.1

class DQNAgent:
    def __init__(self, _state_size, _action_size):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = 0.01
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #print(model.summary())
        return model


    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

def mutateAgent(original, newAgent):
    for i,layer in enumerate(original.model.layers):
        newWeights = []
        for j,weights in enumerate(layer.get_weights()):
            newWeights.append(weights)
            for k,subweights in enumerate(weights):
                if type(subweights) is np.ndarray:
                    for h,individual_weight in enumerate(subweights):
                        if(random.random() < MUTATION_RATE):
                            adjustment = 2 * (random.random() -0.5)* MUTATION_SIZE
                            newWeights[j][k][h] += adjustment
        newAgent.model.layers[i].set_weights(newWeights)  # [j][k][h] = 0

def mutatePopulation(population):
    new_population = population.copy()
    for index in range(int(POPULATION_SIZE * POPULATION_WINNERS)):
        new_population[index + int(POPULATION_SIZE * POPULATION_WINNERS)].model.set_weights( new_population[index].model.get_weights())

    for index in range(int(POPULATION_SIZE * POPULATION_WINNERS),POPULATION_SIZE):
        winner_index = int(POPULATION_SIZE * POPULATION_WINNERS * random.random())
        mutateAgent(new_population[winner_index], new_population[index])
    return new_population

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, '/tmp/LunarLander-v2', force=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    #agent = DQNAgent(state_size, action_size)
    #agent.load("./save/cartpole-master.h5")
    done = False

    population = []
    winners_average = 0
    average_list = []
    average_reached = 0

    plt.plot(average_list)

    for m in range(POPULATION_SIZE):
        population.append(DQNAgent(state_size, action_size))
        #rewards.append(0)
        print ("Created agent: {}\r".format(m)),


    for e in range(EPISODES):
        rewards = [0] * POPULATION_SIZE
        for index, agent in enumerate(population):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(2000):
                #if (e%50 == 0):
                   #env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                rewards[index] += reward
                if done:
                    break

        results = list(zip(rewards, population))
        results.sort(key=lambda t: t[0], reverse=True)

        population_next = [None] * POPULATION_SIZE
        for index in range(int(POPULATION_SIZE)):
            population_next[index] = results[index][1]
        population = mutatePopulation(population_next)

        winners_average = 0
        for index in range(int(POPULATION_SIZE*POPULATION_WINNERS)):
            winners_average+= results[index][0]
        winners_average = winners_average/ (POPULATION_SIZE * POPULATION_WINNERS)
        print("episode: {}/{} best: {} average top {}%: {} average total: {}".format(e * POPULATION_SIZE, EPISODES, results[0][0], 100 * POPULATION_WINNERS, winners_average, sum(rewards) / POPULATION_SIZE))

        average_list.append(sum(rewards)/POPULATION_SIZE)

        if(winners_average >= 150 and average_reached < 150):
            MUTATION_SIZE = MUTATION_SIZE * 0.5
            MUTATION_RATE = MUTATION_RATE * 0.5
            average_reached = 150
            print("Reached average: 150")

        if(winners_average >= 250 and average_reached < 250):
            MUTATION_SIZE = MUTATION_SIZE * 0.5
            MUTATION_RATE = MUTATION_RATE * 0.5
            average_reached = 250
            print("Reached average: 250")

        # plt.close()
        # plt.plot(average_list)
        # plt.pause(0.00005)
        # plt.draw()

        if(sum(rewards)/POPULATION_SIZE >= 215):
            env.close()
            gym.upload('/tmp/LunarLander-v2', api_key='sk_QkSgKzDQWWAOQ0nOwigiQ')
            break