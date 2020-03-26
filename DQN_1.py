#import statements
import tensorflow as tf
from tensorflow import keras
import gym
import atari_py
from collections import deque
import numpy as np
import random
random.seed(0)

#some atari environments
env = gym.make('SpaceInvaders-v0') #spave invaders
#env = gym.make('Assault-v0') #assault

#environment processing network, you can use a CNN without pooling instead.
model = keras.Sequential()
model.add(keras.layers.Dense(32, input_shape=(2,) + env.observation_space.shape, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(env.action_space.n, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#DQN parameters
D = deque()
observetime = 500
epsilon = 0.7 #probability of agent doing a random move
gamma = 0.9 #how much the agent cares about future reward
mb_size = 50
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False

for t in range(observetime):
    Q = model.predict(state) #Q stands for quality
    action = np.argmax(Q) #action with highest q value
    observation_new, reward, info, done = env.step(action) #preform action
    obs_new = np.expand_dims(observation_new, axis=0) #measure reward
    state_new = np.append(np.expand_dims(obs_new, axis=0),state[:, :1, :], axis=1) #update
    D.append((state, action, reward, state_new, done))
    state = state_new
    
minibatch = random.sample(D, mb_size)
input_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros((input_shape))
targets = np.zeros((mb_size, env.action_space.n))
for i in range(0, mb_size):
    state = minibatch[i][0]
    action = minibatch[i][1]
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]
    
    
    #Bellman equation
    inputs[i:i+1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)
    
    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)
        
    model.train_on_batch(inputs, targets)
    
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    env.render()                    # Uncomment to see game running
    Q = model.predict(state)        
    action = np.argmax(Q)         
    observation, reward, done, info = env.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
    tot_reward += reward
print('Game ended! Total reward: {}'.format(reward))