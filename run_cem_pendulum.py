import pickle
import gym
import numpy as np

env = gym.make('Pendulum-v0')
theta = pickle.load(open('cem_pendulum_best.pkl', 'rb'))
theta = pickle.load(open('cem_pendulum_mean.pkl', 'rb'))

def get_action(state, theta):
  # input (1,3), first linear(/dense) layer = (3,8) with relu
  out = np.dot(state, np.reshape(theta[:24],(3,8)))
  out = np.maximum(out, 0.0)
  # second layer 8,1 with tanh
  out = np.dot(out, np.reshape(theta[24:],(8,1)))
  out = np.tanh(out) * 2.0
  return out

ended = False

state = env.reset()
env.render()
r = 0
while not ended:
  action = get_action(state, theta)
  (state, reward, ended, info) = env.step(action)
  r += reward
  env.render()

print(r)
