import random
import argparse
import gym
import numpy as np
from random import shuffle
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten

learning_rate = 0.003
learning_rate_decay = 1e-5
momentum = 0.9
gamma = 0.9
epsilon = 0.7
epsilon_decay = 3e-4
epsilon_min = 0.15
batch_size = 50
checkpoint = 100
iteration = 0
max_iteration = 10000

def shuffle_xy(x,y):
  z = list(zip(x,y))
  shuffle(z)
  x[:], y[:] = zip(*z)
  return (x, y)

class DeepQAgent(object):
  def __init__(self, input_shape, output_size):
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.momentum = momentum
    self.input_shape = input_shape
    self.output_size = output_size
    self._init_model()

  def _init_model(self):
    model = Sequential()
    model.add(Dense(24, input_shape=self.input_shape, kernel_initializer='truncated_normal', activation='relu'))
    model.add(Dense(24, kernel_initializer='truncated_normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='truncated_normal', activation='relu'))
    model.add(Dense(self.output_size, activation='linear'))
    sgd = SGD(lr=self.learning_rate, decay=self.learning_rate_decay, momentum=self.momentum, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    self.model = model

  def act(self, state):
    return self.model.predict(state)

  def save(self, name):
    self.model.save_weights(name)
    print("save: %s" %name)

  def load(self, name):
    self.model.load_weights(name)

  def train(self, batch_x, batch_y):
    self.model.train_on_batch(batch_x, batch_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    parser.add_argument('--mode', nargs='?', default='train', help='train or play the game')
    parser.add_argument('--weight', nargs='?', help='Sepcify a saved weights')
    parser.add_argument('--name', nargs='?', default='dqn', help='prefix for saved weights')
    args = parser.parse_args()

    train_x = []
    train_y = []
    step_hist = []

    env = gym.make(args.env_id)
    agent = DeepQAgent(env.observation_space.shape, env.action_space.n)
    if args.mode == 'train':
      while iteration < max_iteration:
        ep = max(epsilon * (1. / (1. + epsilon_decay * iteration)), epsilon_min)
        if len(train_x) > batch_size:
          train_x, train_y = shuffle_xy(train_x, train_y)
        while len(train_x) > batch_size:
          batch_x = train_x[:batch_size]
          batch_y = train_y[:batch_size]
          train_x = train_x[batch_size:]
          train_y = train_y[batch_size:]
          agent.train(np.array(batch_x), np.array(batch_y))

          iteration += 1
          if iteration % checkpoint == 0:
            print "checkpoint: %d" %iteration
            print "last 50 game steps: %d" %(sum(step_hist[-50:]) / 50)
            print "ep: %f" %ep
            agent.save("weights/%s_%d" %(args.name, iteration))

        state = env.reset()
        done = False
        step = 0
        while not done:
          step += 1
          if np.random.rand() <= ep:
            predict = agent.act(np.expand_dims(state, axis=0))
            action = np.random.randint(0, env.action_space.n, size=1)[0]
          else:
            predict = agent.act(np.expand_dims(state, axis=0))
            action = np.argmax(predict)
          (state_new, reward, done, info) = env.step(action)
          x = state
          train_x.append(state)
          y = predict[0]
          fr = np.max(agent.act(np.expand_dims(state_new, axis=0)))
          if done:
            y[action] = reward
          else:
            y[action] = reward + gamma * fr
          train_y.append(y)
          state = state_new
        step_hist.append(step)
    if args.mode == 'play':
      agent.load(args.weight)
      done = False
      state = env.reset()
      while not done:
        env.render()
        predict = agent.act(np.expand_dims(state, axis=0))
        action = np.argmax(predict)
        (state_new, reward, done, info) = env.step(action)
        state = state_new

    env.close()
