import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
#import gym
#import gym_anytrading
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd
import datetime
import random
import keras
from collections import deque
from tensorflow.keras import backend as K
from google.colab import files

def get_and_preprocess(way_to_df):
  df = pd.read_csv(way_to_df)
  df.columns = ['Date',
              'Open',
              'High',
              'Low',
              'Close',
              'Volume',
              'Close time',
              'Quote asset volume',
              'Number of trades',
              'Taker buy base asset volume',
              'Taker buy quote asset volume',
              'Ignore']

  df = df.drop(columns = ['Date', 'Close time',
              'Quote asset volume',
              'Number of trades',
              'Taker buy base asset volume',
              'Taker buy quote asset volume',
              'Ignore'])

  df['Buy'] = np.zeros(df.shape[0])
  df['Sell'] = np.zeros(df.shape[0])
  print('Size:', df.shape)
  print(df.head())
  return df

#Теперь все действия предсказывает исключительно обученная нейросеть
def get_action(state,
               epsilon,
               action_size):

  Q_values = target_model.predict(np.expand_dims(state, axis = 0))
  action_index = np.argmax(Q_values)

  if epsilon > final_epsilon:
    epsilon -= epsilon_decay_factor

  return epsilon, action_index




#отсутствует функция получения награды т.к. больше не требуется, сейчас она ведёт торги уже обучившись


def make_network(input_shape, action_size, learning_rate):

  state_input = Input(shape=(input_shape))
  x = Dense(512, activation='relu')(state_input)
  state_value = Dense(256, activation='relu')(x)
  state_value = Dense(1)(state_value)
  state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_size,))(state_value)
  action_advantage = Dense(256, activation='relu')(x)
  action_advantage = Dense(action_size)(action_advantage)
  action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(action_advantage)
  state_action_value = Add()([state_value, action_advantage])

  model = Model(state_input, state_action_value) #Создаем модель, которая принимает на вход состояние среды и возвращает все значения Q(s, a)
  model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate))


  return model


def add_to_memory(previous_info, current_info, reward, action_index):
  memory.append((previous_info, current_info, reward, action_index))


gamma = 0.95
observation_steps = 1000
target_model_update_frequency = 5000



initial_epsilon = 1
epsilon = initial_epsilon
final_epsilon = 0.01
epsilon_decay_steps = 200000
epsilon_decay_factor = (initial_epsilon - final_epsilon)/epsilon_decay_steps

timesteps_per_train = 100
learning_rate = 1e-4
batch_size = 32

maximum_memory_length = 40000

memory = deque([], maxlen = maximum_memory_length)
final_profit = 0

your_data = files.upload()#!!!
way_to_df = ''
for fn in your_data.keys():
  way_to_df = fn

df_test = get_and_preprocess(way_to_df)#!!!


current_df_test = df_test.drop(columns = ['Buy', 'Sell']).to_numpy()
current_state_test = np.zeros((current_df_test.shape))
current_state_test = current_df_test[0]
previous_state_test = current_state_test
print(current_state_test[0])
current_info_test = np.zeros((1, 3))
current_info_test.shape = -1
print(current_info_test.shape)
initial_balance_test = 100
current_info_test[1] = initial_balance_test #$
previous_info_test = current_info_test
action_list = ['Buy_Open', 'Buy_High', 'Buy_Low', 'Buy_Close', 'Sell_Open', 'Sell_High', 'Sell_Low', 'Sell_Close', 'Pass']

action_size = 9
info_shape = current_info_test.shape


target_model = make_network(info_shape, action_size, learning_rate)

target_model.load_weights('/content/target_crypto_model_5.h5') #1



for i in range(int(df_test.shape[0])):

    epsilon, action_index = get_action(previous_info_test, epsilon, action_size)


    if action_list[action_index] == 'Buy_Open':
      amount = current_info_test[1] /  current_state_test[0]
      current_info_test[2] += amount
      current_info_test[1] = 0



    elif action_list[action_index] == 'Buy_High':
      amount = current_info_test[1] /  current_state_test[1]
      current_info_test[2] += amount
      current_info_test[1] = 0


    elif action_list[action_index] == 'Buy_Low':
      amount = current_info_test[1] /  current_state_test[2]
      current_info_test[2] += amount
      current_info_test[1] = 0


    elif action_list[action_index] == 'Buy_Close':
      amount = current_info_test[1] /  current_state_test[3]
      current_info_test[2] += amount
      current_info_test[1] = 0


    elif action_list[action_index] == 'Sell_Open':
      current_info_test[1] += current_info_test[2] * current_state_test[0]
      current_info_test[2] = 0

    elif action_list[action_index] == 'Sell_High':
      current_info_test[1] += current_info_test[2] * current_state_test[1]
      current_info_test[2] = 0

    elif action_list[action_index] == 'Sell_Low':
      current_info_test[1] += current_info_test[2] * current_state_test[2]
      current_info_test[2] = 0

    elif action_list[action_index] == 'Sell_Close':
      current_info_test[1] += current_info_test[2] * current_state_test[3]
      current_info_test[2] = 0


    else:
      pass

    current_info_test[0] = (current_info_test[1] + current_info_test[2]) * current_state_test[3] - initial_balance_test
    current_state_test = current_df_test[i]

    previous_state_test = current_state_test
    previous_info_test = current_info_test

    final_profit += current_info_test[0]

print('Final profit: ', final_profit, '$')
