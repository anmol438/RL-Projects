# This algo based on the observation that target model is prone to overestimate the Q-values
# If the actions are equally good, because of approximation one Q values will be greater.
# So the online model is used to select the best actions for next states.
# And then use target model to select Q-values for corresponding best action.

import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

start_time = datetime.now()

env = gym.make("CartPole-v1")


def epsilon_greedy_policy(state, epsilon):

    if np.random.rand() < epsilon:
        return np.random.randint(2) #exploring other possibilities

    Q_values = model.predict(state[None,:], verbose=0)
    return np.argmax(Q_values[0]) # chosing actiong with max Q value

def get_past_experiences(batch_size):
    ind = np.random.randint(len(replay_buffer), size=batch_size) # selecting batch_size number of past env experience
    states, actions, next_states, rewards, terms, truncs = [
        np.array([replay_buffer[i][j] for i in ind]) for j in range(6)]
    
    return states, actions, next_states, rewards, terms, truncs

def training_step(batch_size):
    past_epx = get_past_experiences(batch_size)
    states, actions, next_states, rewards, terms, truncs = past_epx
    # calculating Q_target values for the actions selected by agent:

    # ======== changes in the Algo, use target model =========
    next_Q_values = model.predict(next_states, verbose=0) # cal Q values for next state-action pair
    next_best_action = np.argmax(next_Q_values, axis=1) # selecting best actions for next-state from online model
    next_mask = tf.one_hot(next_best_action, n_output).numpy() # mask fot next best action to filter best Q values of target model
    next_best_Q_values = (target.predict(next_states, verbose=0)*next_mask).sum(axis=1)
    # ======================================
    Q_target = rewards + (1 - (terms + truncs)) * discount_factor * next_best_Q_values # there will not be next state for terminal step
    Q_target = Q_target[:,None]
    # create a mask from actions to filter the approximated Q values from the model to consider only taken actions
    mask = tf.one_hot(actions, n_output)

    with tf.GradientTape() as tape: 

        Q_values = model(states)
        # all_Q.append(Q_values)
        Q_values = tf.reduce_sum(Q_values*mask, axis=1, keepdims=True) # zero out the other actions and collapsing the each array to single value 

        loss = tf.reduce_mean(loss_fn.call(Q_target, Q_values))
        losses.append(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # gradient descent

def play_one_step(env, state, epsilon=0):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, term, trunc, info = env.step(action)

    replay_buffer.append((state, action, next_state, reward, term, trunc))

    return next_state, reward, term, trunc, info

if __name__ == '__main__':

    batch_size = 128
    n_episode = 1500
    n_output = env.action_space.n
    discount_factor = 0.95
    lr = 0.001
    loss_fn = MeanSquaredError('sum_over_batch_size', 'mean_squared_error')
    optimizer = Adam(learning_rate=lr)
    replay_buffer = deque(maxlen=10000)
    losses = []
    ep_reward = []
    all_Q = []

    model = Sequential([
        Dense(32, activation='elu', input_shape=[4]),
        Dense(32, activation='elu'),
        Dense(n_output)
    ])

    target = clone_model(model)
    target.set_weights(model.get_weights())

    for episode in range(n_episode):
        obs = env.reset()[0]
        curr_rew = 0
        for step in range(250):
            epsilon = max(1 - episode/1600 , 0.01) # gradually decreasing epsilon to 0.01 (reducing exploring)
            obs, reward, term, trunc, info =  play_one_step(env, obs, epsilon)
            curr_rew += reward
            if term or trunc:
                break
        if episode%50==0:
            print(f'Episode: {episode},     Reward: {curr_rew}')
            target.set_weights(model.get_weights())
        ep_reward.append(curr_rew)
        if episode >= 1.5*batch_size: # let the replay buffer populated with enough experiences
            training_step(batch_size)

    model.save('deep-q-learning-model.keras')

    pd.DataFrame(np.array(losses)).plot().get_figure().savefig('losses.png')
    pd.DataFrame(ep_reward).plot().get_figure().savefig('rewards.png')
    env.close()


    end_time = datetime.now()
    print(f'=== Time Elapsed : {end_time-start_time} ===')
    