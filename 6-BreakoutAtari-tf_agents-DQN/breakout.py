import tensorflow as tf
import numpy as np

from tf_agents.environments import suite_atari
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent

from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import RMSprop


if __name__ == '__main__':

    max_ep_step = 1000 # max_ep_step*4 ALE frames
    collect_driver_steps = 4 # run 4 steps for each train step
    target_update = 1
    train_step = tf.Variable(0)
    discount_factor = 0.99

    # for epsilon greedy
    decay_steps = 2500 # decay_steps*collect_driver_steps = ALE frames

    # for optimizer
    lr = 1e-4 # size of the steps in gradient descent
    rho = 0.95 # decay rate of the moving average of squared gradients
    epsilon = 1e-7 # Improves numerical stability

    # Creating train env

    env_name = 'BreakoutNoFrameskip-v4'
    train_gym_env = suite_atari.load(
        env_name,
        max_episode_steps=max_ep_step,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4]
    )

    train_tf_env = TFPyEnvironment(train_gym_env)

    # Create a Q network

    normalized_layer = Lambda(lambda inp: tf.cast(inp, np.float32)/255.0) # normalizing from 0 to 1
    conv_layer = [(32,(8,8),4), (64,(4,4),2), (64,(4,4),1)] # 3 convolutional layers (filters, kernal size(height, width), stride)
    fc_layer = [512] # 1 dense layer with 512 neurons

    q_net = QNetwork(
        train_tf_env.observation_spec(),
        train_tf_env.action_spec(),
        preprocessing_layers=normalized_layer,
        conv_layer_params=conv_layer,
        fc_layer_params=fc_layer
    )

    # Create a DQN agent

    epsilon_greedy = PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=decay_steps,
        end_learning_rate=0.01
    )

    optimizer = RMSprop(
        learning_rate=lr,
        rho=rho,
        epsilon=epsilon,
        centered=True
    )

    loss_fn = Huber(reduction='none')

    agent = DqnAgent(
        train_tf_env.time_step_spec(),
        train_tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=lambda : epsilon_greedy(train_step),
        target_update_period=target_update,
        td_errors_loss_fn=loss_fn,
        train_step_counter=train_step,
        gamma=discount_factor
    )
    
    agent.initialize()
