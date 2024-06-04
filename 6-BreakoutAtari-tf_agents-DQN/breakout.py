import tensorflow as tf
import numpy as np

from tf_agents.environments import suite_atari
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.networks.q_network import QNetwork
from tensorflow.keras.layers import Lambda


if __name__ == '__main__':

    max_ep_step = 1000 # max_ep_step*4 ALE frames

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
