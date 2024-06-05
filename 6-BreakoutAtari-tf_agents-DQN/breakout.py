import tensorflow as tf
import numpy as np

from tf_agents.environments import suite_atari
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
from tf_agents.eval.metric_utils import log_metrics

from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import RMSprop

from datetime import datetime
import logging
logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    
    start_time = datetime.now()

    env_name = 'BreakoutNoFrameskip-v4'
    max_ep_step = 1000 # max_ep_step*4 = ALE frames per episode

    # for epsilon greedy
    decay_steps = 2500 # decay_steps*collect_driver_steps*4 = ALE frames for decaying

    # for optimizer
    lr = 1e-4 # size of the steps in gradient descent
    rho = 0.95 # decay rate of the moving average of squared gradients
    epsilon = 1e-7 # Improves numerical stability

    rb_len = 10000
    collect_driver_steps = 4 # run 4 steps for each train step
    initial_driver_steps = 1000
    target_update = 1
    train_step = tf.Variable(0) # collect_driver_steps*4 = ALE frames per train step
    discount_factor = 0.99
    batch_size = 64
    training_iterations = 1000 # training_iterations*collect_driver_steps*4 number of ALE frames

    # Creating train env

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

    # Create a Replay Buffer and Observers

    replay_buffer = TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_tf_env.batch_size,
        max_length=rb_len
    )

    rb_observer = replay_buffer.add_batch

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric()
    ]

    # Create a driver to pre populate the replay buffer

    initial_driver = DynamicStepDriver(
        train_tf_env,
        RandomTFPolicy(train_tf_env.time_step_spec(), train_tf_env.action_spec()),
        observers=[rb_observer],
        num_steps=initial_driver_steps
    )

    initial_driver.run()

    # Create the main collect driver

    collect_driver = DynamicStepDriver(
        train_tf_env,
        agent.collect_policy,
        observers=[rb_observer]+train_metrics,
        num_steps=collect_driver_steps
    )

    # Create a dataset to sample trajectories

    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2,
        num_parallel_calls=3
    ).prefetch(3)

    # Time to train
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)
    
    time_step = None
    it = iter(dataset)
    for _ in range(100):
        time_step, __ = collect_driver.run(time_step)
        trajectories, ___ = next(it)
        train_loss = agent.train(trajectories)
        
        if train_step%10 == 0:
            log_metrics(train_metrics, prefix=f'       Step : {train_step.numpy()}')
            print(f'                 Loss : {train_loss.loss.numpy()}\n')

    end_time = datetime.now()
    print(f'======= Finished Training in {end_time-start_time} =======')
