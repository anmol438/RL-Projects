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
from tf_agents.utils.common import function, Checkpointer
from tf_agents.eval.metric_utils import log_metrics

from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import RMSprop

import imageio
from datetime import datetime
import logging
import warnings

tf.config.run_functions_eagerly(False)
logging.getLogger().setLevel(logging.INFO)
# warnings.filterwarnings('ignore')

# Configure logging to log to both console and file
log_file = 'training_log.txt'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s \n',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

def create_eval_video(test_gym_env, test_tf_env, policy, filename, num_episodes = 5, fps = 30, fire_after_life_lost = False):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
        time_step = test_tf_env.reset()
        lives = test_gym_env.gym.ale.lives()
        time_step = test_tf_env.step(1)
        video.append_data(test_gym_env.render())
        while not time_step.is_last():
            curr_lives = test_gym_env.gym.ale.lives()
            if fire_after_life_lost and (curr_lives < lives):
                time_step = test_tf_env.step(1)
            else:
                action_step = policy.action(time_step)
                time_step = test_tf_env.step(action_step.action)
            lives = curr_lives
            video.append_data(test_gym_env.render())

# Tried the TF graph version
def create_eval_video_tf_fn(test_gym_env, test_tf_env, policy, filename, num_episodes = 5, fps = 30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
        time_step = test_tf_env.reset()
        lives = test_gym_env.gym.ale.lives()
        time_step = test_tf_env.step(tf.constant([1],dtype='int64'))
        video.append_data(test_gym_env.render())
        loop_cond = lambda time_step, lives : tf.logical_not(time_step.is_last())
        def loop_body(time_step, lives):
            curr_lives = test_gym_env.gym.ale.lives()
            def true_branch(time_step):
                time_step = test_tf_env.step(tf.constant([1],dtype='int64'))
                return time_step
            def false_branch(time_step):
                action_step = policy.action(time_step)
                time_step = test_tf_env.step(action_step.action)
                return time_step

            time_step = tf.cond(
               tf.less(curr_lives,lives),
               lambda:true_branch(time_step),
               lambda:false_branch(time_step)
            )

            lives = curr_lives
            video.append_data(test_gym_env.render())
            return time_step, lives
        time_step, lives = tf.while_loop(loop_cond, loop_body, [time_step, lives])

def record_training():
    if ((train_step + training_video_length - 1)%training_video_interval == 0 or len(render_data) != 0):
        render_data.append(train_gym_env.render())

        if len(render_data) >= training_video_length:
            filename = f'training_video_step_{train_step.read_value()}' + ".mp4"
            with imageio.get_writer(filename, fps=30) as video:
                for data in render_data:
                    video.append_data(data)
            render_data.clear()

if __name__ == '__main__':
    
    start_time = datetime.now()

    env_name = 'BreakoutNoFrameskip-v4'
    max_ep_step = 27000 # max_ep_step*4 = ALE frames per episode

    # for epsilon greedy
    decay_steps = 2500 # decay_steps*collect_driver_steps*4 = ALE frames for decaying

    # for optimizer
    lr = 1e-4 # size of the steps in gradient descent
    rho = 0.95 # decay rate of the moving average of squared gradients
    epsilon = 1e-7 # Improves numerical stability

    rb_len = 10000
    collect_driver_steps = 4 # run 4 steps for each train step
    initial_driver_steps = 1000
    target_update = 20
    train_step = tf.Variable(0) # collect_driver_steps*4 = ALE frames per train step
    discount_factor = 0.99
    batch_size = 64
    training_iterations = 10000 # training_iterations*collect_driver_steps*4 number of ALE frames

    log_interval = training_iterations // 100
    eval_interval = training_iterations // 4

    training_video_interval = training_iterations // 4
    training_video_length = 250
    record_training_flag = True # whether to record training or not

    checkpoint_interval = 1000

    # Creating train and test env

    train_gym_env = suite_atari.load(
        env_name,
        max_episode_steps=max_ep_step,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4]
    )
    
    test_gym_env = suite_atari.load(
        env_name,
        max_episode_steps=max_ep_step,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4]
    )

    train_tf_env = TFPyEnvironment(train_gym_env)
    test_tf_env = TFPyEnvironment(test_gym_env)

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
    it = iter(dataset)

    # Time to train
    render_data = []
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

    # Load any Checkpointer if it exist

    train_checkpointer = Checkpointer(
        ckpt_dir='./checkpointer',
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step,
        train_metrics=train_metrics

    )
    if train_checkpointer.checkpoint_exists:
        train_checkpointer.initialize_or_restore()
        logging.info(f"Restored training from last checkpoint at step {train_step.read_value()}")
    else:
        logging.info("Starting training from scratch")

    time_step = train_tf_env.reset()
    for _ in range(training_iterations):
        time_step, __ = collect_driver.run(time_step)
        trajectories, ___ = next(it)
        train_loss = agent.train(trajectories)

        if train_step%checkpoint_interval == 0:
            train_checkpointer.save(train_step)

        if train_step%log_interval == 0:
            log_metrics(train_metrics, prefix=f'\n         Step : {train_step.read_value()}\n         Loss : {train_loss.loss}')
            
        if record_training_flag:
            record_training()

        if train_step%eval_interval == 0:
            create_eval_video(
                test_gym_env=test_gym_env,
                test_tf_env=test_tf_env,
                policy=agent.policy,
                filename=f'eval_video_step_{train_step.read_value()}',
                num_episodes=2,
                fps=30,
                fire_after_life_lost=True
            )

            # create_eval_video_tf_fn(
            #     test_gym_env=test_gym_env,
            #     test_tf_env=test_tf_env,
            #     policy=agent.policy,
            #     filename=f'eval_video_step_{train_step.read_value()}',
            #     num_episodes=2,
            #     fps=60
            # )

    end_time = datetime.now()
    logging.info(f'======= Finished Training in {end_time-start_time} =======')
