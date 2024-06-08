import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
from tf_agents.policies.policy_saver import PolicySaver

from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import RMSprop

from datetime import datetime
import imageio
import logging


tf.config.run_functions_eagerly(False)
logging.getLogger().setLevel(logging.INFO)

# Configure logging to log to both console and file
log_file = 'training_log.txt'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s\n%(message)s \n',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

def create_eval_video(test_gym_env, test_tf_env, policy, filename, num_episodes = 5, fps = 30, fire_after_life_lost = False):
    logging.info(f'======= Evaluating agent and saving evaluation video at step {train_step.read_value()} =======')
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
            logging.info(f'======= Saving training video at step {train_step.read_value()} =======')
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
    decay_steps = 250000 # decay_steps*collect_driver_steps*4 = ALE frames for decaying

    # for optimizer
    lr = 1e-4 # size of the steps in gradient descent
    rho = 0.95 # decay rate of the moving average of squared gradients
    epsilon = 1e-7 # Improves numerical stability

    rb_len = 1000000
    collect_driver_steps = 4 # run 4 steps for each train step
    initial_driver_steps = 20000
    target_update = 2000
    train_step = tf.Variable(0) # collect_driver_steps*4 = ALE frames per train step
    last_train_step = train_step.value() # stores train_step from last checkpoint if it exist
    discount_factor = 0.99
    batch_size = 64
    max_training_iterations = 10000000 # = total number of iterations for training. max_training_iterations*collect_driver_steps*4 number of ALE frames
    segmented_iterations = 500000 # dividing the total iterations to run in small segments. e.g. 1 segment = 1/20 of total iterations, take around 8hrs on my system
    n_segment_runs = 1 # run a segmented iteration this number of times ( better to be greater than one )
    iterations = n_segment_runs*segmented_iterations # the train loop will also add any number of iterations left from from previous checkpoint because of any failure

    log_interval = segmented_iterations // 50
    eval_interval = segmented_iterations // 1

    training_video_interval = segmented_iterations // 1
    training_video_length = 1000
    record_training_flag = True # whether to record training or not

    checkpoint_interval = segmented_iterations // 5

    render_data = []
    avg_returns = []

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

    # Create the main collect driver to run during training

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

    # Load any Checkpointer if it exist

    train_checkpointer = Checkpointer(
        ckpt_dir='./checkpoint',
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        # replay_buffer=replay_buffer, # removed because of very large storage size
        global_step=train_step,
        train_metrics=train_metrics

    )
    if train_checkpointer.checkpoint_exists:
        train_checkpointer.initialize_or_restore().expect_partial()

    if train_step < max_training_iterations:
        if train_step == 0: # training from scratch
            logging.info("Starting training from scratch")
            initial_collect_policy = RandomTFPolicy(train_tf_env.time_step_spec(), train_tf_env.action_spec())

        else: # restored from a checkpoint
            logging.info(f"Restored training from last checkpoint at step {train_step.read_value()}")
            last_train_step = train_step.value()
            initial_collect_policy = agent.collect_policy
    
        # Create a policy saver
        policy_saver = PolicySaver(agent.policy)

        # Create a driver to pre populate the replay buffer before training

        initial_driver = DynamicStepDriver(
            train_tf_env,
            initial_collect_policy,
            observers=[rb_observer],
            num_steps=initial_driver_steps
        )

        # Time to train
        collect_driver.run = function(collect_driver.run)
        initial_driver.run = function(initial_driver.run)
        agent.train = function(agent.train)

        initial_driver.run()

        time_step = train_tf_env.reset()
        tot_iteration_to_loop = iterations - last_train_step%(-segmented_iterations) # second term for including any remaining iterations from previous checkpoint
        for iteration in range(tot_iteration_to_loop):
            time_step, __ = collect_driver.run(time_step)
            trajectories, ___ = next(it)
            train_loss = agent.train(trajectories)

            if train_step%log_interval == 0:
                avg_returns.append(train_metrics[2].result().numpy())
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

            if train_step%checkpoint_interval == 0:
                logging.info(f'======= Saving checkpoint at step {train_step.read_value()} =======')
                train_checkpointer.save(train_step)

            if train_step%segmented_iterations == 0:
                plt.plot(range(last_train_step, train_step.value(), log_interval), avg_returns)
                plt.xlabel('Iterations')
                plt.ylabel('Average Return')
                plt.savefig(f'Average_return_{train_step.read_value()}.png')
                plt.clf()

                avg_returns.clear()
                last_train_step = train_step.value()
                logging.info(f'======= A segment completed at step {train_step.read_value()} =======')

            if train_step>=max_training_iterations:
                break
            
        logging.info(f'======= Saving policy at step {train_step.read_value()} =======')
        policy_saver.save('./policy') # load the trained policy with tf_agents.policies.policy_loader.load


        end_time = datetime.now()
        logging.info(f'========================================================== Finished {iteration+1} Training iteration in: {end_time-start_time}. Total training steps: {train_step.read_value()} ==========================================================')
    
    else: # if train step exceed max training iterations
        logging.info("Training already completed. Load the checkpoint or policy to evaluate")
