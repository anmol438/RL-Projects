import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import suite_gym, wrappers
from tf_agents.environments import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import PolynomialDecay
from tf_agents.utils import common
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.greedy_policy import GreedyPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.metrics import tf_metrics
from tensorflow.python.data.ops.map_op import _MapDataset
from tf_agents.policies.policy_saver import PolicySaver
import imageio

from datetime import datetime

tf.config.run_functions_eagerly(False)


def cal_avg_rewards(env : TFPyEnvironment, policy : GreedyPolicy, n_episodes : int = 5) -> float:
    tot_rewards = 0.0

    for _ in range(n_episodes):
        time_step = env.reset()
        curr_reward = 0.0
        while not time_step.is_last().numpy()[0]:
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            curr_reward += time_step.reward
        tot_rewards += curr_reward
    
    return (tot_rewards / n_episodes).numpy()[0]


def train_agent(n_iterations : int, test_env : TFPyEnvironment,  agent : DqnAgent, driver : DynamicEpisodeDriver, dataset : _MapDataset, log_interval :int = 200) -> tuple[list,list]:
    
    driver.run = common.function(driver.run)
    agent.train = common.function(agent.train)

    avg_returns = []
    avg_rewards = []
    time_step = None
    agent.train_step_counter.assign(0)
    iterator = iter(dataset)
    for _ in range(n_iterations):
        time_step, __ = driver.run(time_step)
        trajectories, ___ = next(iterator)
        train_loss = agent.train(trajectories)

        if agent.train_step_counter%log_interval == 0:
            loss = train_loss.loss
            avg_return = train_metrics[0].result().numpy()
            
            avg_reward = cal_avg_rewards(test_env, agent.policy, 2)
            avg_returns.append(avg_return)
            avg_rewards.append(avg_reward)

            print(f'trainstep: {agent.train_step_counter.read_value()} Ep: {train_metrics[1].result().numpy()} envstep: {train_metrics[2].result().numpy()} Avg return: {avg_return} Avg reward: {avg_reward}')

    return avg_returns, avg_rewards

def create_policy_eval_video(test_py_env : wrappers.TimeLimit, test_env : TFPyEnvironment, policy : GreedyPolicy, filename : str, num_episodes : int = 5, fps : int = 30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = test_env.reset()
      video.append_data(test_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = test_env.step(action_step.action)
        video.append_data(test_py_env.render())
#   return embed_mp4(filename)


if __name__ == '__main__':

    start_time = datetime.now()

    # Hyperparameters
    lr = 1e-3
    train_step_counter = tf.Variable(0)
    discount_factor = 1
    target_model_update = 1
    rb_max_len = 100000
    training_batch_size = 64
    driver_steps = 1
    n_iterations = 30000
    log_interval = int(0.01*n_iterations)

    # functions
    optimizer = Adam(learning_rate=lr)
    epsilon_greedy = PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=2000,
        end_learning_rate=0.01,
        power=1.0
    )
    loss_fn = common.element_wise_squared_loss

    # Create two environments
    env_name = 'CartPole-v1'

    train_gym_env = suite_gym.load(env_name)
    test_gym_env = suite_gym.load(env_name)

    # Wrap with tfpy environment
    train_env = TFPyEnvironment(train_gym_env)
    test_env = TFPyEnvironment(test_gym_env)

    # Create a deep Q network
    q_net = QNetwork(train_env.observation_spec(), train_env.action_spec())

    # Create a DQN agent
    agent = DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=lambda: epsilon_greedy(train_step_counter),
        target_update_period=target_model_update,
        td_errors_loss_fn=loss_fn,
        gamma=discount_factor,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    # Create a replay buffer to store trajectories
    replay_buffer = TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=rb_max_len
    )

    # Create a observer to write trajectories in replay buffer
    rb_observer = replay_buffer.add_batch

    # Create a dataset to fetch trajectories from replay buffer for collect driver
    dataset = replay_buffer.as_dataset(
        sample_batch_size=training_batch_size,
        num_steps=2
    )

    # Create a initial collect step driver to prepopulate the replay buffer with random collect policy
    initial_collect_driver = DynamicEpisodeDriver(
        env=train_env,
        policy=RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()),
        observers=[rb_observer],
        num_episodes=100
    )
    # Run the initial collect driver to prepopulate
    initial_collect_driver.run()

    # Create metrics list to observe the training progress
    train_metrics = [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps()
    ]

    # Create the main collect driver to explore the environment and collect trajectories
    collect_driver = DynamicEpisodeDriver(
        env=train_env,
        policy=agent.collect_policy,
        observers=[rb_observer] + train_metrics,
        num_episodes=driver_steps
    )

    # training the agents
    avg_returns, avg_rewards = train_agent(
        n_iterations=n_iterations,
        test_env=test_env,
        agent=agent,
        driver=collect_driver,
        dataset=dataset,
        log_interval=log_interval
    )

    # issue with some package versions
    # policy_saver = PolicySaver(agent.policy)

    # Saving plots
    interval = range(0, n_iterations, log_interval)

    plt.plot(interval, avg_returns)
    plt.plot(interval, avg_rewards)

    plt.legend(['Average Return', 'Average Rewards'])

    plt.xlabel('Iterations')
    plt.ylabel('Averages')
    plt.savefig('Averages.png')


    end_time = datetime.now()

    print(f'====> Training ended : {end_time-start_time}')

    print('====> Exploring environment with trained agent and saving in a video')
    create_policy_eval_video(test_gym_env, test_env, agent.policy, 'trained_agent')

    print('====> Video saved')
