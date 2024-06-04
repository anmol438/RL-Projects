from tf_agents.environments import suite_atari
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4


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

    print(train_tf_env.current_time_step())
