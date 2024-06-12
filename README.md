# RL-Projects

## 1. Policy Gradients

* I have used CartPole gym enviroment to implement PG algorithm.  
* PG algorithms optimize the policy by following the gradients towards higher rewards. I used the famous **Reinforce Algorithm**.  
* Here a neural network first explore the environment by predicting actions and also compute the gradients.  
* After running several episode, we calculate each action's advantage for each episodes so every action has the weight according to the future rewards.  
* Then compute the reward weighted mean for the gradient vectors for each training variable.  
* Finally perform the gradient descent step to update the variables.
* It took around **100 training** iteration to train the model.
* This algorithm is easy to implement but it is difficult to scale for complex environments.
  
## 2. Deep Q Learning

* This algorithm is based on the concept of **Markov Decision Process** which is a stochastic process.
* Along with Bellman's equation, we can compute **State value (V)** and **State-action value (Q)**
* Given a policy, V is good to evaluate the policy, but it does not give the best policy for the agent.
* With the Q values we can get the optimal policy for the agent.
* In the Deep Q Learning, we try to train a network which gives the best Q values to perform an action for a given state.
* While we explore the environment with **epsilon greedy policy**, the agent saves every experience (current observations and the next state) to the **replay buffer**.
* During each iteration the agent is trained with a sample of experiences from replay buffer. We can compute the target Q value using Bellman's equation, then predict the model's Q value and calculate the loss.
* Compute and apply the gradients to update the variables.
* It took around 2000 iteration to train.

![img](./2-Deep%20Q%20Learning/rewards.png "DQL Rewards")
* As it can be seen in the above image the rewards reaches to 250 which is max in this env, but there is unstability as the reward suddenly drops sometimes. This is called **Catastrophic forgetting** and it is because of what agent learned in one part of env may not work in another part of env.
* To overcome this we can use large replay buffers and small learning rates.
* The problem with this algorithm is that the same model which is being trained is used to predict the next Q values for the calculation of Q target in Bellman's equation. This creates a kind of feedback loop which may diverge, oscillate or freeze.

## 3. DQL with fixed Q targets

* This algorithm is a little improvement over basic DQL.
* Here two models are used - **Online Model** and **Target Model**
* Online model is the one which is being trained and to predict the Q values.
* Target model is introduced just to calculate the Q target values.
* It is a clone of the online model, but the weights of online model are copied at regular intervals while training.
  
![img](./3-DQL%20with%20fixed%20Q%20targets/rewards.png "DQL with fixed Q targets Rewards")
* Since the target model is update much less often so the feedback loop is damped.

## 4. Double DQN

* This is an also another improvement to stabilize the training and increase the performance.
* This update is based on the observation the target model overestimate the Q target values because they are approximation and for equally good action, one Q value will be greater than other and overestimated.
* So online model is first used to select best action corresponding to max Q values for next state and then target model is used to select the Q values corresponding to that actions.

![img](./4-Double%20DQN/rewards.png "DQL with fixed Q targets Rewards")
* This algorithm trained much faster and is much stable.

# TF Agents

The TF-Agents library is Reinforcement Learning library based on Tensorflow. It implements the above discussed algorithms along with many more. It also provides the various RL components which are efficient, scalable and customizable. It provide a training architecture to implement any algorithm with tf agents.  
The architecture is divided into two parts that run in parallel:
* Collection
  + Collect Policy
  + Collect Driver
  + Environments
  + Observers
  + Replay Buffers
* Training
  + Agent
  + Networks
  + Dataset

### The architecture is usually setup like this with DQN

* A **DQN** agent will be trained for a environment with **Deep Q Network** in the backend.
* Tf agents provides a **uniform replay buffer** which will be populated with experiences during environment exploration by the **collect driver**.
* Between replay buffer and collect driver there is a **observer** which will write the experiences coming from driver to the replay buffer.
* For collect driver, we have **Dynamic Step Driver** and **Dynamic Episode Driver** which run a step and episode in the environment respectively.
* A **Dataset** is defined to collect a sample of experiences from the replay buffer and pass to the agent for training.
* To train the agent, for each training step, a collect driver runs, experiences are fetched from dataset and the agent is trained with these experiences.
* To observe the training process, we can log **tf_metrics** or the rewards by testing the agent at regular intervals.
* The choice of the driver depends on the specific training goals. 
* Step driver has fine-grained control and has consistent data flow which allows the number of steps to be collected and it is useful for steady and incremental training.
* Episode driver is useful for learning policies that requires full episode but it can be unstable as full episode data is collected in one training step.

## 5. CartPole Env with DQN

The agent is trained for CartPole-v1 environment with tf_agents.
### With Dynamic Step Driver

![img](./5-CartPole-tf_agents-DQN/with_step_driver/Averages.png)

[Watch the video](https://github.com/anmol438/RL-Projects/assets/50985412/db72afa5-7279-4e5a-9746-0e4b94fc0497)

### With Dynamic Episode Driver

![img](./5-CartPole-tf_agents-DQN/with_episode_driver/Averages.png)

https://github.com/anmol438/RL-Projects/assets/50985412/c4608f7f-0065-4d5a-b531-7c95795fb32d

* The agent is well trained and performed perfect.
* If we look at average return, with the step driver the training is smooth while with episode driver it is unsteady.


## 6. Atari - Breakout Env

* This env is not simple as CartPole env. So I will explain it in little detail.
* The Atari env needs to be installed separately for gym. Please read **"./6-BreakoutAtari-tf_agents-DQN/readme.md"** for installation help.
* The raw Breakout env looks like this:  
  ![img](./6-BreakoutAtari-tf_agents-DQN/raw_env.png)
* For efficient training, the grayscaled and framestacked environment is good. After preprocessing it looks like this:  
  ![img](./6-BreakoutAtari-tf_agents-DQN/stacked_env.png)
* Four colours represent four consecutive different frames.
* The **Q-network** is set up with a normalized layer, convolutional layer and then a dense layer.
* This network is feed in the **DQN agent** with **epsilon greedy policy**.
* **Dynamic Step Driver** is used to explore the environment and collect trajectories to **replay buffer**.
* To make the agent perform excellently, it took around 4 days of training for **4 million** steps.
* I also used the **Checkpoint** which stores the trained state to the hard disk. In case something wrong happens during the training, it can be restores from the last checkpoint.
* **Policy Saver** can be used to play in the environment with the trained agent's policy.
* I regularly observed the training and also evaluated the agent during the training.
* Below are the observations:
#### Initial Phase:
* Average Return  
  ![img](./6-BreakoutAtari-tf_agents-DQN/Average_return_1000000.png)
* Training Video  
  <video controls src="6-BreakoutAtari-tf_agents-DQN/training_video_step_500000.mp4" title="Title"></video>
* Agent Evaluation Video:  
  <video controls src="6-BreakoutAtari-tf_agents-DQN/eval_video_step_500000.mp4" title="Title"></video>
#### At 4 million steps:
* Average Return  
  ![img](./6-BreakoutAtari-tf_agents-DQN/Average_return_4000000.png)
* Training Video  
  <video controls src="6-BreakoutAtari-tf_agents-DQN/training_video_step_4000000.mp4" title="Title"></video>
* Agent Evaluation Video:  
  <video controls src="6-BreakoutAtari-tf_agents-DQN/eval_video_step_4000000.mp4" title="Title"></video>

* The agent started learning from initial phase that it needs to move the paddle to make the ball break the bricks.
* But after a long training it is fascinating to see that the agent is achieving superhuman capability by learning that it can dig up a tunnel through the brick walls for the maximum score and do nothing when the ball goes above the brick walls.
* It's still need more training to 100% break the bricks.