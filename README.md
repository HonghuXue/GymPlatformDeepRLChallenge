#  Multi-Pass Deep Q-Networks

This repository includes a state-of-the-art DRL algorithm solution (MP-DQN)[[Bester et al. 2019]](https://arxiv.org/abs/1905.04388) for parameterised action space MDPs (PAMDP), after referring to multiple PAMDP RL algorithms PA-DDPG[[1]](#references), Q-PAMDP[[2]](#references), P-DQN[[3]](#references):



Multi-Pass Deep Q-Networks (MP-DQN) fixes the over-paramaterisation problem of P-DQN by splitting the action-parameter inputs to the Q-network using several passes (in a parallel batch). Split Deep Q-Networks (SP-DQN) is a much slower solution which uses multiple Q-networks with/without shared feature-extraction layers. A weighted-indexed action-parameter loss function is also provided for P-DQN.

## Additional Features from Honghu
This repository is based on the following implementation: https://github.com/cycraig/MP-DQN/tree/master, which includes the orignal MP-DQN implementation.

In additional to the original implementation, further improvements are integrated: **These improvements are in orthogonal directions and can be activated in a combinatorial manner.**

### (1) Double Learning for the Q-critic (DDQN). [[Hasselt et al. 2015]](https://arxiv.org/abs/1509.06461)
DDQN is introduced to cancel maximization bias. The key equation for TD-target $`y`$ goes as:
```math
    y = r + \gamma (1 - d) Q_{\phi'}(s', \text{argmax}_{a'}Q_{\phi}(s',a')),
```
where $`\phi`$ and $`\phi'`$ denote the parameters of running network and target network respectively.

### (2) Twin-Delayed DDPG (TD3) to replace hthe original module of DDPG-actor, where target policy smoothing and delayed policy updates are implemented. [[Fujimoto et al. 2018]](https://arxiv.org/pdf/1802.09477.pdf)

**target policy smoothing**: Actions used to form the Q-learning target are based on the target policy, $`\mu_{\theta_{\text{targ}}}`$, but with clipped noise added on each dimension of the action. After adding the clipped noise, the target action is then clipped to lie in the valid action range (all valid actions, $`a`$, satisfy $`a_{Low} \leq a \leq a_{High})`$. The target actions are thus: 

```math
    a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)
```

**clipped double Q-learning**:
In the original implementation, the key equation for TD-target $`y`$ goes as:
```math
    y = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_{i, \text{targ}}}(s', a'(s')),
```
However, I refer to a minimalistic implementation of DDQN instead of really using two running and target networks, as MP-DQN finally learns the Q-value in discrete action space. And the equation is as follows:
```math
    y = r + \gamma (1 - d) Q_{\phi'}(s', \mu_{\theta}(s')),
```
where $`\mu_{\theta}'`$ stands for the running actor network and $`a' = \mu_{\theta}(s')`$ represents the action that maximizes the Q-value of $`s'`$ in the running network. This shares the same idea of DDQN.


### (3) Implicit Quantile Network (IQN) to replace the Q-network with a distribution on Q estimates with a set of quantiles. [[Dabney et al. 2019]](https://arxiv.org/abs/1806.06923)



### (4) Noisy Network for Exploration <!---(Additionally decouples the noise scaling for training and acting. The training procedure features a linear decay schedule for noise, so that the training can be accelerated. However it doesn't degrade the exploration as the noise for acting still assumes the original/undecayed noise. Note the noisy network module replaces the original exploration schedule of decaying epsilon-greedy algorithm and ornstein noise applied to DDPG actor)--> [[Fortunato et al. 2017]](https://arxiv.org/abs/1706.10295)

In this implementation, the noisy network is applied to DDPG/TD3 actor and Q-network. In the IQN mode, noisy network is applied to DDPG/TD3 actor, quantile network, state-action embedding network, but excluding the cosine network. Theoretically, cosine network could also use noisy network.


### (5) Proportion-based Prioritized Experience Replay [[Schaul et al. 2015]](https://arxiv.org/abs/1511.05952)
The idea of PER is to prioritized sampling the experiences featuring large TD-loss, resulting in a new proposal distribution. Importance-sampling ratio is then included for each sample to make each sample is still sampled as if from a uniform distribution, i.e., target distribution. For a detailed understanding, please refer to https://danieltakeshi.github.io/2019/07/14/per/ .
In this implementation, the replay buffer size must be of the size $`2^n`$ (due to sum-tree structure), as the semi-roulette-wheel sampling strategy is applied to reduce the sampling variance, which is exactly suggested in the original work. 

The IS-ratio $`w`$ in the original work goes as:
```math
    w_{i} = \frac{\left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta}}{w_{max}},
```
where $`N`$ refers to the current experience replay size, and $`P(i)`$ represents the probability of sampling data point $`i`$ according to priorities and $`w_{max}`$ refers to the maiximal IS-ratio value among all the stored experience. 

However, I found IS-ratio could result in a slow learning and osciallation in Q-value estimates due to the boostrapping nature. Therefore, I set IS-ratio for each samples to just be $`1`$, i.e., without "IS-ratio integration".








## Dependencies

- Python 3.5+ (tested with 3.5 & 3.6 & 3.12)
- pytorch 2.2.0 (1.0+ should work but will be slower)
- gym 0.10.5
- numpy 1.26.4
- click 8.1.7 
- tensorboard 2.16.2

The code is successfully tested in win11.
<!---## Domains

The simplest installation method for the above OpenAI Gym environments is as follows:
```bash
pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
```

If something goes wrong, follow the installation instructions given by the repositories above. Note that gym-soccer has been updated for a later gym version and the reward function changed to reflect the one used in the code by Hausknecht & Stone [2016] (https://github.com/mhauskn/dqn-hfo). So use the one linked above rather than the OpenAI repository.-->

## Example Usage

**Learning from scratch**

It is recommeded to directly run *run_platform_pdqn.py* in your IDE in this implementation, since the click flags are configured to make it easier to run experiments and hyper-parameter searches in batches, which is better for scripts but makes it more annoying to type out.

**Load a trained Model**

please set the following parameters in the **click options** in *run_platform_pdqn.py*
```bash
"evaluation_mode = True", "seed = 5" and "load_model_idx = 60000"
```

<!---
To run vanilla P-DQN on the Platform domain with default flags:
```bash
python run_platform_pdqn.py 
```

SP-DQN on the Platform domain, rendering each episode:
```bash
python run_platform_pdqn.py--split True --visualise True --render-freq 1
```

MP-DQN on the Platform domain with four hidden layers (note no spaces) and the weighted-indexed loss function:
```bash
python run_platform_pdqn.py  --multipass True --layers [1024,512,256,128] --weighted True --indexed True
```
-->

The training stage is specified by the number of training episodes. After the training is completed, evaluation is performed by running 3000 episodes in the default configuration.


## Performance Curve

Maximual episodic return = 1

- Training Statistics (Seed 5 , IQN + DDQN + TD3 + PER + Noisy Net)

![Training Performance Seed 5](runs/seed_5.png)

- Training Statistics (Seed 4 , IQN + DDQN + TD3 + PER + Noisy Net)

![Training Performance Seed 4](runs/seed_4.png)

- Training Statistics (Seed 3 , IQN + DDQN + TD3 + PER + Noisy Net)

![Training Performance Seed 3](runs/seed_3.png)

- Training Statistics (Seed 2 , IQN + DDQN + TD3 + PER + Noisy Net)

![Training Performance Seed 2](runs/seed_2.png)


## Loss Function Monitoring

Monitor the loss function to avoid divergence

- quantile loss (IQN + DDQN + TD3 + PER + Noisy Net), showing convergence properties.

![quantile_huber_loss](runs/quantile_huber_loss.png) 

- Q-loss for the actor update (IQN + DDQN + TD3 + PER + Noisy Net), exhibiting convergence.

![actor-Q-loss](runs/Q_loss.png) 

- estimated Q-value that is maximized from the sampled batch during the training episode

![estimated_Q_max](runs/estimated_Q_max.png) 

The evolution of the estimated Q-values are reasonable as the episodic return lies between 0 and 1. When consiering the the discount factor = 0.99, the estimated Q-value should be within this range.

## Evaluation Animation

Trained policy after 20K episodes of training

![evaluation_20000epi](runs/evaluation_20000.gif)


Trained policy after ~2 million frames a.k.a 60K episodes of training

![evaluation_60000epi](runs/evaluation.gif)

References
----------

[1] [PA-DDPG](https://arxiv.org/abs/1511.04143)    
[2] [Q-PAMDP](https://arxiv.org/abs/1509.01644)  
[3] [P-DQN](https://arxiv.org/abs/1810.06394)   


