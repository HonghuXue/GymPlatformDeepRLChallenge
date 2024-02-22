import os
import click
import time
import gym
import gym_platform
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
import numpy as np
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from torch.utils.tensorboard import SummaryWriter

def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, agent, visualise, episodes=1000):
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            if visualise:
                env.render()
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    return np.array(returns)


@click.command()
@click.option('--seed', default=4, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=100, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=95000, help='Number of episodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float) # HH: Changed from 0.9 to 0.99
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=10000, help='Number of transitions required to start learning.',
              type=int)  # HH: Changed from 500 to 10000
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=131072, help='Replay memory size in transitions.', type=int) # HH: changed from 10000 to 500000
@click.option('--epsilon-steps', default=20000, help='Number of episodes over which to linearly anneal epsilon.', type=int) # HH: changed from 1000 to 5000
@click.option('--epsilon-final', default=0.05, help='Final epsilon value.', type=float) # HH: to be changed from 0.01
@click.option('--tau-actor', default=0.0025, help='Soft target network update averaging factor.', type=float) # tau_actor too  HH: changed from 0.1 to 0.001
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=5e-4, help="Q network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too  , HH: 0.001 changed to 0.0001
@click.option('--learning-rate-actor-param', default=1e-4, help="Actor network learning rate.", type=float)  # 0.00001 actor-param learning rate should be smaller than actor lr.
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[32,32]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption) # HH: [128,]
@click.option('--save-freq', default=20000, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--render-freq', default=50000, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="PDDQN", help="Prefix of output files", type=str)
@click.option('--train_interval', default=16, help="Double Learning for updating Q-value", type=int)
@click.option('--ddqn', default=True, help="Double Learning for updating Q-value", type=bool)
@click.option('--delayed_policy_update', default=2, help="Enable delayed policy update in TD3", type=int)
@click.option('--td3_target_policy_smoothing', default=False, help="Enable policy noise injection in TD3", type=bool)
@click.option('--td3_policy_noise_std', default=0.2, help="standard deviation of the injected TD3 policy noise", type=float)
@click.option('--td3_policy_noise_clamp', default=0.5, help="maximum of the injected TD3 target policy smoothing", type=float)
@click.option('--per', default=True, help="prioritized experience replay", type=bool)
@click.option('--per_no_is', default=True, help="cancel the IS weights in PER", type=bool)
@click.option('--noisy_network', default=True, help="noisy network for exploration", type=bool)
@click.option('--noisy_network_noise_decay', default=False, help="noise linear decay", type=bool) # seemingly only degrade the performance!
@click.option('--noisy_net_noise_initial_std', default=1, help="noisy network noise initial std", type=float)
@click.option('--noisy_net_noise_final_std', default=0.001, help="noisy network noise final std", type=float)
@click.option('--noisy_net_noise_decay_step', default=1, help="noisy network noise std linear decay step", type=float)
@click.option('--iqn', default=True, help="implicit quantile network", type=bool)
@click.option('--iqn_embedding_size', default=64, help="IQN embedding size for tau and (s,a)", type=int)
@click.option('--iqn_quantile_num', default=8, help="IQN quantile number", type=int)
@click.option('--iqn_num_cosines', default=64, help="IQN cosine number", type=int)
@click.option('--iqn_embedding_layers', default='[32]', help='IQN embedding network', cls=ClickPythonLiteralOption)
@click.option('--iqn_quantile_layers', default='[32,32]', help='IQN quantile network', cls=ClickPythonLiteralOption)
@click.option('--evaluation_mode', default=False, help='Directly load the trained models for evaluation', type=bool)
@click.option('--load_model_idx', default=60000, help='load the i-th modelUpdate for evaluation, only valid if evaluation_mode is True', type=int)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, train_interval, ddqn, delayed_policy_update,
        td3_target_policy_smoothing, td3_policy_noise_std, td3_policy_noise_clamp, per, per_no_is, noisy_network,
        noisy_network_noise_decay, noisy_net_noise_initial_std, noisy_net_noise_final_std, noisy_net_noise_decay_step,
        iqn, iqn_embedding_size, iqn_quantile_num, iqn_num_cosines, iqn_embedding_layers, iqn_quantile_layers,
        evaluation_mode, load_model_idx):

    writer = SummaryWriter('../runs/')

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)
    assert not (save_frames and visualise)
    if visualise:
        assert render_freq > 0
    if save_frames:
        assert render_freq > 0
        vidir = os.path.join(save_dir, "frames")
        os.makedirs(vidir, exist_ok=True)

    env = gym.make('Platform-v0')
    initial_params_ = [3., 10., 400.]
    if scale_actions:
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
        initial_params_ = [arr.item() for arr in initial_params_]
    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)

    dir = os.path.join(save_dir,title)
    env = Monitor(env, directory=os.path.join(dir,str(seed)), video_callable=False, write_upon_reset=False, force=True)
    env.seed(seed)
    np.random.seed(seed)

    print(env.observation_space)

    from agents.pdqn import PDQNAgent
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent
    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        agent_class = SplitPDQNAgent
    elif multipass:
        agent_class = MultiPassPDQNAgent
    agent = agent_class(
                       env.observation_space.spaces[0], env.action_space,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       weighted=weighted,
                       average=average,
                       random_weighted=random_weighted,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer,
                                     'noisy_network': noisy_network,
                                     'noisy_network_noise_decay': noisy_network_noise_decay,
                                     'IQN_embedding_size': iqn_embedding_size,
                                     'num_cosines': iqn_num_cosines,
                                     'iqn_embedding_layers': iqn_embedding_layers,
                                     'iqn_quantile_layers': iqn_quantile_layers,
                                     'noisy_net_noise_initial_std': noisy_net_noise_initial_std,
                                     'noisy_net_noise_final_std': noisy_net_noise_final_std,
                                     'noisy_net_noise_decay_step': noisy_net_noise_decay_step
                                     },
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,
                                           'noisy_network': noisy_network,
                                           'noisy_network_noise_decay': noisy_network_noise_decay,
                                           'noisy_net_noise_initial_std': noisy_net_noise_initial_std,
                                           'noisy_net_noise_final_std': noisy_net_noise_final_std,
                                           'noisy_net_noise_decay_step': noisy_net_noise_decay_step
                                           },
                       zero_index_gradients=zero_index_gradients,
                       seed=seed,
                       train_interval=train_interval,
                       DDQN=ddqn,
                       delayed_policy_update = delayed_policy_update,
                       td3_target_policy_smoothing = td3_target_policy_smoothing,
                       td3_policy_noise_std = td3_policy_noise_std,
                       td3_policy_noise_clamp = td3_policy_noise_clamp,
                       PER = per,
                       per_no_is = per_no_is,
                       noisy_network = noisy_network,
                       noisy_network_noise_decay = noisy_network_noise_decay,
                       iqn = iqn,
                       iqn_quantile_num = iqn_quantile_num,
                       tensorboard_writer = writer
                       )

    if initialise_params:
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        #D:\Job Application\InstaDeep\Program_interview\pythonProject\MP-DQN-master\run_platform_pdqn.py:182: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
        for a in range(env.action_space.spaces[0].n):
            print(a, initial_bias.shape, initial_params_[a])
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)
    max_steps = 1500 # HH: changed from 250 to 1500
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    # agent.epsilon_final = 0.
    # agent.epsilon = 0.
    # agent.noise = None
    if not evaluation_mode:
        for i in range(episodes):
            if save_freq > 0 and save_dir and i % save_freq == 0:
                agent.save_models(os.path.join(save_dir, str(i)))
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32, copy=False)
            if visualise and i % render_freq == 0:
                env.render()

            act, act_param, all_action_parameters = agent.act(state)
            # print(act, act_param, all_action_parameters)
            action = pad_action(act, act_param)
            episode_reward = 0.
            agent.start_episode()
            for j in range(max_steps):

                ret = env.step(action)
                (next_state, steps), reward, terminal, _ = ret
                next_state = np.array(next_state, dtype=np.float32, copy=False)
                # print('Done : {}'.format(terminal))
                next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
                next_action = pad_action(next_act, next_act_param)
                agent.step(state, (act, all_action_parameters), reward, next_state,
                           (next_act, next_all_action_parameters), terminal, steps)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
                action = next_action
                state = next_state

                episode_reward += reward
                if visualise and i % render_freq == 0:
                    env.render()

                if terminal:
                    break
            agent.end_episode()

            if save_frames and i % render_freq == render_freq-1:
                video_index = env.unwrapped.save_render_states(vidir, title, video_index)

            returns.append(episode_reward)
            total_reward += episode_reward

            if i % 100 == 0:
                print('Episode:{0:5s} Step:{1:8s} Averaged return:{2:.4f} Most recent 100 returns:{3:.4f}'.format(str(i), str(agent._step), total_reward / (i + 1), np.array(returns[-100:]).mean()))
            writer.add_scalar("Returns/returns", episode_reward, i)
            # if i % 20000 == 0:
        writer.flush()

        end_time = time.time()
        print("Took %.2f seconds" % (end_time - start_time))
        env.close()
        if save_freq > 0 and save_dir:
            # print(os.path.join(save_dir, str(i)))
            agent.save_models(os.path.join(save_dir, str(i)))

        returns = env.get_episode_rewards()
        print("Ave. return =", sum(returns) / len(returns))
        print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

        np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)

    #-------------------start evaluation------------------
    if evaluation_episodes:
        # load models
        agent.load_models(os.path.join(save_dir, str(load_model_idx)))

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        # ----HH: ----
        agent.actor_param.eval()
        agent.actor.eval()
        evaluation_returns = evaluate(env, agent, visualise, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)

    writer.close()

if __name__ == '__main__':
    run()
