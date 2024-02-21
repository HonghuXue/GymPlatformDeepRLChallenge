import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math as ms
import random
from collections import Counter
from torch.autograd import Variable
from copy import deepcopy

from agents.agent import Agent
from agents.memory.memory import Memory, Prioritized_Experience_Replay
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise
from agents.utils.additional_networks import NoisyLinear, QuantileNetwork, CosineEmbeddingNetwork, \
    StateActionEmbeddingNetwork
from agents.utils.additional_functions import evaluate_quantile_at_action, calculate_quantile_huber_loss, generate_taus


class QActor(nn.Module):
    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="leaky_relu", **kwargs):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation
        if 'noisy_network' in kwargs:
            self.noisy_network = kwargs['noisy_network']
            self.noisy_network_noise_decay = kwargs['noisy_network_noise_decay']
        linear = NoisyLinear if self.noisy_network else nn.Linear

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        if not self.noisy_network:
            for i in range(0, len(self.layers) - 1):
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
                nn.init.zeros_(self.layers[i].bias)
            if output_layer_init_std is not None:
                nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
            # else:
            #     nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.2

        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        Q = self.layers[-1](x)
        return Q

    def sample_noise(self, noise_decay=False):
        if self.noisy_network:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample(noise_decay=noise_decay)


class ParamActor(nn.Module):
    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type="kaiming", activation="leaky_relu", init_std=None, **kwargs):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet
        if 'noisy_network' in kwargs:
            self.noisy_network = kwargs['noisy_network']
            self.noisy_network_noise_decay = kwargs['noisy_network_noise_decay']
            self.noisy_net_noise_initial_std = kwargs['noisy_net_noise_initial_std']
            self.noisy_net_noise_final_std = kwargs['noisy_net_noise_final_std']
            self.noisy_net_noise_decay_step = kwargs['noisy_net_noise_decay_step']
        linear = NoisyLinear if self.noisy_network else nn.Linear

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            if self.noisy_network:
                self.layers.append(linear(inputSize, hidden_layers[0],
                                          noise_decay=self.noisy_network_noise_decay,
                                          noise_std_initial=self.noisy_net_noise_initial_std,
                                          noise_std_final=self.noisy_net_noise_final_std,
                                          noise_step=self.noisy_net_noise_decay_step))
            else:
                self.layers.append(linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                if self.noisy_network:
                    self.layers.append(linear(hidden_layers[i - 1], hidden_layers[i],
                                              noise_decay=self.noisy_network_noise_decay,
                                              noise_std_initial=self.noisy_net_noise_initial_std,
                                              noise_std_final=self.noisy_net_noise_final_std,
                                              noise_step=self.noisy_net_noise_decay_step))
                else:
                    self.layers.append(linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        if self.noisy_network:
            self.action_parameters_output_layer = linear(lastHiddenLayerSize, self.action_parameter_size,
                                                        noise_decay=self.noisy_network_noise_decay,
                                                        noise_std_initial=self.noisy_net_noise_initial_std,
                                                        noise_std_final=self.noisy_net_noise_final_std,
                                                        noise_step=self.noisy_net_noise_decay_step)
        else:
            self.action_parameters_output_layer = linear(lastHiddenLayerSize, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        if not self.noisy_network:
            for i in range(0, len(self.layers)):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
                elif init_type == "normal":
                    nn.init.normal_(self.layers[i].weight, std=init_std)
                else:
                    raise ValueError("Unknown init_type " + str(init_type))
                nn.init.zeros_(self.layers[i].bias)
            if output_layer_init_std is not None:
                nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
            else:
                nn.init.zeros_(self.action_parameters_output_layer.weight)
            nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        '''Note: self.layers[-1](x) has undergone non-linearity here!'''
        x = state
        negative_slope = 0.2
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):  # Note : NOT  num_hidden_layers - 1
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)

        if self.squashing_function:
            assert False  # scaling not implemented yet
            action_params = action_params.tanh()
            action_params = action_params * self.action_param_lim
        # action_params = action_params / torch.norm(action_params) ## REMOVE --- normalisation layer?? for pointmass
        return action_params

    def sample_noise(self, noise_decay=False):
        if self.noisy_network:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample(noise_decay=noise_decay)


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """
    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,
                 # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss,  # F.mse_loss
                 clip_grad=10,
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None,
                 train_interval=8,
                 DDQN=False,
                 delayed_policy_update=1,
                 td3_target_policy_smoothing=False,
                 td3_policy_noise_std=0.2,
                 td3_policy_noise_clamp=0.5,
                 PER=False,
                 per_no_is=True,
                 noisy_network=False,
                 noisy_network_noise_decay=False,
                 noisy_net_noise_initial_std=1.0,
                 noisy_net_noise_final_std=0.01,
                 noisy_net_noise_decay_step=2,
                 iqn=False,
                 iqn_quantile_num=8,
                 tensorboard_writer=None
                 ):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, self.num_actions + 1)])
        print("self.action_parameter_sizes", self.action_parameter_sizes)
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()
        print([self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)])
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        print('action_max: {}, action_min: {}, action_parameter_max_numpy: {}, action_parameter_min_numpy: {}'.format(
            self.action_max, self.action_min, self.action_parameter_max, self.action_parameter_min))

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = None
        self.seed = seed
        self._seed(seed)
        # --- HH: ---
        self.train_interval = train_interval
        self.double_learning = DDQN
        self.delayed_policy_update = delayed_policy_update
        self.td3_target_policy_smoothing = td3_target_policy_smoothing
        self.td3_policy_noise_std = td3_policy_noise_std
        self.td3_policy_noise_clamp = td3_policy_noise_clamp
        self.PER = PER
        self.PER_NO_IS = per_no_is
        self.noisy_network = noisy_network
        self.noisy_network_noise_decay = noisy_network_noise_decay
        self.noisy_net_noise_initial_std = noisy_net_noise_initial_std
        self.noisy_net_noise_final_std = noisy_net_noise_final_std
        self.noisy_net_noise_decay_step = noisy_net_noise_decay_step
        self.IQN = iqn
        self.N = iqn_quantile_num
        self.N_dash = iqn_quantile_num
        self.writer = tensorboard_writer

        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0.,
                                                  theta=0.15, sigma=0.0001)  # , theta=0.01, sigma=0.01)
        print(self.num_actions + self.action_parameter_size)

        if not self.PER:
            self.replay_memory = Memory(replay_memory_size, observation_space.shape, (1 + self.action_parameter_size,),
                                        next_actions=False)
        else:
            assert ms.log2(replay_memory_size) % 1 == 0
            self.replay_memory = Prioritized_Experience_Replay(replay_memory_size, observation_space.shape,
                                                               (1 + self.action_parameter_size,), next_actions=False,
                                                               PER_NO_IS=self.PER_NO_IS)

        if not self.IQN:
            self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                                     **actor_kwargs).to(device)
            self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions,
                                            self.action_parameter_size, **actor_kwargs).to(device)
            hard_update_target_network(self.actor, self.actor_target)
            self.actor_target.eval()
        else:
            self.sa_embedding_net = StateActionEmbeddingNetwork(self.observation_space.shape[0], self.num_actions,
                                                                self.action_parameter_sizes, **actor_kwargs).to(device)
            self.cosine_net = CosineEmbeddingNetwork(**actor_kwargs).to(device)
            self.quantile_net = QuantileNetwork(embedding_size=actor_kwargs['IQN_embedding_size'],
                                                action_size=self.num_actions,
                                                action_parameter_size_list=self.action_parameter_sizes,
                                                **actor_kwargs).to(device)
            self.sa_embedding_net_target = StateActionEmbeddingNetwork(self.observation_space.shape[0],
                                                                       self.num_actions, self.action_parameter_sizes,
                                                                       **actor_kwargs).to(device)
            self.cosine_net_target = CosineEmbeddingNetwork(**actor_kwargs).to(device)
            self.quantile_net_target = QuantileNetwork(embedding_size=actor_kwargs['IQN_embedding_size'],
                                                       action_size=self.num_actions,
                                                       action_parameter_size_list=self.action_parameter_sizes,
                                                       **actor_kwargs).to(device)
            hard_update_target_network(self.sa_embedding_net, self.sa_embedding_net_target)
            hard_update_target_network(self.cosine_net, self.cosine_net_target)
            hard_update_target_network(self.quantile_net, self.quantile_net_target)
            self.sa_embedding_net_target.eval()
            self.cosine_net_target.eval()
            self.quantile_net_target.eval()

        self.actor_param = actor_param_class(self.observation_space.shape[0], self.num_actions,
                                             self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(self.observation_space.shape[0], self.num_actions,
                                                    self.action_parameter_size, **actor_param_kwargs).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        # HH:
        if self.PER:
            self.loss_func = nn.MSELoss(reduction='none')
        else:
            self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        if not self.IQN:
            self.actor_optimiser = optim.Adam(self.actor.parameters(),
                                              lr=self.learning_rate_actor)  # , betas=(0.95, 0.999))
        else:
            self.sa_embedding_net_optimiser = optim.Adam(self.sa_embedding_net.parameters(),
                                                         lr=self.learning_rate_actor)
            self.cosine_net_optimiser = optim.Adam(self.cosine_net.parameters(), lr=self.learning_rate_actor)
            self.quantile_net_optimiser = optim.Adam(self.quantile_net.parameters(), lr=self.learning_rate_actor)
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(),
                                                lr=self.learning_rate_actor_param)  # , betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def TD3_policy_smoothing(self, next_action_parameters, running_actor_param_network: bool = False) -> torch.Tensor:
        if not running_actor_param_network:
            policy_noise = torch.randn(next_action_parameters.shape,
                                       device=self.device) * self.td3_policy_noise_std  # .to(self.device)
            policy_noise = policy_noise.clamp(-self.td3_policy_noise_clamp, self.td3_policy_noise_clamp)
            next_action = next_action_parameters + policy_noise
            # next_action = self.actor_param_target(next_states) + policy_noise
            pred_next_action_parameters = next_action.clamp(self.action_min, self.action_max)
        else:
            policy_noise = torch.randn(next_action_parameters.shape,
                                       device=self.device) * self.td3_policy_noise_std  # .to(self.device)
            policy_noise = policy_noise.clamp(-self.td3_policy_noise_clamp, self.td3_policy_noise_clamp)
            # next_action = self.actor_param(next_states) + policy_noise
            next_action = next_action_parameters + policy_noise
            pred_next_action_parameters = next_action.clamp(self.action_min, self.action_max)
        return pred_next_action_parameters

    def calculate_quantiles(self, states: torch.Tensor, action_parameters: torch.Tensor, tau_hats: torch.Tensor,
                            running_network: bool = False) -> torch.Tensor:
        " Calculate the quantiles."
        if not running_network:
            sa_embeddings = self.sa_embedding_net_target(states, action_parameters)
            tau_embeddings = self.cosine_net_target(tau_hats)
            quantiles = self.quantile_net_target(sa_embeddings, tau_embeddings)
        else:
            sa_embeddings = self.sa_embedding_net(states, action_parameters)
            tau_embeddings = self.cosine_net(tau_hats)
            quantiles = self.quantile_net(sa_embeddings, tau_embeddings)
        # assert next_quantiles.shape == (self.batch_size, self.N, self.num_actions)
        return quantiles

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        print(initial_weights.shape)
        print(passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            print(initial_bias.shape)
            print(passthrough_layer.bias.data.size())
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.
        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def act(self, state):
        with torch.no_grad():
            # --HH: add noisy net for exploration when performing behavior policy
            if self.noisy_network:
                self.actor_param.sample_noise(noise_decay=False)
                if self.IQN:
                    self.sa_embedding_net.sample_noise(noise_decay=False)
                    self.quantile_net.sample_noise(noise_decay=False)
                    self.cosine_net.sample_noise(noise_decay=False)
                else:
                    self.actor.sample_noise(noise_decay=False)

            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)

            if self.noisy_network:
                # select maximum action
                if self.IQN:
                    # Sample fractions.
                    taus, tau_hats = generate_taus(batch_size=1, N=self.N, device=self.device)

                    sa_embeddings = self.sa_embedding_net(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    tau_embeddings = self.cosine_net(tau_hats)
                    quantiles = self.quantile_net(sa_embeddings, tau_embeddings)
                    Q_a = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantiles).sum(dim=1)
                    Q_a = Q_a.detach().cpu().data.numpy()
                    action = np.argmax(Q_a)
                else:
                    Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    Q_a = Q_a.detach().cpu().data.numpy()
                    action = np.argmax(Q_a)
            else:
                # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
                rnd = self.np_random.uniform()
                if rnd < self.epsilon:
                    action = self.np_random.choice(self.num_actions)
                    if not self.use_ornstein_noise:
                        all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                                   self.action_parameter_max_numpy))
                else:
                    # select maximum action
                    Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    Q_a = Q_a.detach().cpu().data.numpy()
                    action = np.argmax(Q_a)

            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            if self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += self.noise.sample()[
                                                                                              offset:offset +
                                                                                                     self.action_parameter_sizes[
                                                                                                         action]]
            action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a + 1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        act, all_action_parameters = action
        self._step += 1

        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        self._add_sample(state, np.concatenate(([act], all_action_parameters)).ravel(), reward, next_state,
                         np.concatenate(([next_action[0]], next_action[1])).ravel(), terminal=terminal)

        if (self._step >= self.batch_size) and (self._step >= self.initial_memory_threshold) and (
                self._step % self.train_interval == 0):
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        self.actor.train()
        self.actor_target.train()
        self.actor_param.train()
        self.actor_param_target.train()

        if self.noisy_network:
            if not self.IQN:
                self.actor.sample_noise(self.noisy_network_noise_decay)
                self.actor_target.sample_noise(self.noisy_network_noise_decay)
                #---- below is for trialing---
                # self.actor_param.sample_noise(self.noisy_network_noise_decay)
                # self.actor_param_target.sample_noise(self.noisy_network_noise_decay)
            else:
                self.quantile_net.sample_noise(self.noisy_network_noise_decay)
                self.sa_embedding_net.sample_noise(self.noisy_network_noise_decay)
                # self.cosine_net.sample_noise(self.noisy_network_noise_decay)
                self.quantile_net_target.sample_noise(self.noisy_network_noise_decay)
                self.sa_embedding_net_target.sample_noise(self.noisy_network_noise_decay)
                # self.cosine_net_target.sample_noise(self.noisy_network_noise_decay)
                self.actor_param.sample_noise(self.noisy_network_noise_decay)
                self.actor_param_target.sample_noise(self.noisy_network_noise_decay)

        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        if self.PER:
            transitions, idxs, weights = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
            states, actions, rewards, next_states, terminals = transitions
            weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        else:
            states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size,
                                                                                         random_machine=self.np_random)
        # print(states.shape, actions.shape, rewards.shape, terminals.shape) #(128, 9) (128, 4) (128, 1) (128, 1)
        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        if not self.IQN:  # ---DQN mode----
            # -- First compute the td-targets, no gradient required--
            # HH:---original implementation: DQN----
            if not self.double_learning:
                with torch.no_grad():
                    pred_next_action_parameters = self.actor_param_target.forward(next_states)
                    # ------ HH: add TD3 policy noise------
                    if self.td3_target_policy_smoothing:
                        pred_next_action_parameters = self.TD3_policy_smoothing(pred_next_action_parameters,
                                                                                running_actor_param_network=False)
                    pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
                    Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            # HH:--Proposed implementation :DDQN---
            else:
                with torch.no_grad():
                    pred_next_action_parameters_running = self.actor_param.forward(next_states)
                    pred_next_action_parameters = self.actor_param_target.forward(next_states)
                    if self.td3_target_policy_smoothing:
                        pred_next_action_parameters = self.TD3_policy_smoothing(pred_next_action_parameters,
                                                                                running_actor_param_network=False)
                    pred_Q_a_tmp = self.actor(next_states, pred_next_action_parameters_running)
                    pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
                    Qprime = pred_Q_a.gather(1, torch.max(pred_Q_a_tmp, 1)[1].unsqueeze(1)).squeeze(1)

            # Compute the TD error
            with torch.no_grad():
                target = rewards + (1 - terminals) * self.gamma * Qprime
            y_expected = target.detach()
            # Compute current Q-values using policy network
            q_values = self.actor(states, action_parameters)
            y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()

            if self.PER:
                loss_Q = (weights * self.loss_func(y_predicted, y_expected)).mean()
                td_error_abs = F.l1_loss(y_predicted, y_expected, reduction='none').cpu().detach().numpy() + 1e-8
                # update priority for trained samples
                for idx, td_error in zip(idxs, td_error_abs):
                    self.replay_memory.update_priority(idx, td_error)

            else:
                loss_Q = self.loss_func(y_predicted, y_expected)

            self.actor_optimiser.zero_grad()
            loss_Q.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
            self.actor_optimiser.step()
        else:  # ---------------------IQN---------------------

            # Sample fractions.
            taus, tau_hats = generate_taus(self.batch_size, self.N, device=self.device)
            next_taus, next_tau_hats = generate_taus(self.batch_size, self.N, device=self.device)

            # Calculate quantile values of current states and actions at tau_hats.
            quantiles = self.calculate_quantiles(states, action_parameters, tau_hats, running_network=True)
            current_sa_quantiles = evaluate_quantile_at_action(quantiles, actions)
            # assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

            with torch.no_grad():
                # Calculate Q values of next states.
                if self.double_learning:  # double Q-learning
                    # ------next_q = self.online_net.calculate_q(states=next_states) # (Batch, |A|)------
                    pred_next_action_parameters = self.actor_param.forward(next_states)
                    next_quantiles = self.calculate_quantiles(next_states, pred_next_action_parameters, next_tau_hats,
                                                              running_network=True)
                    # assert next_quantiles.shape == (self.batch_size, self.N, self.num_actions)
                    # Calculate expectations of value distribution.
                    next_q = ((next_taus[:, 1:, None] - next_taus[:, :-1, None]) * next_quantiles).sum(
                        dim=1)  # assert q.shape == (self.batch_size, self.num_actions)
                    # -----------------------------------------------------------------------------------
                else:  # Normal Quantile learning, not double learning
                    pred_next_action_parameters = self.actor_param_target.forward(next_states)
                    # ------ HH: add TD3 policy noise------
                    if self.td3_target_policy_smoothing:
                        pred_next_action_parameters = self.TD3_policy_smoothing(pred_next_action_parameters,
                                                                                running_actor_param_network=False)
                    next_quantiles = self.calculate_quantiles(next_states, pred_next_action_parameters, next_tau_hats,
                                                              running_network=False)
                    # assert next_quantiles.shape == (self.batch_size, self.N, self.num_actions)

                    # Calculate expectations of value distribution.
                    next_q = ((next_taus[:, 1:, None] - next_taus[:, :-1, None]) * next_quantiles).sum(
                        dim=1)  # assert q.shape == (self.batch_size, self.num_actions)
                    # ------------------------------------------------
                # Calculate greedy actions.
                next_actions = torch.argmax(next_q, dim=1,
                                            keepdim=False)  # assert next_actions.shape == (self.batch_size, 1)

                # Calculate features of next states.
                if self.double_learning:
                    # --------next_state_embeddings = self.target_net.calculate_state_embeddings(next_states)---------
                    pred_next_action_parameters = self.actor_param_target.forward(next_states)
                    # ------ HH: add TD3 policy noise------
                    if self.td3_target_policy_smoothing:
                        pred_next_action_parameters = self.TD3_policy_smoothing(pred_next_action_parameters,
                                                                                running_actor_param_network=False)
                    next_quantiles = self.calculate_quantiles(next_states, pred_next_action_parameters, next_tau_hats,
                                                              running_network=False)
                    # ------------------------------------------------------------------------------------------------
                # Calculate quantile values of next states and next actions.
                next_sa_quantiles = evaluate_quantile_at_action(next_quantiles, next_actions).transpose(1,
                                                                                                        2)  # assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

                # Calculate target quantile values.
                target_sa_quantiles = rewards[..., None, None] + (1.0 - terminals[
                    ..., None, None]) * self.gamma * next_sa_quantiles  # assert target_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

            td_errors = target_sa_quantiles.detach() - current_sa_quantiles  # assert td_errors.shape == (self.batch_size, self.N, self.N_dash)
            errors = td_errors.detach().abs().sum(dim=1).mean(dim=1,
                                                              keepdim=True)  # assert errors.shape == (self.batch_size, 1)
            quantile_huber_loss = calculate_quantile_huber_loss(td_errors, tau_hats, weights, kappa=1.0)
            if self.updates % 1000 == 999:
                print('quantile_huber_loss', quantile_huber_loss)
            self.quantile_net_optimiser.zero_grad()
            self.sa_embedding_net_optimiser.zero_grad()
            self.cosine_net_optimiser.zero_grad()
            quantile_huber_loss.backward()
            self.writer.add_scalar("Loss/quantile huber loss", quantile_huber_loss, self.updates)
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.quantile_net.parameters(), self.clip_grad)
                torch.nn.utils.clip_grad_norm_(self.sa_embedding_net.parameters(), self.clip_grad)
                torch.nn.utils.clip_grad_norm_(self.cosine_net.parameters(), self.clip_grad)
            self.quantile_net_optimiser.step()
            self.sa_embedding_net_optimiser.step()
            self.cosine_net_optimiser.step()

            if self.PER:
                for idx, td_error in zip(idxs, errors.squeeze().cpu().detach().numpy()):
                    self.replay_memory.update_priority(idx, td_error)

        # ---------------------- optimize actor ----------------------
        # states, action_params, tau_hats
        if self.updates % self.delayed_policy_update == self.delayed_policy_update - 1:
            # if self.noisy_network:
            #     self.actor_param.sample_noise(self.noisy_network_noise_decay)
            #     self.actor_param_target.sample_noise(self.noisy_network_noise_decay)

            # --HH:  -Integrate delayed states-
            try:  # for the case of delayed_policy_update > 1
                states = torch.cat((self.state_policy_delayed, states), 0)
                weights = torch.cat((self.weights, weights), 0)
                tau_hats = torch.cat((self.tau_hats, tau_hats), 0)
                taus = torch.cat((self.taus, taus), 0)
            except:  # for the case of delayed_policy_update = 1
                pass

            with torch.no_grad():
                action_params = self.actor_param(states)
            action_params.requires_grad = True
            assert (self.weighted ^ self.average ^ self.random_weighted) or not (
                        self.weighted or self.average or self.random_weighted)
            if not self.IQN:
                Q = self.actor(states, action_params)
            else:
                quantiles = self.calculate_quantiles(states, action_params, tau_hats,
                                                     running_network=True)  # assert next_quantiles.shape == (self.batch_size, self.N, self.num_actions)

                # Calculate expectations of value distribution.
                Q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantiles).sum(
                    dim=1)  # assert q.shape == (self.batch_size, self.num_actions)
                # self.writer.add_scalar("Estimated-Q Median", Q.median(), self.updates)
                self.writer.add_scalar("Estimated-Q Max", Q.max(), self.updates)
                self.writer.add_scalar("Estimated-Q Mean", Q.mean(), self.updates)
            Q_val = Q
            if self.weighted:
                # approximate categorical probability density (i.e. counting)
                counts = Counter(actions.cpu().numpy())
                weights = torch.from_numpy(
                    np.array([counts[a] / actions.shape[0] for a in range(self.num_actions)])).float().to(self.device)
                Q_val = weights * Q
            elif self.average:
                Q_val = Q / self.num_actions
            elif self.random_weighted:
                weights = np.random.uniform(0, 1., self.num_actions)
                weights /= np.linalg.norm(weights)
                weights = torch.from_numpy(weights).float().to(self.device)
                Q_val = weights * Q
            if self.indexed:
                Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
                Q_loss = torch.mean(Q_indexed)
            else:
                if self.PER:
                    Q_loss = torch.mean(weights * torch.sum(Q_val, 1))
                else:
                    Q_loss = torch.mean(torch.sum(Q_val, 1))
            self.actor.zero_grad()
            self.writer.add_scalar("Loss/Q loss", Q_loss, self.updates)
            Q_loss.backward()  # Important step: first generate the gradient, then examine inverting/ zero gradients

            delta_a = deepcopy(action_params.grad.data)
            # step 2
            action_params = self.actor_param(Variable(states))
            delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
            if self.zero_index_gradients:
                delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)

            out = -torch.mul(delta_a, action_params)  # important step: so that delta can be back propagated
            self.writer.add_scalar("Loss/out mean", out.mean(), self.updates)
            self.actor_param.zero_grad()
            out.backward(torch.ones(out.shape).to(self.device))
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

            self.actor_param_optimiser.step()

        elif self.updates % self.delayed_policy_update == 0:
            self.state_policy_delayed = states
            self.weights = weights
            self.tau_hats = tau_hats
            self.taus = taus
        else:
            self.state_policy_delayed = torch.cat((self.state_policy_delayed, states), 0)
            self.weights = torch.cat((self.weights, weights), 0)
            self.tau_hats = torch.cat((self.tau_hats, tau_hats), 0)
            self.taus = torch.cat((self.taus, taus), 0)

        if self.IQN:
            soft_update_target_network(self.quantile_net, self.quantile_net_target, self.tau_actor)
            soft_update_target_network(self.cosine_net, self.cosine_net_target, self.tau_actor)
            soft_update_target_network(self.sa_embedding_net, self.sa_embedding_net_target, self.tau_actor)
        else:
            soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)
        # self.writer.flush()

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target network too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')
