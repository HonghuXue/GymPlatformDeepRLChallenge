import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.pdqn import PDQNAgent
from agents.utils import hard_update_target_network
from agents.utils.additional_networks import NoisyLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiPassQActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=(100,),
                 output_layer_init_std=None, activation="relu", **kwargs):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size_list = action_parameter_size_list
        self.action_parameter_size = sum(action_parameter_size_list)
        self.activation = activation
        if 'noisy_network' in kwargs:
            self.noisy_network = kwargs['noisy_network']
            self.noisy_network_noise_decay = kwargs['noisy_network_noise_decay']
            self.noisy_net_noise_initial_std = kwargs['noisy_net_noise_initial_std']
            self.noisy_net_noise_final_std = kwargs['noisy_net_noise_final_std']
            self.noisy_net_noise_decay_step = kwargs['noisy_net_noise_decay_step']
        linear = NoisyLinear if self.noisy_network else nn.Linear

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
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
            self.layers.append(linear(lastHiddenLayerSize, self.action_size,
                                      noise_decay=self.noisy_network_noise_decay,
                                      noise_std_initial=self.noisy_net_noise_initial_std,
                                      noise_std_final=self.noisy_net_noise_final_std,
                                      noise_step=self.noisy_net_noise_decay_step))
        else:
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

        self.offsets = self.action_parameter_size_list.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.2

        Q = []
        # duplicate inputs so we can process all actions in a single pass
        batch_size = state.shape[0]
        # with torch.no_grad():
        x = torch.cat((state, torch.zeros_like(action_parameters)), dim=1)
        x = x.repeat(self.action_size, 1) # (self.action_size = 3, action_dim + state+dim = 12)
        for a in range(self.action_size):
            x[a*batch_size:(a+1)*batch_size, self.state_size + self.offsets[a]: self.state_size + self.offsets[a+1]] \
                = action_parameters[:, self.offsets[a]:self.offsets[a+1]]
        # print(state.shape, x.shape)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        Qall = self.layers[-1](x)

        # extract Q-values for each action
        for a in range(self.action_size):
            Qa = Qall[a*batch_size:(a+1)*batch_size, a]
            if len(Qa.shape) == 1:
                Qa = Qa.unsqueeze(1)
            Q.append(Qa)
        Q = torch.cat(Q, dim=1)
        return Q # (B,3)

    def sample_noise(self, noise_decay:bool=False):
        if self.noisy_network:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample(noise_decay=noise_decay)



class MultiPassPDQNAgent(PDQNAgent):
    NAME = "Multi-Pass P-DQN Agent"

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                     **kwargs['actor_kwargs']).to(device)
        self.actor_target = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                            **kwargs['actor_kwargs']).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
