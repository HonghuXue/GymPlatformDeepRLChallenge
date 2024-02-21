import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5, noise_decay=True, noise_std_initial=1, noise_std_final=1,
                 noise_step=20000):
        super(NoisyLinear, self).__init__()
        # --HH: added decaying noise--
        self.noise_decay = noise_decay
        self.noise_std_initial = noise_std_initial
        self.noise_std_final = noise_std_final
        self.noise_step = noise_step
        self.step = 0
        # Learnable parameters.
        self.mu_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def f(self, x):
        if self.noise_decay:
            if self.step < self.noise_step:
                std = self.noise_std_initial - (self.noise_std_initial - self.noise_std_final) * (
                        self.step / self.noise_step)
            else:
                std = self.noise_std_final
            return x.normal_(std=std).sign().mul(x.abs().sqrt())
        else:
            return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self, noise_decay=False):
        """HH: Added decaying noise with on-going training"""
        if noise_decay:
            self.step += 1
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x):
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class StateActionEmbeddingNetwork(nn.Module):
    def __init__(self, state_size, action_size, action_parameter_size_list, embedding_size=64,
                 iqn_embedding_layers=(64,),
                 output_layer_init_std=None, activation="leaky_relu", **kwargs):
        super(StateActionEmbeddingNetwork, self).__init__()
        if 'noisy_network' in kwargs:
            self.noisy_network = kwargs['noisy_network']
            self.noisy_network_noise_decay = kwargs['noisy_network_noise_decay']
            self.noisy_net_noise_initial_std = kwargs['noisy_net_noise_initial_std']
            self.noisy_net_noise_final_std = kwargs['noisy_net_noise_final_std']
            self.noisy_net_noise_decay_step = kwargs['noisy_net_noise_decay_step']
        linear = NoisyLinear if self.noisy_network else nn.Linear
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size_list = action_parameter_size_list
        self.activation = activation
        # create layers
        self.layers = nn.ModuleList()
        input_size = state_size + action_size
        last_hidden_layer_size = input_size
        if iqn_embedding_layers is not None:
            nh = len(iqn_embedding_layers)
            if self.noisy_network:
                self.layers.append(linear(input_size, iqn_embedding_layers[0], noise_decay=self.noisy_network_noise_decay,
                                          noise_std_initial=self.noisy_net_noise_initial_std,
                                          noise_std_final=self.noisy_net_noise_final_std,
                                          noise_step=self.noisy_net_noise_decay_step))
            else:
                self.layers.append(linear(input_size, iqn_embedding_layers[0]))
            for i in range(1, nh):
                if self.noisy_network:
                    self.layers.append(linear(iqn_embedding_layers[i - 1], iqn_embedding_layers[i], noise_decay=self.noisy_network_noise_decay,
                                          noise_std_initial=self.noisy_net_noise_initial_std,
                                          noise_std_final=self.noisy_net_noise_final_std,
                                          noise_step=self.noisy_net_noise_decay_step))
                else:
                    self.layers.append(linear(iqn_embedding_layers[i - 1], iqn_embedding_layers[i]))
            last_hidden_layer_size = iqn_embedding_layers[nh - 1]
        if self.noisy_network:
            self.layers.append(linear(last_hidden_layer_size, embedding_size, noise_decay=self.noisy_network_noise_decay,
                                          noise_std_initial=self.noisy_net_noise_initial_std,
                                          noise_std_final=self.noisy_net_noise_final_std,
                                          noise_step=self.noisy_net_noise_decay_step))
        else:
            self.layers.append(linear(last_hidden_layer_size, embedding_size))

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
        negative_slope = 0.2
        batch_size = state.shape[0]
        # Calculate embeddings of states.
        # state_embedding = self.net(states)  # assert state_embedding.shape == (batch_size, self.embedding_size)

        x = torch.cat((state, torch.zeros_like(action_parameters)), dim=1)
        x = x.repeat(self.action_size, 1)  # (self.action_size = 3, action_dim + state+dim = 12)
        for a in range(self.action_size):
            x[a * batch_size:(a + 1) * batch_size,
            self.state_size + self.offsets[a]: self.state_size + self.offsets[a + 1]] \
                = action_parameters[:, self.offsets[a]:self.offsets[a + 1]]

        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        sa_embedding = self.layers[-1](x)  # (Batch * |A|, |emb|)

        return sa_embedding

    def sample_noise(self, noise_decay=False):
        if self.noisy_network:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample(noise_decay=noise_decay)


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines=64, embedding_size=64, hidden_layers=None, output_layer_init_std=None,
                 activation="leaky_relu", **kwargs):
        """Note: No noisy network for exploration in embedding network"""
        super(CosineEmbeddingNetwork, self).__init__()
        self.activation = activation
        self.num_cosines = num_cosines
        self.embedding_size = embedding_size
        if 'noisy_network' in kwargs:
            self.noisy_network = kwargs['noisy_network']
            self.noisy_network_noise_decay = kwargs['noisy_network_noise_decay']
        # linear = NoisyLinear if self.noisy_network else nn.Linear
        linear = nn.Linear
        # self.net = nn.Sequential(
        #     linear(num_cosines, embedding_size),
        #     nn.ReLU()
        # )

        # create layers
        self.layers = nn.ModuleList()
        inputSize = num_cosines
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(linear(lastHiddenLayerSize, embedding_size))

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

    def forward(self, taus):
        # implement forward
        negative_slope = 0.2
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        x = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        # tau_embeddings = self.net(cosines).view(
        #     batch_size, N, self.embedding_size)

        num_layers = len(self.layers)
        for i in range(0, num_layers):  # Note: Not num_layers -1 , conforming to the original IQN design
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))

        tau_embeddings = x.view(batch_size, N, self.embedding_size)
        return tau_embeddings

    def sample_noise(self, noise_decay=False):
        if self.noisy_network:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample(noise_decay=noise_decay)


class QuantileNetwork(nn.Module):
    def __init__(self, embedding_size, action_size, action_parameter_size_list, iqn_quantile_layers=(128,),
                 output_layer_init_std=None, activation="leaky_relu", **kwargs):
        super(QuantileNetwork, self).__init__()
        '''embedding size already includes the embedding of (state + action).'''
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.action_parameter_size_list = action_parameter_size_list
        self.action_parameter_size = np.sum(action_parameter_size_list)
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
        inputSize = self.embedding_size
        lastHiddenLayerSize = inputSize
        if iqn_quantile_layers is not None:
            nh = len(iqn_quantile_layers)
            # self.layers.append(linear(inputSize, iqn_quantile_layers[0]))
            if self.noisy_network:
                self.layers.append(linear(inputSize, iqn_quantile_layers[0], noise_decay=self.noisy_network_noise_decay,
                                          noise_std_initial=self.noisy_net_noise_initial_std,
                                          noise_std_final=self.noisy_net_noise_final_std,
                                          noise_step=self.noisy_net_noise_decay_step))
            else:
                self.layers.append(linear(inputSize, iqn_quantile_layers[0]))

            for i in range(1, nh):
                if self.noisy_network:
                    self.layers.append(linear(iqn_quantile_layers[i - 1], iqn_quantile_layers[i], noise_decay=self.noisy_network_noise_decay,
                                          noise_std_initial=self.noisy_net_noise_initial_std,
                                          noise_std_final=self.noisy_net_noise_final_std,
                                          noise_step=self.noisy_net_noise_decay_step))
                else:
                    self.layers.append(linear(iqn_quantile_layers[i - 1], iqn_quantile_layers[i]))
            lastHiddenLayerSize = iqn_quantile_layers[nh - 1]
        if self.noisy_network:
            self.layers.append(linear(lastHiddenLayerSize, self.action_size, noise_decay=self.noisy_network_noise_decay,
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

    def forward(self, sa_embeddings, tau_embeddings):
        """Note: state embeddings here should already concatenate the original state embedding and action."""
        # implement forward
        negative_slope = 0.2
        # assert sa_embeddings.shape[0] == tau_embeddings.shape[0]
        # assert sa_embeddings.shape[1] == tau_embeddings.shape[2]

        # NOTE: Because variable taus correspond to either \tau or \hat \tau.  in the paper, N isn't neccesarily the
        # same as fqf.N.
        batch_size = tau_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        #  ----HH: Key modifications---- reshape both sa_embeddings and tau_embeddings into  (Batch * |A|, N, |emb|)
        sa_embeddings = sa_embeddings.view(batch_size * self.action_parameter_size, 1, self.embedding_size)
        tau_embeddings = tau_embeddings.repeat(self.action_parameter_size, 1, 1)
        # Calculate embeddings of states and taus.
        # embeddings = (sa_embeddings * tau_embeddings).view(batch_size * N, self.embedding_size)
        x = (sa_embeddings * tau_embeddings).view(batch_size * N * self.action_parameter_size, self.embedding_size)

        # Calculate quantile values.
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        quantiles = self.layers[-1](x)  # (Batch * N * |A|, |A|)

        # ---HH: Key modification : select the index of |A|, such that (Batch * N * |A|, |A|) --> (Batch, N, |A|)  ---
        quantiles = quantiles.view(batch_size * self.action_parameter_size, N, self.action_parameter_size)
        # quantiles = quantiles.view(batch_size, self.action_parameter_size, N, self.action_parameter_size)

        # extract Q-values for each action
        Q = []
        for a in range(self.action_size):
            Qa = quantiles[a * batch_size:(a + 1) * batch_size, :, a]
            # if len(Qa.shape) == 1:
            #     Qa = Qa.unsqueeze(1)
            Q.append(Qa)
        # quantiles = torch.cat(Q, dim=1) # Should be (batch_size, N, self.action_parameter_size)
        quantiles = torch.stack(Q, dim=-1)
        return quantiles

    def sample_noise(self, noise_decay=False):
        if self.noisy_network:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample(noise_decay=noise_decay)
