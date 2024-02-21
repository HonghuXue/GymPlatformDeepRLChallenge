import numpy as np
import torch


def update_params(optim, loss, networks, retain_graph=False,
                  grad_clipping=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stabilize training.
    if grad_clipping:
        for net in networks:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clipping)
    optim.step()


# def disable_gradients(network):
#     # Disable calculations of gradients.
#     for param in network.parameters():
#         param.requires_grad = False


def calculate_huber_loss(td_errors: torch.Tensor, kappa: float=1.0) -> torch.Tensor:
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors: torch.Tensor, taus: torch.Tensor, weights: torch.Tensor=None, kappa: float=1.0) -> torch.Tensor:
    assert not taus.requires_grad
    # batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    # assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    # assert element_wise_quantile_huber_loss.shape == (batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
    # assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


def evaluate_quantile_at_action(s_quantiles: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    actions = actions.unsqueeze(-1)
    action_index = actions[..., None].expand(batch_size, N, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles


def generate_taus(batch_size: int, N: int, device: torch.device) -> (torch.Tensor, torch.Tensor):
    taus = np.random.uniform(0, 1, size=(batch_size, N + 1))  # + 0.1
    taus[:, 0] = 0
    taus = np.cumsum(taus, axis=1) / np.sum(taus, axis=1, keepdims=True)
    tau_hats = (taus[:, 1:] + taus[:, :-1]) / 2.
    taus = torch.tensor(taus, dtype=torch.float, device=device)
    tau_hats = torch.tensor(tau_hats, dtype=torch.float, device=device)
    return taus, tau_hats
