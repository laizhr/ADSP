from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Settings:
    steps: int = 1000
    samples: int = 100
    size: int = 8
    sweeps: int = 10
    burn_in: int = 100
    lr: float = 0.01
    beta_init: float = 0.11
    lr_beta: float = 0.01
    lr_theta: float = 0.0001
    theta_final_ratio: float = 0.01
    temperature_start: float = 1.0
    temperature_end: float = 0.5
    attention_gap: float = 8.0
    init: str = 'near_identity'
    grad_clip: float = 10.0
    adam_eps: float = 1e-08
    flv_weight: float = 0.1
    epsilon: float = 0.15
    pgd_steps: int = 10
    pgd_step_size: float = 0.02
    seed: int = 42
    device: str = 'cpu'
    attack_objective: str = 'feature_norm'
    mapping: str = 'learned'
    attack_scope: str = 'both'
    lr_final_ratio: float = 0.01

class Sampler:

    def __init__(self, model, cfg, seed):
        if model not in ('ising', 'potts') or cfg.size % 2:
            raise ValueError('Checkerboard sampler requires Ising/Potts and an even lattice size')
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.integers(0, 2 if model == 'ising' else 3, (cfg.samples, cfg.size, cfg.size))
        if model == 'ising':
            self.spins = 2 * self.spins - 1
        y, x = np.indices((cfg.size, cfg.size))
        self.parity = (y + x) % 2

    def draw(self, beta, sweeps):
        s = self.spins
        for _ in range(sweeps):
            for color in (0, 1):
                neighbors = [np.roll(s, shift, axis) for axis in (1, 2) for shift in (-1, 1)]
                if self.model == 'ising':
                    proposal = -s
                    delta_energy = 2 * s * sum(neighbors)
                else:
                    proposal = (s + self.rng.integers(1, 3, s.shape)) % 3
                    delta_energy = sum(((s == n).astype(int) - (proposal == n) for n in neighbors))
                accept = (self.parity == color) & (np.log(self.rng.random(s.shape)) < -beta * delta_energy)
                if self.model == 'ising':
                    accept &= self.rng.random(s.shape) < 0.5
                s[accept] = proposal[accept]
        if self.model == 'ising':
            links = np.stack((s * np.roll(s, -1, 2), s * np.roll(s, -1, 1)), 1)
            sufficient = links.sum((1, 2, 3))
        else:
            matches = np.stack((s == np.roll(s, -1, 2), s == np.roll(s, -1, 1)), 1)
            sufficient = matches.sum((1, 2, 3))
            links = 1.5 * matches - 0.5
        return (links.astype(np.float32), sufficient.astype(np.float32))

def features(links):
    h, v = links.unbind(1)
    h_down = h.roll(-1, 1)
    v_right = v.roll(-1, 2)
    v_up = v.roll(1, 1)
    v_up_right = v_up.roll(-1, 2)
    terms = (h, v, h * v_right, h * v_up_right, h * v, h * v_up, h * h_down, v * v_right, v * h_down * v_right, v * h_down * h, v * v_right * h, h_down * v_right * h, h * v * h_down * v_right)
    return torch.stack([term.mean((1, 2)) for term in terms], 1)

def neighborhood_indices(size):

    def idx(direction, y, x):
        return direction * size * size + y % size * size + x % size
    result = []
    for direction in (0, 1):
        for y in range(size):
            for x in range(size):
                if direction == 0:
                    edges = [(1, y, x), (0, y, x - 1), (1, y - 1, x), (1, y - 1, x + 1), (0, y, x + 1), (1, y, x + 1)]
                else:
                    edges = [(0, y, x), (1, y - 1, x), (0, y, x - 1), (0, y + 1, x - 1), (1, y + 1, x), (0, y + 1, x)]
                result.append([idx(*e) for e in edges] + [idx(direction, y, x)])
    return torch.tensor(result)

class ObservableMap(nn.Module):

    def __init__(self, size, mapping='learned', init='near_identity', attention_gap=8.0):
        super().__init__()
        self.mapping = mapping
        self.register_buffer('indices', neighborhood_indices(size))
        self.embedding = nn.Parameter(torch.randn(7))
        self.weight = nn.Parameter(torch.empty(()).uniform_(-1, 1))
        self.bias = nn.Parameter(torch.empty(()).uniform_(-1, 1))
        if init == 'near_identity':
            with torch.no_grad():
                self.embedding.zero_()
                self.embedding[6] = attention_gap
                self.weight.fill_(1.0)
                self.bias.zero_()
        elif init != 'paper_random':
            raise ValueError(f'Unknown initialization: {init}')
        if mapping not in ('identity', 'learned'):
            raise ValueError(f'Unknown mapping: {mapping}')
        if mapping == 'identity':
            self.requires_grad_(False)

    def attention(self, temperature, noise=None, hard=True):
        if noise is None:
            return F.gumbel_softmax(self.embedding, tau=temperature, hard=hard)
        soft = ((self.embedding + noise) / temperature).softmax(-1)
        if not hard:
            return soft
        onehot = F.one_hot(soft.argmax(), 7).to(soft)
        return onehot - soft.detach() + soft

    def forward(self, links, attention):
        if self.mapping == 'identity':
            return links
        local = links.flatten(1)[:, self.indices]
        selected = (local * attention).sum(-1).reshape_as(links)
        return self.weight * selected + self.bias

def pgd(links, mapping, attention, target, cfg):
    if cfg.epsilon == 0 or cfg.pgd_steps == 0:
        return links.detach()
    original = links.detach()
    adversarial = original.clone()
    for _ in range(cfg.pgd_steps):
        adversarial.requires_grad_(True)
        phi = features(mapping(adversarial, attention.detach()))
        if cfg.attack_objective == 'feature_norm':
            objective = -torch.linalg.vector_norm(phi, dim=1).mean()
        else:
            objective = (phi.mean(0) - target).square().sum()
        gradient, = torch.autograd.grad(objective, adversarial)
        adversarial = adversarial.detach() + cfg.pgd_step_size * gradient.sign()
        adversarial = torch.maximum(torch.minimum(adversarial, original + cfg.epsilon), original - cfg.epsilon).detach()
    return adversarial

def expectation_with_score(phi, sufficient, beta):
    centered = sufficient - sufficient.mean()
    jacobian = (centered[:, None] * (phi - phi.mean(0))).sum(0) / (len(phi) - 1)
    return phi.mean(0) + (beta - beta.detach()) * jacobian.detach()

def objective(mean_features, target, regularization):
    residual = mean_features - target
    base = residual.square().sum()
    flv = 2 * torch.linalg.vector_norm(residual)
    return (base + regularization * flv, base, flv)

def generate_observations(model, coupling, cfg, data_seed):
    sampler = Sampler(model, cfg, data_seed)
    sampler.draw(coupling, cfg.burn_in)
    observed = np.empty((cfg.steps, cfg.samples, 2, cfg.size, cfg.size), dtype=bool)
    for k in range(cfg.steps):
        links, _ = sampler.draw(coupling, cfg.sweeps)
        observed[k] = links == 1
    return observed


def estimate(observations, cfg, model):
    if observations.shape != (cfg.steps, cfg.samples, 2, cfg.size, cfg.size):
        raise ValueError('Observation shape does not match the configuration')
    if observations.dtype != np.bool_ or cfg.mapping != 'learned':
        raise ValueError('Expected boolean observations and a learned mapping')
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    sampler = Sampler(model, cfg, cfg.seed + 1)
    sampler.draw(cfg.beta_init, cfg.burn_in)
    mapping = ObservableMap(cfg.size, cfg.mapping, cfg.init, cfg.attention_gap).to(device)
    beta = nn.Parameter(torch.tensor(cfg.beta_init, device=device))
    theta = list(mapping.parameters())
    optimizer = torch.optim.Adam([
        {'params': [beta], 'lr': cfg.lr_beta, 'name': 'beta'},
        {'params': theta, 'lr': cfg.lr_theta, 'name': 'theta'},
    ], eps=cfg.adam_eps)
    initial = torch.cat([p.detach().reshape(-1) for p in theta]).cpu().tolist()
    target = torch.zeros(13, device=device)
    history = np.empty(cfg.steps + 1)
    history[0] = beta.item()
    gradient_max = 0.0
    for k in range(cfg.steps):
        fraction = min(k / 999, 1.0)
        cosine = (1 + np.cos(np.pi * fraction)) / 2
        optimizer.param_groups[0]['lr'] = cfg.lr_beta * (cfg.lr_final_ratio + (1-cfg.lr_final_ratio)*cosine)
        optimizer.param_groups[1]['lr'] = cfg.lr_theta * (cfg.theta_final_ratio + (1-cfg.theta_final_ratio)*cosine)
        samples, sufficient = sampler.draw(float(beta.detach()), cfg.sweeps)
        data = torch.as_tensor(observations[k].astype(np.float32), device=device)
        data = (2*data-1) if model == 'ising' else (1.5*data-.5)
        samples = torch.as_tensor(samples, device=device)
        sufficient = torch.as_tensor(sufficient, device=device)
        temperature = cfg.temperature_start + (cfg.temperature_end-cfg.temperature_start)*fraction
        attention = mapping.attention(temperature)
        if cfg.attack_scope == 'both':
            data = pgd(data, lambda x, a: x, attention, features(samples).mean(0), cfg)
        target += (features(data).mean(0).detach()-target)/(k+1)
        adversarial = pgd(samples, mapping, attention, target, cfg)
        phi = features(mapping(adversarial, attention))
        mean = expectation_with_score(phi, sufficient, beta)
        loss, _, _ = objective(mean, target, cfg.flv_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in [beta, *theta]):
            raise RuntimeError(f'Nonfinite gradient at step {k+1}')
        gradient_max = max(gradient_max, torch.cat([p.grad.reshape(-1) for p in theta]).norm().item())
        torch.nn.utils.clip_grad_norm_([beta], cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(theta, cfg.grad_clip)
        optimizer.step()
        if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in [beta, *theta]):
            raise RuntimeError(f'Nonfinite parameter or loss at step {k+1}')
        history[k+1] = beta.item()
    return history, {
        'theta_parameters': sum(p.numel() for p in theta),
        'theta_initial': initial,
        'theta_final': torch.cat([p.detach().reshape(-1) for p in theta]).cpu().tolist(),
        'theta_gradient_max': gradient_max,
    }
