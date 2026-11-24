import time
import numpy as np
from numba import njit
import torch
from torch import nn
from torch.nn import functional as F
from .adsp import features, neighborhood_indices

@njit(cache=True)
def serial_samples(size, beta, samples, sample_rate, rng):
    lattice = np.ones((size, size))
    lattices = np.empty((samples, size, size))
    warmup = 50 * size * size
    count = 0
    for step in range(warmup + samples * sample_rate):
        x = rng.integers(0, size)
        y = rng.integers(0, size)
        neighbors = lattice[(x + 1) % size, y] + lattice[(x - 1) % size, y] + lattice[x, (y + 1) % size] + lattice[x, (y - 1) % size]
        delta = 2 * lattice[x, y] * neighbors * beta
        if rng.random() < np.exp(-delta):
            lattice[x, y] *= -1
        if step >= warmup and step % sample_rate == 0:
            lattices[count] = lattice
            count += 1
    return lattices

@njit(cache=True)
def serial_samples_potts(size, beta, samples, sample_rate, rng):
    lattice = np.zeros((size, size), dtype=np.int64)
    lattices = np.empty((samples, size, size), dtype=np.int64)
    warmup = 50 * size * size
    count = 0
    for step in range(warmup + samples * sample_rate):
        x = rng.integers(0, size)
        y = rng.integers(0, size)
        current = lattice[x, y]
        proposal = rng.integers(0, 3)
        neighbors = (lattice[(x + 1) % size, y], lattice[(x - 1) % size, y], lattice[x, (y + 1) % size], lattice[x, (y - 1) % size])
        delta = 0
        for neighbor in neighbors:
            delta += int(neighbor == current) - int(neighbor == proposal)
        if rng.random() < np.exp(-beta * delta):
            lattice[x, y] = proposal
        if step >= warmup and step % sample_rate == 0:
            lattices[count] = lattice
            count += 1
    return lattices

class OriginalMap(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.f = nn.Linear(1, 1)
        with torch.no_grad():
            self.f.weight.fill_(-0.25)
            self.f.bias.fill_(0.75)
        self.embedding = nn.Parameter(torch.empty(7).normal_(0, 0.1))
        indices = neighborhood_indices(size)
        self.register_buffer('indices', indices)

    def forward(self, links, pointwise=False):
        attention = F.gumbel_softmax(self.embedding, tau=0.5, hard=False)
        flat = links.flatten(1)
        local = flat[..., None] if pointwise else flat[:, self.indices]
        selected = (local * attention[None, None]).sum(-1, keepdim=True)
        return self.f(selected).reshape_as(links)

def original_pgd(links, mapping, cfg):
    adversarial = links.clone().detach()
    for _ in range(cfg.pgd_steps):
        adversarial.requires_grad_(True)
        phi = features(mapping(adversarial, pointwise=True))
        attack_loss = -torch.linalg.vector_norm(phi, dim=1).mean()
        gradient, = torch.autograd.grad(attack_loss, adversarial)
        adversarial = adversarial + cfg.pgd_step_size * gradient.sign()
        adversarial = torch.clamp(adversarial, links - cfg.epsilon, links + cfg.epsilon).detach()
    return adversarial

def original_beta_gradient(phi, target, attacked_links):
    residual = (phi.detach() - target).cpu().numpy()
    energy = (-attacked_links.flatten(1).sum(1)).cpu().numpy()[:, None]
    average = energy.mean(0)
    centered = average[None] - energy
    gradient = 2 * (residual.mean(0)[None] * (centered[..., None] * residual[:, None]).mean(0)).mean(1)
    x2 = average ** 2 - (energy ** 2).mean(0)
    second = ((centered[..., None] ** 2 * residual[:, None]).mean(0) - x2[..., None] * residual.mean(0)[None]).mean(1)
    return float((gradient / (second + 1e-08))[0])

def train_baseline(model, target_beta, cfg, method, progress=True):
    if method not in ('original', 'at'):
        raise ValueError('Expected Original or AT baseline')
    if model not in ('ising', 'potts'):
        raise ValueError('Expected Ising or Potts model')
    if cfg.attack_objective != 'feature_norm' or cfg.attack_scope != 'model' or cfg.mapping != 'learned':
        raise ValueError('Baselines require feature_norm PGD, model-side attack and learned mapping')
    if cfg.lr_final_ratio != 1:
        raise ValueError('Baselines use loss-stagnation LR halving, not cosine decay')
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    mapping = OriginalMap(cfg.size).to(device)
    beta = nn.Parameter(torch.tensor([cfg.beta_init], device=device))
    optimizer = torch.optim.Adam(list(mapping.parameters()) + [beta], lr=cfg.lr)
    running_target = torch.zeros(13, device=device)
    best, stale, history = (float('inf'), 0, [])
    start = time.perf_counter()

    def draw(coupling):
        sampler = serial_samples if model == 'ising' else serial_samples_potts
        spins = sampler(cfg.size, coupling, cfg.samples, 10 * cfg.size ** 2, np.random.default_rng(rng.integers(2 ** 32 - 1)))
        if model == 'ising':
            links = np.stack((spins * np.roll(spins, -1, 2), spins * np.roll(spins, -1, 1)), 1)
        else:
            links = np.stack([np.cos(2 * np.pi * (spins - np.roll(spins, -1, axis)) / 3) for axis in (2, 1)], 1)
        return torch.as_tensor(links, dtype=torch.float32, device=device)
    for k in range(cfg.steps):
        data, samples = (draw(target_beta), draw(beta.detach().item()))
        adversarial = original_pgd(samples, mapping, cfg)
        with torch.no_grad():
            mapping(adversarial, pointwise=True)
        phi = features(mapping(adversarial))
        running_target = (features(data).mean(0) + running_target * k) / (k + 1)
        base = (phi.mean(0) - running_target).square().mean()
        loss = base
        if method == 'at':
            clean = (features(mapping(samples)).mean(0) - running_target).square().mean()
            loss = clean + base
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        beta.grad = torch.tensor([original_beta_gradient(phi, running_target, adversarial)], device=device)
        previous = beta.detach().clone()
        optimizer.step()
        with torch.no_grad():
            if k > 0:
                beta.copy_(0.9 * beta + 0.1 * previous)
            for hard in (True, True, False, False, False):
                F.gumbel_softmax(mapping.embedding, tau=0.5, hard=hard)
        step = k + 1
        if not torch.isfinite(beta).all() or not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite {method} result at step {step}')
        history.append({'step': step, 'beta': beta.item(), 'loss': loss.item(), 'base_loss': base.item(), 'flv': 0.0})
        if loss.item() < best:
            best, stale = (loss.item(), 0)
        else:
            stale += 1
            if stale % 100 == 0:
                optimizer.param_groups[0]['lr'] *= 0.5
        if progress and (step % 200 == 0 or step == cfg.steps):
            print(f'{model:5} beta={target_beta:.2f} {method:8} step={step:4d} estimate={beta.item():.4f} loss={loss.item():.4g} ({time.perf_counter() - start:.1f}s)', flush=True)
    return history
