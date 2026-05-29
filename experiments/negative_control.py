"""
FolT-MCMC V2 — Negative Control: Certificate II Discriminative Power
=====================================================================
Critical experiment for JASA: prove that Certificate II does NOT
blindly certify any flow. An under-trained or poorly configured flow
must produce a clearly worse core certificate than a well-trained one.

Design:
  Same target (banana D=10), three flow conditions:
    A. Well-trained:  FolT-OG-Anneal, 8000 epochs, full penalties
    B. Under-trained: NLL-only, 200 epochs
    C. Misspecified:  Too-small architecture (4 layers, 32 hidden)

If Certificate II gives gamma_A >> gamma_B and gamma_A >> gamma_C,
it proves discriminative power.

Usage:
  conda activate lcnf
  cd C:\\FolT-MCMC\\experiments
  python negative_control.py
  python negative_control.py --quick
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import argparse
import math
from scipy.stats import chi2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# Target & Flow (same as other experiments)
# ══════════════════════════════════════════════════════════════

class BananaTarget:
    def __init__(self, D=10, sigma1=2.0, sigma2=1.0, b=0.1):
        self.D = D
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.b = b

    def U(self, theta):
        t1, t2 = theta[:, 0], theta[:, 1]
        u = t1**2 / (2 * self.sigma1**2) + (t2 - self.b * t1**2)**2 / (2 * self.sigma2**2)
        if self.D > 2:
            u = u + 0.5 * (theta[:, 2:]**2).sum(dim=-1)
        return u

    def log_prob(self, theta):
        return -self.U(theta)

    def sample_mala(self, n, n_warmup=5000, step_size=0.05, thin=5):
        theta = torch.randn(self.D)
        samples = []
        for i in range(n * thin + n_warmup):
            theta_r = theta.detach().requires_grad_(True)
            lp = self.log_prob(theta_r.unsqueeze(0)).squeeze()
            lp.backward()
            grad = theta_r.grad.detach()
            noise = torch.randn(self.D)
            proposal = theta + step_size * grad + math.sqrt(2 * step_size) * noise
            lp_prop = self.log_prob(proposal.unsqueeze(0)).squeeze().item()
            lp_curr = self.log_prob(theta.unsqueeze(0)).squeeze().item()
            prop_r = proposal.detach().requires_grad_(True)
            lp_p = self.log_prob(prop_r.unsqueeze(0)).squeeze()
            lp_p.backward()
            grad_prop = prop_r.grad.detach()
            log_q_fwd = -0.25/step_size * ((proposal - theta - step_size*grad)**2).sum().item()
            log_q_bwd = -0.25/step_size * ((theta - proposal - step_size*grad_prop)**2).sum().item()
            log_alpha = (lp_prop - lp_curr) + (log_q_bwd - log_q_fwd)
            if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
                theta = proposal.detach()
            if i >= n_warmup and (i - n_warmup) % thin == 0:
                samples.append(theta.clone())
        return torch.stack(samples[:n])


def sn_linear(in_f, out_f):
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f))

class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask, scale_clip=0.7):
        super().__init__()
        self.register_buffer('mask', mask)
        self.scale_clip = scale_clip
        self.s_net = nn.Sequential(
            sn_linear(dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, dim))
        self.t_net = nn.Sequential(
            sn_linear(dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, dim))

    def forward(self, z):
        z_m = z * self.mask
        s = self.s_net(z_m) * (1 - self.mask)
        s = self.scale_clip * torch.tanh(s)
        t = self.t_net(z_m) * (1 - self.mask)
        return z * torch.exp(s) + t, s.sum(dim=-1)

    def inverse(self, theta):
        th_m = theta * self.mask
        s = self.s_net(th_m) * (1 - self.mask)
        s = self.scale_clip * torch.tanh(s)
        t = self.t_net(th_m) * (1 - self.mask)
        return (theta - t) * torch.exp(-s), -s.sum(dim=-1)

class SNRealNVP(nn.Module):
    def __init__(self, dim=2, n_layers=8, hidden_dim=64, scale_clip=0.7):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % dim] = 1.0
            self.layers.append(CouplingLayer(dim, hidden_dim, mask, scale_clip))

    def forward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det

    def inverse(self, theta):
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        z = theta
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det += ld
        return z, log_det

    def log_prob(self, theta):
        z, log_det = self.inverse(theta)
        log_q0 = -0.5 * (z**2).sum(dim=-1) - 0.5 * self.dim * math.log(2*math.pi)
        return log_q0 + log_det


def compute_residual(flow, target, z):
    theta, log_det = flow(z)
    return -target.U(theta) + log_det + 0.5 * (z**2).sum(dim=-1)

def smooth_oscillation(r, tau=0.1):
    return tau * torch.logsumexp(r / tau, dim=0) + tau * torch.logsumexp(-r / tau, dim=0)

def gradient_norm_mean(flow, target, z):
    z = z.detach().requires_grad_(True)
    r = compute_residual(flow, target, z)
    grad_r = torch.autograd.grad(r.sum(), z, create_graph=True)[0]
    norms = grad_r.norm(dim=-1)
    return norms.mean(), norms.max()


# ══════════════════════════════════════════════════════════════
# Training variants
# ══════════════════════════════════════════════════════════════

def train_flow(target, train_samples, D, n_epochs, n_layers, hidden_dim,
               use_cert_loss=True, label='Flow'):
    """Train a flow with configurable architecture and loss."""
    flow = SNRealNVP(dim=D, n_layers=n_layers,
                     hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_dev = train_samples.to(DEVICE)
    N = train_dev.shape[0]

    lambda_o, lambda_g, tau = 0.1, 0.05, 0.1
    t0 = time.time()

    for epoch in range(n_epochs):
        flow.train()

        # FolT-OG-Anneal: ramp after 40%
        if use_cert_loss:
            ramp = max(0.0, (epoch / n_epochs - 0.4)) / 0.6
            eff_lo, eff_lg = lambda_o * ramp, lambda_g * ramp
        else:
            eff_lo, eff_lg = 0.0, 0.0

        idx = torch.randint(0, N, (512,), device=DEVICE)
        nll = -flow.log_prob(train_dev[idx]).mean()

        osc_loss = grad_loss = torch.tensor(0.0, device=DEVICE)
        if eff_lo > 0 or eff_lg > 0:
            z_f = torch.randn(512, D, device=DEVICE)
            r = compute_residual(flow, target, z_f)
            if eff_lo > 0:
                osc_loss = smooth_oscillation(r, tau=tau)
            if eff_lg > 0:
                z_g = torch.randn(min(32, 512), D, device=DEVICE)
                gm, _ = gradient_norm_mean(flow, target, z_g)
                grad_loss = gm

        loss = nll + eff_lo * osc_loss + eff_lg * grad_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % max(n_epochs // 5, 50) == 0 or epoch == n_epochs - 1:
            print(f"    [{label}] [{epoch+1:5d}/{n_epochs}] "
                  f"NLL={nll.item():.3f} | {time.time()-t0:.0f}s")

    flow.eval()
    return flow


# ══════════════════════════════════════════════════════════════
# Quantile diagnostic + IMH
# ══════════════════════════════════════════════════════════════

def quantile_diagnostic(flow, target, D, n_cert=200000, zeta=0.05):
    flow.eval()
    batch_size = 50000
    r_list = []
    with torch.no_grad():
        for i in range(0, n_cert, batch_size):
            nb = min(batch_size, n_cert - i)
            z = torch.randn(nb, D, device=DEVICE)
            r = compute_residual(flow, target, z)
            r_list.append(r.cpu())
    r_all = torch.cat(r_list).numpy()
    n = len(r_all)
    eps_n = math.sqrt(math.log(2.0 / zeta) / (2 * n))

    stats = {
        'n': n, 'mean': float(np.mean(r_all)),
        'std': float(np.std(r_all)), 'var': float(np.var(r_all)),
        'full_osc': float(np.ptp(r_all)),
    }

    rho_levels = [0.01, 0.05, 0.10, 0.25]
    for rho in rho_levels:
        if rho <= eps_n:
            stats[rho] = {'feasible': False}
            continue
        q_lo_w = float(np.quantile(r_all, rho - eps_n))
        q_hi_w = float(np.quantile(r_all, 1 - rho + eps_n))
        c_formal = q_hi_w - q_lo_w
        gamma_core = 2.0 / (1.0 + math.exp(min(c_formal, 500)))
        in_core = (r_all >= q_lo_w) & (r_all <= q_hi_w)
        r_shifted = r_all - r_all.max()
        w = np.exp(r_shifted)
        pi_hat = float(w[in_core].sum() / w.sum())
        stats[rho] = {
            'feasible': True,
            'c_formal': c_formal,
            'gamma_core': gamma_core,
            'mu_core': float(in_core.sum() / n),
            'pi_hat': pi_hat,
        }

    return stats


def run_imh(flow, target, D, n_samples=3000, n_warmup=500):
    """Quick independence MH to get acceptance rate."""
    flow.eval()
    with torch.no_grad():
        z = torch.randn(1, D, device=DEVICE)
        theta, _ = flow(z)
    theta = theta.squeeze(0)
    log_w = (target.log_prob(theta.unsqueeze(0)) - flow.log_prob(theta.unsqueeze(0))).item()
    n_accept = 0

    for i in range(n_samples + n_warmup):
        with torch.no_grad():
            z_prop = torch.randn(1, D, device=DEVICE)
            theta_prop, _ = flow(z_prop)
        theta_prop = theta_prop.squeeze(0)
        log_w_prop = (target.log_prob(theta_prop.unsqueeze(0))
                     - flow.log_prob(theta_prop.unsqueeze(0))).item()
        log_alpha = log_w_prop - log_w
        if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
            theta, log_w = theta_prop, log_w_prop
            if i >= n_warmup:
                n_accept += 1

    return n_accept / n_samples


# ══════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='results_v2')
    args = parser.parse_args()

    D = args.dim
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Shared training data ──
    target = BananaTarget(D=D)
    n_train = 50000 if not args.quick else 10000
    n_cert = 200000 if not args.quick else 50000
    n_mh = 3000 if not args.quick else 1000

    print(f"\n{'='*75}")
    print(f"  NEGATIVE CONTROL — Banana D={D}")
    print(f"  Training samples: {n_train}, Cert samples: {n_cert:,}")
    print(f"{'='*75}")

    print(f"\n  Generating {n_train} training samples...")
    t0 = time.time()
    train_samples = target.sample_mala(n_train, n_warmup=max(5000, n_train//2))
    print(f"  Done in {time.time()-t0:.0f}s")

    # ── Three flow conditions ──
    conditions = {
        'A_well_trained': {
            'desc': 'FolT-OG-Anneal, 12 layers, 128 hidden, 8000 epochs',
            'n_layers': 12, 'hidden_dim': 128,
            'n_epochs': 8000 if not args.quick else 2000,
            'use_cert_loss': True,
        },
        'B_under_trained': {
            'desc': 'NLL-only, 12 layers, 128 hidden, 200 epochs',
            'n_layers': 12, 'hidden_dim': 128,
            'n_epochs': 200 if not args.quick else 100,
            'use_cert_loss': False,
        },
        'C_misspecified': {
            'desc': 'NLL-only, 4 layers, 32 hidden, 2000 epochs',
            'n_layers': 4, 'hidden_dim': 32,
            'n_epochs': 2000 if not args.quick else 500,
            'use_cert_loss': False,
        },
    }

    all_results = {}

    for cond_name, cfg in conditions.items():
        print(f"\n  {'─'*60}")
        print(f"  Condition: {cond_name}")
        print(f"  {cfg['desc']}")
        print(f"  {'─'*60}")

        # Train
        t0 = time.time()
        flow = train_flow(
            target, train_samples, D,
            n_epochs=cfg['n_epochs'],
            n_layers=cfg['n_layers'],
            hidden_dim=cfg['hidden_dim'],
            use_cert_loss=cfg['use_cert_loss'],
            label=cond_name,
        )
        train_time = time.time() - t0
        print(f"  Training: {train_time:.0f}s")

        # Quantile diagnostic
        print(f"  Quantile diagnostic ({n_cert:,} samples)...")
        t0 = time.time()
        qstats = quantile_diagnostic(flow, target, D, n_cert=n_cert)
        print(f"  Done in {time.time()-t0:.1f}s")

        # IMH acceptance rate
        print(f"  Independence MH ({n_mh} samples)...")
        accept_rate = run_imh(flow, target, D, n_samples=n_mh)
        print(f"  Accept rate: {accept_rate:.4f}")

        all_results[cond_name] = {
            'config': cfg['desc'],
            'std_r': qstats['std'],
            'var_r': qstats['var'],
            'full_osc': qstats['full_osc'],
            'accept_rate': accept_rate,
            'quantiles': {},
        }

        for rho in [0.01, 0.05, 0.10, 0.25]:
            q = qstats.get(rho, {})
            if q.get('feasible'):
                all_results[cond_name]['quantiles'][str(rho)] = {
                    'c_formal': q['c_formal'],
                    'gamma_core': q['gamma_core'],
                    'mu_core': q['mu_core'],
                    'pi_hat': q['pi_hat'],
                }

    # ── Summary ──
    print(f"\n{'='*75}")
    print(f"  NEGATIVE CONTROL SUMMARY — Banana D={D}")
    print(f"{'='*75}")

    # Header
    print(f"\n  {'Condition':<22} {'std(r)':>8} {'full_osc':>10} {'accept':>8}"
          f"  {'C_0.01':>8} {'gamma_01':>10} {'C_0.05':>8} {'gamma_05':>10}")
    print(f"  {'─'*98}")

    for cond_name, res in all_results.items():
        q01 = res['quantiles'].get('0.01', {})
        q05 = res['quantiles'].get('0.05', {})
        c01 = q01.get('c_formal', float('inf'))
        g01 = q01.get('gamma_core', 0)
        c05 = q05.get('c_formal', float('inf'))
        g05 = q05.get('gamma_core', 0)
        print(f"  {cond_name:<22} {res['std_r']:>8.4f} {res['full_osc']:>10.4f} "
              f"{res['accept_rate']:>8.4f}  {c01:>8.4f} {g01:>10.6f} "
              f"{c05:>8.4f} {g05:>10.6f}")

    # Discrimination ratios
    a = all_results.get('A_well_trained', {})
    b = all_results.get('B_under_trained', {})
    c = all_results.get('C_misspecified', {})

    a_g01 = a.get('quantiles', {}).get('0.01', {}).get('gamma_core', 0)
    b_g01 = b.get('quantiles', {}).get('0.01', {}).get('gamma_core', 0)
    c_g01 = c.get('quantiles', {}).get('0.01', {}).get('gamma_core', 0)

    a_g05 = a.get('quantiles', {}).get('0.05', {}).get('gamma_core', 0)
    b_g05 = b.get('quantiles', {}).get('0.05', {}).get('gamma_core', 0)
    c_g05 = c.get('quantiles', {}).get('0.05', {}).get('gamma_core', 0)

    print(f"\n  DISCRIMINATION RATIOS (rho=0.01):")
    if b_g01 > 0:
        print(f"    Well-trained / Under-trained: "
              f"{a_g01:.4f} / {b_g01:.4f} = {a_g01/b_g01:.1f}x")
    else:
        print(f"    Well-trained / Under-trained: {a_g01:.4f} / {b_g01:.6f}")
    if c_g01 > 0:
        print(f"    Well-trained / Misspecified:  "
              f"{a_g01:.4f} / {c_g01:.4f} = {a_g01/c_g01:.1f}x")
    else:
        print(f"    Well-trained / Misspecified:  {a_g01:.4f} / {c_g01:.6f}")

    print(f"\n  DISCRIMINATION RATIOS (rho=0.05):")
    if b_g05 > 0:
        print(f"    Well-trained / Under-trained: "
              f"{a_g05:.4f} / {b_g05:.4f} = {a_g05/b_g05:.1f}x")
    if c_g05 > 0:
        print(f"    Well-trained / Misspecified:  "
              f"{a_g05:.4f} / {c_g05:.4f} = {a_g05/c_g05:.1f}x")

    # Verdict
    print(f"\n  VERDICT:")
    if a_g01 > 10 * max(b_g01, 1e-10) and a_g01 > 10 * max(c_g01, 1e-10):
        print(f"  STRONG DISCRIMINATION: Certificate II clearly separates "
              f"good and bad flows (>10x ratio)")
    elif a_g01 > 3 * max(b_g01, 1e-10) or a_g01 > 3 * max(c_g01, 1e-10):
        print(f"  MODERATE DISCRIMINATION: Certificate II shows meaningful "
              f"differences (>3x ratio)")
    else:
        print(f"  WEAK DISCRIMINATION: Certificate II does not clearly "
              f"separate flow quality")

    # Also check acceptance rate discrimination
    print(f"\n  ACCEPTANCE RATE CHECK:")
    print(f"    Well-trained: {a.get('accept_rate', 0):.4f}")
    print(f"    Under-trained: {b.get('accept_rate', 0):.4f}")
    print(f"    Misspecified: {c.get('accept_rate', 0):.4f}")

    # Save
    with open(out_dir / f'negative_control_D{D}.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Plot ──
    make_negative_control_plot(all_results, D, out_dir)

    print(f"\n  Results saved to {out_dir}/")
    print(f"\n{'='*75}")
    print("  NEGATIVE CONTROL COMPLETE")
    print(f"{'='*75}")


def make_negative_control_plot(all_results, D, out_dir):
    """Bar chart comparing conditions."""
    conds = list(all_results.keys())
    labels = ['Well-trained\n(FolT-OG-Anneal)', 'Under-trained\n(200 ep, NLL only)',
              'Misspecified\n(4 layers, 32 hid)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Negative Control: Certificate II Discriminative Power (banana $D={D}$)',
                 fontsize=13)

    x = np.arange(len(conds))
    colors = ['#44AA66', '#DDAA44', '#CC4444']

    # Panel 1: Core gamma at rho=0.01
    gammas_01 = [all_results[c].get('quantiles', {}).get('0.01', {}).get('gamma_core', 0)
                 for c in conds]
    axes[0].bar(x, gammas_01, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_title(r'Core $\gamma$ ($\rho = 0.01$)', fontsize=11)
    axes[0].set_ylabel(r'$\gamma_{\mathrm{core}}$')
    for i, v in enumerate(gammas_01):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Panel 2: Core gamma at rho=0.05
    gammas_05 = [all_results[c].get('quantiles', {}).get('0.05', {}).get('gamma_core', 0)
                 for c in conds]
    axes[1].bar(x, gammas_05, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_title(r'Core $\gamma$ ($\rho = 0.05$)', fontsize=11)
    axes[1].set_ylabel(r'$\gamma_{\mathrm{core}}$')
    for i, v in enumerate(gammas_05):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Panel 3: acceptance rate only (simpler, no ambiguity)
    accepts = [all_results[c]['accept_rate'] for c in conds]
    axes[2].bar(x, accepts, color=colors, alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_title('Acceptance rate', fontsize=11)
    axes[2].set_ylabel('Accept rate')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(accepts):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / f'negative_control_D{D}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {out_dir / f'negative_control_D{D}.png'}")


if __name__ == '__main__':
    main()
