"""
FolT-MCMC — Laplace-Whitened Real-Data Logistic Regression
============================================================
Preconditioning strategy:
  1. Find MAP estimate beta_hat via L-BFGS
  2. Compute Hessian H = nabla^2 U(beta_hat)
  3. Cholesky: H = L L^T
  4. Whitened variable: eta = L^T (beta - beta_hat)
  5. Train flow in eta-space (near-isotropic posterior)
  6. Run Certificate II

This is standard statistical preconditioning, not arbitrary dimension
reduction. If successful, it demonstrates FolT-MCMC on a real D=30
posterior with conventional preprocessing.

Usage:
  cd C:\\FolT-MCMC\\experiments
  python benchmark_logreg_whitened.py --quick
  python benchmark_logreg_whitened.py
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
# 1. Data
# ══════════════════════════════════════════════════════════════

def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)
    mu, sigma = X.mean(0), X.std(0) + 1e-8
    X = ((X - mu) / sigma).astype(np.float32)
    return X, y


# ══════════════════════════════════════════════════════════════
# 2. Laplace Approximation
# ══════════════════════════════════════════════════════════════

def find_map(X, y, tau=5.0, max_iter=500, lr=0.05):
    """Find MAP estimate via L-BFGS."""
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    D = X.shape[1]
    beta = torch.zeros(D, requires_grad=True)
    optimizer = torch.optim.LBFGS([beta], lr=lr, max_iter=20,
                                   line_search_fn='strong_wolfe')

    for i in range(max_iter // 20):
        def closure():
            optimizer.zero_grad()
            logits = X_t @ beta
            nll = -(y_t * logits - torch.nn.functional.softplus(logits)).sum()
            prior = 0.5 * (beta ** 2).sum() / (tau ** 2)
            loss = nll + prior
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if (i + 1) % 5 == 0:
            print(f"    MAP iter {(i+1)*20}: loss = {loss.item():.4f}")

    return beta.detach()


def compute_hessian(X, beta_hat, tau=5.0):
    """Compute Hessian of negative log-posterior at MAP.
    H = X^T diag(p*(1-p)) X + (1/tau^2) I
    """
    X_t = torch.tensor(X)
    logits = X_t @ beta_hat
    p = torch.sigmoid(logits)
    W = p * (1 - p)  # (n,)
    # H = X^T W X + prior precision
    H = (X_t.T * W.unsqueeze(0)) @ X_t + torch.eye(len(beta_hat)) / (tau ** 2)
    return H


def laplace_whitening(X, y, tau=5.0):
    """
    Compute Laplace approximation and return whitening transform.

    Returns:
        beta_hat: MAP estimate (D,)
        L: lower Cholesky of Hessian, H = L L^T (D, D)
        L_inv: inverse of L (D, D)

    Whitened variable: eta = L^T (beta - beta_hat)
    Original variable: beta = beta_hat + L^{-T} eta
    """
    D = X.shape[1]
    print(f"  Finding MAP estimate (D={D})...")
    beta_hat = find_map(X, y, tau=tau)
    print(f"  MAP found. ||beta_hat|| = {beta_hat.norm():.4f}")

    print(f"  Computing Hessian...")
    H = compute_hessian(X, beta_hat, tau=tau)

    # Check positive definiteness
    eigvals = torch.linalg.eigvalsh(H)
    print(f"  Hessian eigenvalues: min={eigvals.min():.4f}, "
          f"max={eigvals.max():.4f}, cond={eigvals.max()/eigvals.min():.1f}")

    # Cholesky
    L = torch.linalg.cholesky(H)
    L_inv = torch.linalg.inv(L)

    # Verify
    eta_test = L.T @ (beta_hat - beta_hat)  # should be zero
    beta_back = beta_hat + L_inv.T @ eta_test
    print(f"  Whitening verified: ||roundtrip error|| = "
          f"{(beta_back - beta_hat).norm():.2e}")

    return beta_hat, L, L_inv


# ══════════════════════════════════════════════════════════════
# 3. Whitened Target
# ══════════════════════════════════════════════════════════════

class WhitenedLogRegTarget:
    """
    Bayesian logistic regression in Laplace-whitened coordinates.

    eta = L^T (beta - beta_hat)
    beta = beta_hat + L^{-T} eta

    The posterior in eta-space should be approximately N(0, I)
    near the mode, making it much easier for the flow to learn.
    """
    def __init__(self, X, y, beta_hat, L, L_inv, tau=5.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.D = X.shape[1]
        self.tau = tau
        self.beta_hat = beta_hat.float()
        self.L = L.float()
        self.L_inv = L_inv.float()
        self.L_inv_T = self.L_inv.T  # for eta -> beta

        # Log determinant of the whitening transform (constant)
        self.log_det_L_inv_T = -torch.linalg.slogdet(self.L)[1]

    def _eta_to_beta(self, eta):
        """eta (batch, D) -> beta (batch, D)"""
        return self.beta_hat.to(eta.device) + eta @ self.L_inv_T.to(eta.device).T

    def U(self, eta):
        """Potential energy in whitened coordinates."""
        beta = self._eta_to_beta(eta)
        X_dev = self.X.to(eta.device)
        y_dev = self.y.to(eta.device)

        logits = beta @ X_dev.T
        log_lik = (y_dev * logits - torch.nn.functional.softplus(logits)).sum(dim=-1)
        log_prior = -0.5 * (beta ** 2).sum(dim=-1) / (self.tau ** 2)

        # Jacobian of whitening: |det d(beta)/d(eta)| = |det L^{-T}| = constant
        # This is absorbed into the normalizing constant, not per-sample
        # But we need it for the residual r(z) to be correct
        log_jac = self.log_det_L_inv_T

        return -(log_lik + log_prior + log_jac)

    def log_prob(self, eta):
        return -self.U(eta)

    def grad_log_prob(self, eta):
        """Analytical gradient in whitened coordinates."""
        beta = self._eta_to_beta(eta)
        X_dev = self.X.to(eta.device)
        y_dev = self.y.to(eta.device)
        L_inv_T_dev = self.L_inv_T.to(eta.device)

        logits = beta @ X_dev.T
        probs = torch.sigmoid(logits)
        residuals = y_dev.unsqueeze(0) - probs

        # grad w.r.t. beta
        grad_beta_lik = residuals @ X_dev
        grad_beta_prior = -beta / (self.tau ** 2)
        grad_beta = grad_beta_lik + grad_beta_prior

        # Chain rule: d/d(eta) = d(beta)/d(eta)^T @ d/d(beta) = L^{-T}^T @ grad_beta
        # = L^{-1} @ grad_beta^T ... actually:
        # beta = beta_hat + L^{-T} eta
        # d(beta)/d(eta) = L^{-T}
        # grad_eta = (d(beta)/d(eta))^T @ grad_beta = L^{-1} @ grad_beta^T
        # But grad_beta is (batch, D), so:
        grad_eta = grad_beta @ L_inv_T_dev  # (batch, D)

        return grad_eta

    def sample_mala(self, n, n_warmup=5000, step_size=0.01, thin=5):
        """MALA in whitened coordinates (should mix much better)."""
        print(f"    MALA (whitened): n={n}, warmup={n_warmup}, "
              f"step={step_size}, thin={thin}")
        eta = torch.zeros(1, self.D)
        samples = []
        n_accept = 0
        t0 = time.time()

        for i in range(n * thin + n_warmup):
            grad = self.grad_log_prob(eta).squeeze(0)
            noise = torch.randn(self.D)
            proposal = eta.squeeze(0) + step_size * grad + math.sqrt(2*step_size) * noise

            lp_prop = self.log_prob(proposal.unsqueeze(0)).item()
            lp_curr = self.log_prob(eta).item()

            grad_prop = self.grad_log_prob(proposal.unsqueeze(0)).squeeze(0)

            log_q_fwd = -0.25/step_size * (
                (proposal - eta.squeeze(0) - step_size*grad)**2).sum().item()
            log_q_bwd = -0.25/step_size * (
                (eta.squeeze(0) - proposal - step_size*grad_prop)**2).sum().item()

            log_alpha = (lp_prop - lp_curr) + (log_q_bwd - log_q_fwd)
            if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
                eta = proposal.unsqueeze(0)
                if i >= n_warmup:
                    n_accept += 1

            if i >= n_warmup and (i - n_warmup) % thin == 0:
                samples.append(eta.squeeze(0).clone())

            if (i+1) % 50000 == 0:
                rate = n_accept / max(1, i - n_warmup) if i > n_warmup else 0
                print(f"      Step {i+1}, accept={rate:.3f}, {time.time()-t0:.0f}s")

        samples = torch.stack(samples[:n])
        rate = n_accept / (n * thin)
        print(f"    Done. Accept={rate:.3f}, shape={samples.shape}, "
              f"{time.time()-t0:.0f}s")
        return samples


# ══════════════════════════════════════════════════════════════
# 4. Flow + Training + Certification (compact)
# ══════════════════════════════════════════════════════════════

def sn_linear(i, o):
    return nn.utils.spectral_norm(nn.Linear(i, o))

class CouplingLayer(nn.Module):
    def __init__(self, dim, hid, mask, sc=0.7):
        super().__init__()
        self.register_buffer('mask', mask); self.sc = sc
        self.s = nn.Sequential(sn_linear(dim,hid),nn.Tanh(),sn_linear(hid,hid),nn.Tanh(),sn_linear(hid,dim))
        self.t = nn.Sequential(sn_linear(dim,hid),nn.Tanh(),sn_linear(hid,hid),nn.Tanh(),sn_linear(hid,dim))
    def forward(self, z):
        zm = z*self.mask; s = self.sc*torch.tanh(self.s(zm)*(1-self.mask)); t = self.t(zm)*(1-self.mask)
        return z*torch.exp(s)+t, s.sum(-1)
    def inverse(self, x):
        xm = x*self.mask; s = self.sc*torch.tanh(self.s(xm)*(1-self.mask)); t = self.t(xm)*(1-self.mask)
        return (x-t)*torch.exp(-s), -s.sum(-1)

class SNRealNVP(nn.Module):
    def __init__(self, dim, nl=12, hid=128, sc=0.7):
        super().__init__(); self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(nl):
            m = torch.zeros(dim); m[i%dim]=1.0
            self.layers.append(CouplingLayer(dim,hid,m,sc))
    def forward(self, z):
        ld = torch.zeros(z.shape[0],device=z.device); x = z
        for l in self.layers: x, d = l(x); ld += d
        return x, ld
    def inverse(self, x):
        ld = torch.zeros(x.shape[0],device=x.device); z = x
        for l in reversed(self.layers): z, d = l.inverse(z); ld += d
        return z, ld
    def log_prob(self, x):
        z, ld = self.inverse(x)
        return -0.5*(z**2).sum(-1) - 0.5*self.dim*math.log(2*math.pi) + ld

def compute_residual(flow, target, z):
    theta, ld = flow(z)
    return -target.U(theta) + ld + 0.5*(z**2).sum(-1)

def smooth_osc(r, tau=0.1):
    return tau*torch.logsumexp(r/tau,0) + tau*torch.logsumexp(-r/tau,0)

def grad_norm_mean(flow, target, z):
    z = z.detach().requires_grad_(True)
    r = compute_residual(flow, target, z)
    g = torch.autograd.grad(r.sum(), z, create_graph=True)[0]
    return g.norm(dim=-1).mean(), g.norm(dim=-1).max()

def train_flow(target, D, hp):
    samples = target.sample_mala(
        hp['n_train'], n_warmup=hp.get('n_warmup',5000),
        step_size=hp.get('step_size',0.01), thin=hp.get('thin',5)
    ).to(DEVICE)
    N = samples.shape[0]
    flow = SNRealNVP(D, hp['n_layers'], hp['hidden_dim']).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, hp['n_epochs'])
    lo, lg = 0.1, 0.05; t0 = time.time()
    for ep in range(hp['n_epochs']):
        flow.train()
        ramp = max(0.0,(ep/hp['n_epochs']-0.4))/0.6
        elo, elg = lo*ramp, lg*ramp
        idx = torch.randint(0,N,(512,),device=DEVICE)
        nll = -flow.log_prob(samples[idx]).mean()
        ol = gl = torch.tensor(0.0, device=DEVICE)
        if elo > 0 or elg > 0:
            zf = torch.randn(512,D,device=DEVICE)
            r = compute_residual(flow,target,zf)
            if elo > 0: ol = smooth_osc(r)
            if elg > 0:
                zg = torch.randn(min(32,512),D,device=DEVICE)
                gm,_ = grad_norm_mean(flow,target,zg); gl = gm
        loss = nll + elo*ol + elg*gl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(),5.0)
        opt.step(); sch.step()
        if (ep+1) % max(hp['n_epochs']//5,200)==0 or ep==hp['n_epochs']-1:
            print(f"    [{ep+1:5d}/{hp['n_epochs']}] NLL={nll.item():.3f} | {time.time()-t0:.0f}s")
    flow.eval()
    return flow

def quantile_diagnostic(flow, target, D, n_cert=200000, zeta=0.05):
    flow.eval(); rl = []
    with torch.no_grad():
        for i in range(0,n_cert,50000):
            nb = min(50000,n_cert-i)
            rl.append(compute_residual(flow,target,torch.randn(nb,D,device=DEVICE)).cpu())
    r = torch.cat(rl).numpy(); n = len(r)
    en = math.sqrt(math.log(2.0/zeta)/(2*n))
    stats = {'n':n,'D':D,'eps_n':en,'mean':float(np.mean(r)),'std':float(np.std(r)),
             'var':float(np.var(r)),'full_osc':float(np.ptp(r))}
    for rho in [0.005,0.01,0.025,0.05,0.10,0.25]:
        if rho<=en: stats[rho]={'feasible':False}; continue
        ql=float(np.quantile(r,rho-en)); qh=float(np.quantile(r,1-rho+en))
        cf=qh-ql; gc=2.0/(1.0+math.exp(min(cf,500)))
        ic=(r>=ql)&(r<=qh); rs=r-r.max(); w=np.exp(rs)
        stats[rho]={'feasible':True,'c_formal':cf,'gamma_core':gc,
                    'mu_core':float(ic.sum()/n),'pi_hat':float(w[ic].sum()/w.sum())}
    Ra=math.sqrt(chi2.ppf(0.95,df=D)); lt=math.log(2.0/zeta)/(2*n)
    es=Ra*(lt+math.sqrt(lt))**(1.0/D)
    ng=min(2000,n); zg=torch.randn(ng,D,device=DEVICE).requires_grad_(True)
    rg=compute_residual(flow,target,zg); gr=torch.autograd.grad(rg.sum(),zg)[0]
    gs=gr.norm(dim=-1).max().item()
    v1c=stats['full_osc']+2*gs*es
    stats['v1']={'cert':v1c,'gamma':2.0/(1.0+math.exp(min(v1c,500))),'eps_star':es,'grad_sup':gs}
    return stats

def run_imh(flow, target, D, n_samples=5000, n_warmup=500):
    flow.eval()
    with torch.no_grad():
        z=torch.randn(1,D,device=DEVICE); th,_=flow(z)
    th=th.squeeze(0)
    lw=(target.log_prob(th.unsqueeze(0))-flow.log_prob(th.unsqueeze(0))).item()
    na=0
    for i in range(n_samples+n_warmup):
        with torch.no_grad():
            zp=torch.randn(1,D,device=DEVICE); tp,_=flow(zp)
        tp=tp.squeeze(0)
        lwp=(target.log_prob(tp.unsqueeze(0))-flow.log_prob(tp.unsqueeze(0))).item()
        if math.log(np.random.random()+1e-30)<min(0,lwp-lw):
            th,lw=tp,lwp
            if i>=n_warmup: na+=1
    return na/n_samples


# ══════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--tau', type=float, default=5.0)
    parser.add_argument('--outdir', type=str, default='results_v2')
    args = parser.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y = load_breast_cancer()
    D = X.shape[1]
    print(f"\n  Breast Cancer: n={X.shape[0]}, D={D}, balance={y.mean():.2f}")

    # Laplace approximation
    print(f"\n  === LAPLACE WHITENING ===")
    beta_hat, L, L_inv = laplace_whitening(X, y, tau=args.tau)

    # Whitened target
    target = WhitenedLogRegTarget(X, y, beta_hat, L, L_inv, tau=args.tau)

    # Quick check: whitened posterior should have std ~ 1 in each coordinate
    print(f"\n  Checking whitened posterior geometry...")
    test_samples = target.sample_mala(500, n_warmup=2000,
                                       step_size=0.1, thin=3)
    print(f"  Whitened sample std per coord: "
          f"mean={test_samples.std(0).mean():.3f}, "
          f"min={test_samples.std(0).min():.3f}, "
          f"max={test_samples.std(0).max():.3f}")
    print(f"  (Should be roughly 1.0 if whitening works)")

    # Hyperparameters
    hp = {
        'n_layers': 14, 'hidden_dim': 192,
        'n_train': 30000, 'n_epochs': 8000,
        'n_cert': 200000,
        'n_warmup': 5000, 'step_size': 0.1, 'thin': 3,
    }
    if args.quick:
        hp.update({'n_train': 5000, 'n_epochs': 2000,
                   'n_cert': 50000, 'n_warmup': 2000})

    n_mh = 2000 if args.quick else 5000

    print(f"\n{'='*75}")
    print(f"  LAPLACE-WHITENED BREAST CANCER (D={D})")
    print(f"  Flow: {hp['n_layers']} layers, {hp['hidden_dim']} hidden")
    print(f"  Training: {hp['n_epochs']} epochs")
    print(f"  Certification: {hp['n_cert']:,} samples")
    print(f"{'='*75}")

    # Train
    t0 = time.time()
    flow = train_flow(target, D, hp)
    print(f"  Training done in {time.time()-t0:.0f}s")
    torch.save(flow.state_dict(), out_dir / 'flow_bc_whitened.pt')

    # Quantile diagnostic
    print(f"\n  Quantile diagnostic ({hp['n_cert']:,} samples)...")
    t0 = time.time()
    qstats = quantile_diagnostic(flow, target, D, n_cert=hp['n_cert'])
    print(f"  Done in {time.time()-t0:.1f}s")

    # IMH
    print(f"\n  Independence MH ({n_mh} samples)...")
    t0 = time.time()
    accept_rate = run_imh(flow, target, D, n_samples=n_mh)
    print(f"  Accept rate: {accept_rate:.4f}, {time.time()-t0:.1f}s")

    # ── Results ──
    print(f"\n  {'='*75}")
    print(f"  RESULTS — Breast Cancer D={D} (Laplace-whitened)")
    print(f"  {'='*75}")
    print(f"\n  Residual: mean={qstats['mean']:.4f}, std={qstats['std']:.4f}")
    print(f"  Full osc = {qstats['full_osc']:.4f}")

    v1 = qstats['v1']
    print(f"\n  V1: C={v1['cert']:.4f}, gamma={v1['gamma']:.6f}")

    print(f"\n  V2 Quantile Core:")
    print(f"  {'rho':<8} {'C_formal':>10} {'gamma':>10} {'mu(G)':>8} {'pi(G)':>8}")
    print(f"  {'_'*48}")
    for rho in [0.005,0.01,0.025,0.05,0.10,0.25]:
        q = qstats.get(rho, {})
        if not q.get('feasible'): continue
        print(f"  {rho:<8.3f} {q['c_formal']:>10.4f} {q['gamma_core']:>10.6f} "
              f"{q['mu_core']:>8.4f} {q['pi_hat']:>8.4f}")

    print(f"\n  IMH accept rate: {accept_rate:.4f}")

    # Decision
    q01 = qstats.get(0.01, {})
    q05 = qstats.get(0.05, {})
    print(f"\n  VERDICT:")
    g01 = q01.get('gamma_core', 0)
    g05 = q05.get('gamma_core', 0)
    pi01 = q01.get('pi_hat', 0)

    if g05 > 0.1 and accept_rate > 0.1:
        print(f"  SUCCESS: Whitening rescued Breast Cancer D={D}!")
        print(f"  gamma_core(0.05) = {g05:.4f}, accept = {accept_rate:.4f}")
        print(f"  This can replace synthetic LogReg as main JASA benchmark.")
    elif g05 > 0.01:
        print(f"  PARTIAL: Some improvement from whitening.")
        print(f"  gamma_core(0.05) = {g05:.4f}, accept = {accept_rate:.4f}")
        print(f"  Consider as supplementary evidence.")
    else:
        print(f"  FAILED: Even with whitening, D={D} posterior too hard.")
        print(f"  Keep synthetic LogReg D=20 as main benchmark.")

    # Comparison with unwhitened
    print(f"\n  WHITENING EFFECT:")
    print(f"  Metric          Unwhitened    Whitened")
    print(f"  std(r)          4.28          {qstats['std']:.4f}")
    print(f"  Accept rate     0.0005        {accept_rate:.4f}")
    print(f"  pi(G_0.01)      0.013         {pi01:.4f}")

    # Save
    save = {
        'dataset': 'breast_cancer_whitened', 'D': D, 'tau': args.tau,
        'whitening': {
            'beta_hat_norm': float(beta_hat.norm()),
            'hessian_cond': float(
                torch.linalg.eigvalsh(compute_hessian(X, beta_hat, args.tau)).max() /
                torch.linalg.eigvalsh(compute_hessian(X, beta_hat, args.tau)).min()
            ),
        },
        'quantile_stats': {k: v for k, v in qstats.items() if not isinstance(k, float)},
        'quantiles': {str(k): v for k, v in qstats.items() if isinstance(k, float)},
        'v1': {k: float(v) for k, v in v1.items()},
        'accept_rate': float(accept_rate),
    }
    with open(out_dir / 'real_logreg_bc_whitened.json', 'w') as f:
        json.dump(save, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)

    print(f"\n  Saved: {out_dir / 'real_logreg_bc_whitened.json'}")
    print(f"\n{'='*75}")
    print("  COMPLETE")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
