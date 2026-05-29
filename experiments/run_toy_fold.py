"""
FolT-MCMC toy experiment: symmetric double banana.

Compares:
  (A) CerT-MCMC in original theta-space (2 modes)
  (B) CerT-MCMC in folded z-space (1 mode after reflection fold)

Metrics:
  - Oscillation bound: osc(log q/pi) vs osc(log q_F/pi_F)
  - Certified spectral gap: gamma vs gamma_F
  - ESS per gradient evaluation
  - Mixing visualization
"""

# TODO: import from existing CerT-MCMC-v2 modules (transport, certification, MH kernel)
# TODO: wire up ReflectionFold + SymmetricDoubleBanana
# TODO: training loop: train transport on folded target
# TODO: certification: compute oscillation bound + spectral gap in both spaces
# TODO: sampling: run chains in both spaces, compute ESS
# TODO: visualization: trace plots, 2D scatter, oscillation heatmaps

if __name__ == "__main__":
    pass
