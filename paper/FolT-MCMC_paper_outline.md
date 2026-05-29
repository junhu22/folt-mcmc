# FolT-MCMC Paper Outline — JRSS-B

**Title:** Folded Transport MCMC: Certified Sampling of Symmetric Multi-Modal Posteriors via Quotient-Space Reduction

**Author:** Jun Hu

**Target:** Journal of the Royal Statistical Society, Series B

**Estimated length:** 28-32 pages (main) + 10-15 pages (supplement)

**Style notes:** JRSS-B 要求简洁、precise、每句话有信息量。Introduction ≤ 1.5 pages。数学记号统一。Discussion paper 潜力。

---

## Section 1: Introduction (1.5 pages)

**开篇（2段）：** Transport MCMC 的成功 + 现有认证框架（LCNF, CerT-MCMC）的 limitation：multimodality 导致 oscillation bound 随 mode 数 / 维度崩塌。

**Problem statement（1段）：** 对称多模态后验（label-switching, reflection symmetry）是 Bayesian inference 中的核心难题。现有方法（post-hoc relabeling, tempering）不解决 transport 层面的问题。

**Our contribution（1段，3 points）：**
1. FolT-MCMC：quotient-space independence sampler with symmetrized proposal，消除 cross-mode oscillation
2. 理论：LCNF certificate 在 quotient metric 下保持 full ball-volume，certified spectral gap lower bound 严格改善
3. 实验：γ̲_F/γ̲_U 从 2× 到 145×；folded certificate 对维度和 mode 数近乎不变

**Paper structure（1段）：** 简要路线图。

**关键 figure：** 无（Introduction 不放 figure）。

---

## Section 2: Background (2.5 pages)

### 2.1 Transport MCMC and independence MH (0.7 page)

- Transport map T: R^d → R^d，base N(0,I)，proposal q = T_# N(0,I)
- Independence MH kernel
- Mengersen-Tweedie bound: γ ≥ 2/(1+exp(osc(log π/q)))
- 引用 Hoffman et al. (2019), Parno & Marzouk (2018), 自己的 MSSP1/MSSP2

### 2.2 LCNF certification framework (1 page)

- SN-RealNVP: per-layer Lipschitz bound (Theorem, 引用 LCNF)
- Empirical oscillation theorem: osc ≤ ô_n + 2Mε* (引用 LCNF Thm 5.1)
- Three assumptions: HPD set, smooth boundary, local regularity
- Covering lemma + spectral gap corollary
- 说明 limitation：multimodal targets 导致 ô_n 由 cross-mode gap 主导

### 2.3 Symmetry in Bayesian inference (0.8 page)

- Label-switching in mixture models: S_m permutation symmetry
- Reflection symmetry in structural identification
- 现有方法简述：Stephens (2000) pivotal reordering, Celeux et al. (2000), Frühwirth-Schnatter (2001) random permutation sampler
- 关键区别：这些都是 post-hoc，FolT 是 pre-sampling

**关键引用：** LCNF (Hu 2026), CerT-MCMC (Hu 2026), Mengersen-Tweedie (1996), Hoffman (2019), Stephens (2000), Celeux (2000), Frühwirth-Schnatter (2001)

---

## Section 3: FolT-MCMC Framework (3 pages)

### 3.1 Quotient target (0.8 page)

- Definition 1: G-invariant target, fundamental domain D, π_F = s·π|_D
- Proposition 1 (HPD identity): K_α^F = K_α ∩ D (a.e.)
  - Proof: distribution identity via G-invariance
  - Corollary: diam ≤, π_min^F = s·π_min^D

### 3.2 Quotient proposal and log-density ratio (0.7 page)

- Definition 2: q_F(z) = Σ_{g∈G} q_T(g·z)
- Normalization proof
- h_F = log π_F − log q_F
- Remark: 和原空间 MH 的关系（不等价，除非 q_T 是 G-symmetric）

### 3.3 FolT-MCMC algorithm (0.7 page)

- Algorithm box:
  1. Train SN-RealNVP T on π_F (folded training data)
  2. Define q_F via quotient proposal
  3. Run independence MH on D with target π_F, proposal q_F
  4. Certify via quotient-metric LCNF
- 两个实例：ReflectionFold (Z₂), PermutationFold (S_m)

### 3.4 Quotient metric (0.8 page)

- Definition 3: d_G([x],[y]) = min_{g∈G} ||x−gy||
- Lemma 4 (Lipschitz): |h_F(x)−h_F(y)| ≤ M·d_G(x,y)
  - Proof: G-invariance of h_F + MVT + min over g
- Property (P3): quotient ball volume = full ball volume (away from fixed-point strata)
- Condition (G): generic stabilizer on K_α^F

**Figure 1:** Schematic of FolT-MCMC pipeline:
原空间 π (k modes) → fold → 折叠空间 π_F (1 mode) → transport T_F → MH → certify → unfold
（简洁的 2-panel diagram，左边是 multi-modal，右边是 single-modal after folding）

---

## Section 4: Theoretical Guarantees (3.5 pages)

### 4.1 Folded LCNF certificate (1 page)

- Verification of LCNF Assumptions 1'-3' on (π_F, D, d_G)
  - Assumption 1': Proposition 1 (HPD identity)
  - Assumption 2': Lemma 1 (boundary structure, corner set codim ≥ 2)
  - Assumption 3': Lemma 2 (C^1 regularity of h_F)
- **Theorem 1 (FolT-MCMC certificate):** 
  LCNF theorem applies on quotient space → osc ≤ ô_n^F + 2M_F ε*_F

### 4.2 Covering radius improvement (0.8 page)

- Quotient covering lemma (Lemma 1')
  - N_G ≤ N_E
  - Ball mass: π_min^F · V_d · ε^d under condition (G)
- **Theorem 2 (ε*_F ≤ ε*_U):**
  Monotonicity of covering condition: D_F ≤ D_U, π_min^F ≥ s·π_min → ε*_F ≤ ε*_U
  - Proof 完整写出（6-8 行）

### 4.3 Certified gap improvement (1.2 page)

- Conditions (M'), (W), (R): cross-mode deficiency, main-mode fit, folded-proposal residual
- **Lemma 3' (Oscillation gap):** ô_n − ô_n^F ≥ min_j Δ_j − 2r₁ − r_F − O(√...)
  - Proof sketch（main steps，details in supplement）
- **Theorem 3 (Certified improvement):**
  Under (M')+(W)+(R)+(S): C_F < C_U, γ̲_F/γ̲_U > 1
  - Union bound: prob ≥ 1−δ_U−δ_F

### 4.4 Discussion of conditions (0.5 page)

- When does (M') hold: finite-capacity SN-RealNVP, large mode separation
- When does (G) hold: well-separated modes, HPD set doesn't contain fixed-point strata
- When does folding **not** help: perfectly expressive flow, single-modal target, fold boundary through high-density region (banana diagnostic)

**Table 1:** Conditions summary — 每个条件的含义、何时成立、何时不成立

---

## Section 5: Experiments (5 pages)

### 5.1 Setup (0.5 page)

- Architecture: SN-RealNVP with OscReg (same as LCNF/CerT-MCMC)
- Training: NLL + annealed oscillation/gradient regularization
- Certification: quantile-core certificate (CerT-MCMC v2)
- Hardware: RTX 4090, conda env lcnf
- Code: github.com/junhu22/FolT-MCMC

### 5.2 Diagnostic: fold boundary location (1 page)

**Experiment:** Asymmetric double banana (D=2) — fold boundary through high-density region

**Purpose:** 展示 folding 不是万能的；fold boundary 位置是关键 design choice。

**Results:**
- full osc: folded 58.4 > unfolded 30.7 (反升)
- qc γ (ρ=0.05): folded 0.320 > unfolded 0.272 (仍有温和改善)
- NLL: folded 3.34 < unfolded 4.34 (transport 拟合更好)

**Message:** 当 fold boundary 穿过高密度区，full oscillation 被 boundary 尖峰抬高。但 quantile-core certificate 仍然改善（因为 boundary 尖峰被 ρ-trimming 排除）。

**Figure 2 (2 panels):**
(a) banana target contour + fold line
(b) oscillation heatmap: unfolded vs folded

**Table 2:** banana 完整数值（full osc, qc γ at ρ=0.025/0.05/0.10, NLL, ESS, acceptance）

### 5.3 Dimension scaling: well-separated mixture (1.5 pages)

**Experiment:** Gaussian mixture (s=2, ReflectionFold), D=2,5,10,20

**Purpose:** Headline result — certified γ 对维度近乎不变。

**Results:**
| D | γ̲_U | γ̲_F | ratio |
|---|------|------|-------|
| 2 | 0.402 | 0.936 | 2.3× |
| 5 | 0.094 | 0.941 | 10× |
| 10| 0.016 | 0.922 | 59× |
| 20| 0.016 | 0.902 | 57× |

**Figure 3 (headline figure, 1 panel):**
γ̲ vs D，两条线（unfolded 崩塌 vs folded flat）

**Figure 4 (2 panels):**
(a) qc oscillation vs D
(b) NLL vs D

**Table 3:** 完整数值

### 5.4 Label-switching scaling (1.5 pages)

**Experiment:** m-component Gaussian mixture (PermutationFold), configs k2p2/k3p2/k3p4/k4p2

**Purpose:** 验证 PermutationFold + 展示 k! 是 certifiability 的主要瓶颈（不是 D）。

**Results:** γ̲_F/γ̲_U 从 2.2× 到 145×。k3_p2 vs k3_p4 分离 D 和 k! 的效应。

**Figure 5 (1 panel):**
γ̲ vs s=|G|，两条线

**Table 4:** 完整数值 + D vs k! 分离分析

**Key finding:** "崩塌主要由 mode 数 s 驱动，维度 D 是次要因素"

### 5.5 Comparison with random permutation sampler (0.5 page)

**Experiment:** k=3 mixture, 对比 Frühwirth-Schnatter random permutation sampler

**Purpose:** 和最常用的 label-switching 处理方法做 empirical comparison。

**Metrics:** ESS, mixing time, acceptance rate

**Note:** 这个实验还没跑，需要 Phase 5 的 Claude Code 指令。

**Table 5:** FolT-MCMC vs random permutation sampler

---

## Section 6: Application to Structural Modal Identification (2 pages)

**Note:** 这个 section 还没做，需要从 Paper 4 提取简化版。

### 6.1 Problem setup (0.5 page)

- 3 closely-spaced modes of a building (TX2a, TX2b, TY2)
- Bayesian OMA likelihood (simplified Whittle)
- 每个 mode: (frequency, damping), D=6, S₃ permutation symmetry (s=6)

### 6.2 Results (1 page)

- Unfolded: chain gets stuck / label-switches slowly
- FolT-MCMC: sorted-space sampling, efficient mixing
- Certified γ̲ comparison

### 6.3 Discussion (0.5 page)

- Approximate symmetry: modes not exactly exchangeable, but close enough
- Implications for structural health monitoring

**Figure 6 (2 panels):**
(a) PSD showing 3 closely-spaced modes
(b) trace plot: unfolded vs folded

---

## Section 7: Discussion (1.5 pages)

### 7.1 When does folding help? (0.5 page)

- Design principle: fold boundary in low-density region
- banana vs mixture 对比总结
- Practical guideline: check π(∂D) / π_max before applying FolT

### 7.2 Relation to existing work (0.5 page)

- vs equivariant NFs (Köhler et al. 2020): 保持 vs 消除对称性，互补
- vs post-hoc relabeling (Stephens 2000): pre-sampling vs post-hoc
- vs tempering/replica exchange: geometric vs thermal barrier removal

### 7.3 Limitations and future work (0.5 page)

- Approximate symmetry: extension to ε-approximate G-invariance
- Learned folding map: for unknown symmetry groups
- Beyond finite groups: continuous symmetries (Lie groups)
- Scalability: very high-dimensional structural models

---

## Supplement (10-15 pages)

### A. Proofs

- A.1 Proposition 1 (HPD identity) — full proof
- A.2 Lemma 1' (Quotient covering) — complete proof with all LCNF steps
- A.3 Theorem 2 (ε* improvement) — full proof
- A.4 Lemma 3' (Oscillation gap) — complete proof
- A.5 Lemma 4 (Quotient Lipschitz) — full proof

### B. Experimental details

- B.1 Architecture and hyperparameters per experiment
- B.2 Training curves
- B.3 Full quantile-core tables (all ρ values)
- B.4 Additional scatter plots and trace plots

### C. Boundary analysis

- C.1 Fold boundary density computation for Gaussian mixture
- C.2 When K_α^F ∩ ∂D = ∅: precise condition δ > 2σ√(χ²_{d,1−α})
- C.3 Stabilizer analysis for permutation folding

### D. Random permutation sampler baseline

- D.1 Algorithm description
- D.2 Full comparison tables

---

## Figure/Table 清单

| # | Type | Content | Section |
|---|------|---------|---------|
| Fig 1 | Diagram | FolT-MCMC pipeline schematic | 3 |
| Fig 2 | 2-panel | Banana diagnostic: target + oscillation heatmap | 5.2 |
| Fig 3 | 1-panel | **Headline:** γ̲ vs D (mixture, ReflectionFold) | 5.3 |
| Fig 4 | 2-panel | Oscillation + NLL vs D | 5.3 |
| Fig 5 | 1-panel | γ̲ vs s=|G| (label-switching, PermutationFold) | 5.4 |
| Fig 6 | 2-panel | Structural ID: PSD + trace plot | 6 |
| Tab 1 | Summary | Conditions: when they hold/fail | 4.4 |
| Tab 2 | Results | Banana diagnostic full numbers | 5.2 |
| Tab 3 | Results | Mixture dimension scaling | 5.3 |
| Tab 4 | Results | Label-switching scaling + D vs k! ablation | 5.4 |
| Tab 5 | Results | FolT vs random permutation sampler | 5.5 |
| Tab 6 | Results | Structural ID certified γ | 6.2 |

Total: 6 figures, 6 tables（JRSS-B 通常允许 8-10 figures + tables）

---

## 写作顺序建议

1. **Section 3** (FolT-MCMC framework) — 核心方法，最先写
2. **Section 4** (Theory) — 紧接方法，定理框架
3. **Section 5.2-5.4** (已有实验) — 数据都在手里
4. **Section 2** (Background) — 有了 3-4 后回头写更容易
5. **Section 1** (Introduction) — 最后写，因为需要总览全文
6. **Section 5.5** (baseline comparison) — 需要跑 Phase 5
7. **Section 6** (real application) — 需要从 Paper 4 提取
8. **Section 7** (Discussion) — 最后收尾

---

## 预计时间线

| 阶段 | 内容 | 时间 |
|------|------|------|
| Phase 5 | Random permutation sampler experiment | 1 周 |
| Phase 6 | Structural ID simplified experiment | 2 周 |
| Writing round 1 | Sections 3→4→5→2→1→6→7 | 4-6 周 |
| Internal review | 自查 + 理论 polish | 2 周 |
| Writing round 2 | 根据自查修改 | 2 周 |
| Submission | JRSS-B | ~ 2026 Q4 或 2027 Q1 |
