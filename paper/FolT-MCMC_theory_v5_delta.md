# FolT-MCMC 理论 v5 — v4 的四处修正

本文只记录 v4 → v5 的 delta。v4 中未提及的部分保持不变。

---

## Fix 1: Quotient ball mass bound — stabilizer correction

**v4 写法（过强）：**
> π_F(B_G(z,ε) ∩ D) ≥ π_min^F · V_d · ε^d

**问题：** 对 fold boundary 上有非平凡 stabilizer 的点 z（即存在 g≠e 使得 g·z = z），
quotient ball B_G(z,ε) 在原空间中不对应 s 份完整的 ball 碎片，而是更少。
极端情况：z 是所有 g 的不动点（如排列折叠的 "all-equal" 点），
B_G(z,ε) 只对应 1 份 ball，此时 mass = π_F(z) · V_d · ε^d / s 而非 · 1。

**v5 修正：**

设 Stab(z) = {g ∈ G : g·z = z} 是 z 的 stabilizer，|Stab(z)| =: h(z)。
Orbit-stabilizer theorem: |Orb(z)| = s/h(z)。

B_G(z,ε) 在原空间中对应 s/h(z) 份 ball 碎片（每份在不同的 g·D 中），
加上 z 本身所在 D 中的碎片。总体积：

Vol(B_G(z,ε) in R^d) = (s/h(z)) · Vol(B(z,ε) ∩ D ∩ near-boundary)

对 generic z ∈ int(D)：h(z) = 1，Vol = s · (partial ball) ≥ V_d · ε^d（完整球）。
对 z ∈ ∂D with h(z) > 1：Vol = (s/h(z)) · (V_d·ε^d/h(z)) ≈ s/h(z)² · V_d·ε^d。

**但 π_F-mass 的 lower bound 是：**

π_F(B_G(z,ε) ∩ D) = s · π(B_G(z,ε) ∩ D)
                    = s · ∫_{B_G(z,ε) ∩ D} π(θ) dθ

由于 B_G(z,ε) ∩ D ⊇ B(z,ε) ∩ D：

π_F(B_G(z,ε) ∩ D) ≥ s · π_min^D · Vol(B(z,ε) ∩ D)

对 z ∈ int(D)（远离 ∂D）：Vol(B(z,ε) ∩ D) = V_d · ε^d，
所以 π_F(B_G(z,ε) ∩ D) ≥ s · π_min^D · V_d · ε^d = π_min^F · V_d · ε^d。 ✓

对 z 靠近 ∂D：Vol(B(z,ε) ∩ D) ≥ V_d · ε^d / h_max，
其中 h_max := max_{z ∈ K_α^F} |Stab(z)|。

**保守 lower bound：**

$$\pi_F(B_G(z,\varepsilon) \cap D) \geq \frac{s}{h_{\max}} \cdot \pi_{\min}^D \cdot V_d \cdot \varepsilon^d = \frac{\pi_{\min}^F}{h_{\max}} \cdot V_d \cdot \varepsilon^d$$

**与原空间比较：** 原空间用 π_min · ω_d(ε) · ε^d（LCNF 的 curvature-corrected half-ball）。

改善条件：π_min^F / h_max ≥ π_min，即 s/h_max ≥ 1。

| 对称群 | s | h_max | s/h_max | 改善? |
|--------|---|-------|---------|-------|
| Z₂ 反射 | 2 | 2（boundary 上 Stab = Z₂） | 1 | 不差不优 |
| S₃ 排列 (m=3) | 6 | 6（all-equal 不动点） | 1 | 不差不优 |
| S₄ 排列 (m=4) | 24 | 24 | 1 | 不差不优 |

**但 h_max 的不动点集通常不在 K_α^F 中！**
对 well-separated mixture，all-equal 点 (μ,μ,...,μ) 远离任何 mode 中心，
不在 HPD set 中。K_α^F 中 z 靠近 mode 中心，stabilizer 通常 = {e}。

因此 **effective** h_max on K_α^F 通常 = 1，此时：

$$\pi_F(B_G(z,\varepsilon) \cap D) \geq \pi_{\min}^F \cdot V_d \cdot \varepsilon^d \quad \text{for } z \in K_\alpha^F \text{ with } |Stab(z)|=1$$

**v5 主定理中的写法：**

引入条件 **(G) Generic stabilizer on HPD set:**

|Stab(z)| = 1 for all z ∈ K_α^F

（等价于：K_α^F ⊂ int(D)，或 K_α^F 不含 fold boundary 上的不动点。
对 well-separated modes 这自动成立。）

在 (G) 下，quotient ball mass bound 是完整的：
π_F(B_G(z,ε) ∩ D) ≥ π_min^F · V_d · ε^d。

不满足 (G) 时，用 h_max-corrected bound：
π_F(B_G(z,ε) ∩ D) ≥ (π_min^F / h_max) · V_d · ε^d。

---

## Fix 2: Covering number — Euclidean dominates quotient

**v4 写法（不严密）：**
> N_G(K_α^F, ε) ≤ (diam_G(K_α^F)/ε + 1)^d

**问题：** 一般 metric space 中 covering number 不能只用 diameter 和 ε 来 bound，
需要 doubling constant 或额外结构。

**v5 修正：**

由 d_G(x,y) ≤ ||x−y||（(P1)），任何 Euclidean ε-cover 自动是 quotient ε-cover。
因此：

$$N_G(K_\alpha^F, \varepsilon) \leq N_E(K_\alpha^F, \varepsilon) \leq \left(\frac{2\,\mathrm{diam}_E(K_\alpha^F)}{\varepsilon} + 1\right)^d$$

后一个不等式是标准 Euclidean covering number bound（K_α^F ⊂ R^d 紧集）。

---

## Fix 3: MH 等价性声明修正

**v4 写法（过强）：**
> 这和在原空间做 MH 然后 project 是一致的

**v5 修正：**

Projecting samples from q_T onto the fundamental domain D induces 
the quotient proposal density q_F(z) = Σ_{g∈G} q_T(g·z).
Independence MH on D with target π_F and proposal q_F is a valid 
Markov chain with stationary distribution π_F.

It is **not** identical to performing MH in the original space and 
then projecting, unless q_T is itself G-symmetric 
(i.e., q_T(g·θ) = q_T(θ) ∀g). In general,

π_F(z)/q_F(z) = s·π(z) / Σ_g q_T(g·z) ≠ π(z)/q_T(z).

When q_T is concentrated on a single mode (as typical for finite-capacity 
SN-RealNVP), Σ_{g≠e} q_T(g·z) ≈ 0 for z deep in D, and the two ratios 
approximately agree. But this is an empirical observation, not a 
theoretical identity.

---

## Fix 4: Lemma 3 — 增加 main-mode fit 和 folded-proposal residual 条件

**v4 Lemma 3（不够严密）：**
只有 off-mode underfit condition (M')，缺少 main-mode fit 条件。

**v5 Lemma 3'（完整版）：**

**Conditions:**

**(M') Cross-mode proposal deficiency** (同 v4)：
∃ Δ_j > 0, j≠1, s.t. sup_{K_α^{(j)}} q_T ≤ exp(−Δ_j) · inf_{K_α^{(j)}} π

**(W) Main-mode fit quality:**
sup_{θ ∈ K_α^{(1)}} |log q_T(θ) − log π(θ) − c₁| ≤ r₁
for some constant c₁ ∈ R and residual r₁ ≥ 0.

**(R) Folded-proposal residual:**
sup_{z ∈ K_α ∩ D} |log q_F(z) − log q_T(z)| ≤ r_F
（quantifies Σ_{g≠e} q_T(gz) 的贡献；对 well-separated modes, r_F ≈ 0）

**Lemma 3' (Oscillation gap with error terms).**
Under (M'), (W), (R):

$$\hat{o}_n \geq \hat{o}_n^F + \min_j \Delta_j - 2r_1 - r_F - O\!\left(\sqrt{\frac{\log(2/\delta)}{n}}\right)$$

**Proof sketch.**

在原空间 K_α 上：
- Mode 1 内：h(θ) = log π(θ) − log q_T(θ) ∈ [−c₁ − r₁, −c₁ + r₁]
  → osc within mode 1 ≤ 2r₁
- Mode j 内：log q_T(θ) ≤ log π(θ) − Δ_j
  → h(θ) = log π(θ) − log q_T(θ) ≥ Δ_j
  → max over mode j ≥ Δ_j
- 因此 osc_{K_α}(h) ≥ (Δ_j) − (−c₁ + r₁) − (max within mode 1)
  ≥ min_j Δ_j − 2r₁ + osc_{mode 1}(h)

在折叠空间 K_α ∩ D 上：
h_F(z) = log(sπ(z)) − log q_F(z)
       = log(sπ(z)) − log q_T(z) − log(1 + Σ_{g≠e} q_T(gz)/q_T(z))

最后一项 ∈ [0, r_F]（by condition (R)）。

因此 osc_{K_α ∩ D}(h_F) ≤ osc_{mode 1}(h) + r_F + log s − log s = osc_{mode 1}(h) + r_F。

合并：
ô_n ≥ osc_{K_α}(h) − sampling error
    ≥ min_j Δ_j − 2r₁ + osc_{mode 1}(h) − O(√...)
ô_n^F ≤ osc_{mode 1}(h) + r_F + O(√...)

相减：ô_n − ô_n^F ≥ min_j Δ_j − 2r₁ − r_F − O(√...). □

**Remark.** 对 SN-RealNVP with OscReg：
- r₁ ≈ ô_n^F / 2（主 mode 的 oscillation 直接反映 fit quality）
- r_F ≈ 0（well-separated modes，Σ_{g≠e} 指数小）
- Δ_j 由 mode 间距离和 flow capacity 决定

因此 effective gap ≈ min_j Δ_j − ô_n^F，
which is large when the flow fails to fit off-modes (Δ_j ≫ 0) 
but fits the main mode well (ô_n^F small).

---

## Fix 5 (from v4 review item 6): h_F Lipschitz w.r.t. d_G

**需要补的 lemma：**

**Lemma 4 (Quotient Lipschitz property).**
设 h_F(z) = log π_F(z) − log q_F(z) 在 R^d 上 C^1 with
M := sup_{θ ∈ K_α} ||∇h_F(θ)|| < ∞（where h_F extended to R^d 
via h_F(θ) = log(sπ(θ)) − log(Σ_g q_T(gθ))）。

则对 x, y ∈ K_α^F：

|h_F(x) − h_F(y)| ≤ M · d_G(x, y)

**Proof.**
h_F 的 extension 到 R^d 是 G-invariant（因为 π 是 G-invariant，
q_F(z) = Σ_g q_T(gz) 也是 G-invariant by construction）。

对任意 g ∈ G：
|h_F(x) − h_F(y)| = |h_F(x) − h_F(g·y)|  （G-invariance）
                   ≤ M · ||x − g·y||        （mean value theorem）

对 g 取 min：
|h_F(x) − h_F(y)| ≤ M · min_g ||x − g·y|| = M · d_G(x, y). □

**这使得 LCNF 的 interpolation step（从 covering 到 oscillation bound）
在 quotient metric 下合法：**

osc_{K_α^F}(h_F) ≤ ô_n^F + 2 M_F · ε*_F

其中 ε*_F 是 quotient metric covering 的 radius，
M_F = sup_{K_α^F} ||∇h_F|| 是 Euclidean gradient norm
（quotient Lipschitz constant = Euclidean Lipschitz constant，by Lemma 4）。

---

## v5 完整定理框架（最终版）

**Theorem (FolT-MCMC Certified Improvement).**

*Assumptions:*
(F1) π G-invariant, |G|=s, fundamental domain D
(F2) K_α^F = K_α ∩ D satisfies LCNF Assumptions 1'-3' (Proposition 1', Lemmas 1'-2)
(F3) (G) |Stab(z)| = 1 for all z ∈ K_α^F (generic stabilizer; automatic for well-separated modes)
(F4) T trained on π or π_F; q_F(z) = Σ_g q_T(gz); h_F = log π_F − log q_F

*Part I (Folded certificate):*
With prob ≥ 1−δ_F:
osc_{K_α^F}(h_F) ≤ ô_n^F + 2 M_F ε*_F =: C_F
γ̲_F ≥ 2/(1+exp(C_F))

*Part II (Covering improvement):*
ε*_F ≤ ε*_U  (by monotonicity: D_F ≤ D_U, π_min^F ≥ s·π_min)

*Part III (Certified gap improvement):*
Additionally assume (M'), (W), (R) from Lemma 3'.
If min_j Δ_j > 2r₁ + r_F + 2(M_F ε*_F − M_U ε*_U)⁺ + O(√(log/n)):
then with prob ≥ 1−δ_U−δ_F:
C_F < C_U, hence γ̲_F/γ̲_U > 1.

---

## 严密度评估

| Version | 评分 | 主要问题 |
|---------|------|----------|
| v2 | ~70% | 5处过强声称 |
| v3 | ~82% | (A2) 条件化，但 HPD/covering/proposal 未闭合 |
| v4 | ~88% | quotient metric 方向对但 stabilizer/covering number/MH等价 有洞 |
| v5 | ~93% | 四处 surgical fix + Lipschitz lemma，主要剩余：formal proof 需要写完整 |

**v5 剩余的 ~7% gap：**
- Quotient covering lemma 的 formal proof 需要完整写出（不能只说"逐字适用"）
- Approximate symmetry extension（future work，不影响当前定理）
- Real application（实验层面，不影响理论严密度）
