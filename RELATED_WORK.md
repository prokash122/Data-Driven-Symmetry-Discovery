# Related Work & Novelty Positioning

This document positions the Stage 1 symmetry-discovery pipeline against the
existing literature, identifies what is and is not novel, and provides
drop-in framing for a paper's introduction / related-work section.

> **Summary.** The individual mechanisms we use — SVD null spaces,
> Buckingham-Π reduction, bottleneck autoencoders, Lie-generator
> extraction — are each established. Our contribution is **integrative**:
> we couple dimensional analysis with a discrete symmetry-*type* classifier
> so that the discovered Lie-algebra generators live in dimensionless
> Π-space and are dimensionally consistent by construction, and we unify
> three physically named symmetry classes under a single linear-null-space
> mechanism. The claim to defend is the *coupling and the unification*, not
> the SVD.

---

## 1. Closest prior work, ranked by threat

### 🔴 LieGG: Studying Learned Lie Group Generators (NeurIPS 2022)
*arXiv:2210.04345 · https://github.com/amoskalev/liegg*

The mechanical twin of our generator-extraction step. LieGG extracts
Lie-algebra generators from a **trained** network by forming a matrix from
the network's Jacobians and data and taking its **null space via small
singular values (SVD)**. "Generators = null space of a network-derived
matrix, found by SVD" is exactly our Step 4.

- **Overlap:** the SVD-null-space generator mechanism.
- **Our line:** LieGG runs on generic networks (e.g. image classifiers)
  recovering geometric symmetries, with *no* dimensional analysis, *no*
  Π-space, *no* symmetry-type classifier, and only symmetries that are
  **linear in the raw inputs**. We must cite it and draw the boundary
  explicitly.

### 🔴 The Algebra of Units: From Buckingham's Π Theorem to Latent-Variable Learning (Valorani, 2026)
*arXiv:2606.16737*

Nearly our project's title, and very recent. Uses **SVD on log-rescaled
data** to find the low-dimensional manifold set by the dimensionless groups,
then an integer-exponent search + repeating-variable filter to recover the
Π's.

- **Overlap:** our **Stage 0** (dimensionless-group discovery).
- **Our line:** from its abstract, it *stops at discovering the Π groups*
  — it does not train a target autoencoder, classify symmetry type, or
  extract invariance generators of a prediction target. Our contribution
  begins where theirs ends.
- **Action item:** read this paper in full before finalizing any novelty
  statement; it is the one most likely to force a sharper boundary.

### 🟠 Dimensionally Consistent Learning with Buckingham Pi (Bakarji et al., 2022)
*arXiv:2202.04643 · Nature Computational Science, s43588-022-00355-5*

The canonical "Buckingham-Π + autoencoder" paper. Establishes the
BuckiΠ+NN pairing, so it removes novelty from *that combination alone*.
Does not do symmetry-type classification or generator extraction.

### 🟠 Data-driven discovery of dimensionless numbers and governing laws (Xie et al., Nature Communications 2022)
The PyDimension base we extend (Stage 0). Not a threat — it is our
baseline — but it frames our starting point.

### 🟡 The broader "learn Lie generators from data" family
- **LieGAN** (Yang et al., ICML 2023) — adversarial generator discovery.
- **Symmetry-Informed Governing Equation Discovery** (NeurIPS 2024).
- **LaLiGAN / LieSD / nonlinear-symmetry-from-dynamics** (2024–2025) —
  discover *arbitrary nonlinear* symmetries via learned latent maps.

Individually less close (adversarial or dynamics-based, no dimensional
analysis), but collectively they mean *"discover Lie generators from data"*
is a crowded, active field. Our abstract cannot claim "we discover symmetry
generators" as the headline — that space is taken.

---

## 2. What is NOT novel (do not claim)

- **SVD to find a null space.** Textbook linear algebra
  (`scipy.linalg.null_space`).
- **The row-space / null-space split.** This is the **active vs. inactive
  subspace** decomposition (Constantine, *Active Subspaces*, 2015).
- **Extracting Lie generators from a trained net via SVD null space.**
  Done by LieGG (2022).
- **Buckingham-Π + neural network.** Done by Bakarji et al. (2022).
- **Testing a catalog of named symmetry types.** AI Feynman
  (Udrescu & Tegmark, 2020) already tests translational, scaling,
  additive, and multiplicative symmetries. "Covering multiple symmetry
  types" is therefore **not** novel on its own.

---

## 3. What IS defensible

No prior work combines **all** of:

1. Buckingham-Π reduction to dimensionless Π-coordinates, →
2. intrinsic-dimension discovery with a bottleneck autoencoder, →
3. a discrete symmetry-**type** classifier (translational / scaling /
   rotational, selected by held-out loss), →
4. generator extraction **in Π-space**, so every invariance is
   dimensionally consistent by construction, →
5. demonstrated under one protocol across five distinct physical systems
   (keyhole, concrete, LHC dijets, LPBF porosity, porous-media LBM).

### 3.1 The three-type unification (a real mechanism, not just coverage)

The contribution is not *that* we cover three symmetries but *how* they
unify: each named class reduces to the **same** linear-null-space problem
under a fixed linearizing feature map.

| Symmetry | Feature map φ | Why it linearizes |
|---|---|---|
| Translational | `x` (identity) | already linear: `W(x + εg) = Wx` needs `Wg = 0` |
| Scaling | `log\|x\|` | multiplicative → additive in log-space: `x ⊙ exp(εs)` |
| Rotational | `x²` (component-wise) | quadratic invariants: equal-`\|w\|` coords mix by SO(n) |

Because scaling and rotation are **nonlinear in raw `x`**, methods that
find generators linear in `x` (LieGG and classical linear-generator
methods) cannot represent a multiplicative (scaling) symmetry at all. Our
`φ = log` and `φ = square` maps reach multiplicative and quadratic
symmetries **while keeping the interpretable linear-null-space machinery**
— a lightweight, physically named alternative to general nonlinear
symmetry discovery.

**Honest trade-offs to state, not hide:**

- It is a *fixed catalog of three*, not general nonlinear discovery
  (LaLiGAN et al. are more general but far less interpretable and not
  dimensionally grounded).
- Unlike LieGG/LieGAN, we return a **named, physically meaningful class**
  ("this is a scaling law") plus the generator — interpretability by
  classification, valuable for a physics audience.

---

## 4. Drop-in positioning paragraphs

### 4.1 Related-work framing

> Discovering continuous symmetries from data is now an active area:
> methods such as LieGG [NeurIPS 2022] and LieGAN [ICML 2023] recover
> Lie-algebra generators from trained networks or via adversarial games,
> and a parallel line couples the Buckingham-Π theorem with learning
> [Bakarji et al. 2022; Xie et al. 2022; Valorani 2026] to extract
> dimensionless groups directly from measurements. These efforts, however,
> treat symmetry discovery and dimensional analysis separately:
> generator-extraction methods act in raw feature space and yield
> dimensionally inconsistent directions, while dimensionless-learning
> methods stop at identifying the Π groups and do not characterize the
> symmetry group that leaves a prediction target invariant. We close this
> gap. Given only measurements, our pipeline (i) reduces variables to
> dimensionless Π-coordinates, (ii) identifies the target's intrinsic
> dimension with a bottleneck autoencoder, (iii) classifies the symmetry
> *type* among translational, scaling, and rotational via a single-layer
> encoder, and (iv) extracts the Lie-algebra generators as the null space
> of the learned projection — guaranteeing, by construction, that every
> discovered invariance is dimensionally consistent. We demonstrate the
> same protocol across five distinct physical systems.

### 4.2 The unification claim

> Rather than searching for arbitrary generators, we show that three
> canonical physical symmetries — translational, scaling, and rotational —
> each reduce to a linear-null-space problem under a fixed linearizing
> coordinate map (identity, log, square). A single SVD mechanism then
> recovers the generators for whichever type the data selects, extending
> interpretable linear symmetry discovery to multiplicative and quadratic
> invariances while keeping every result dimensionally consistent.

---

## 5. Verdict

The novelty is **real but narrow and integrative**. It lives in *coupling*
dimensional analysis with symmetry-type classification (so generators come
out dimensionless) and in the *unifying mechanism* (one null-space problem,
three linearizing maps, a physically named output). It does **not** live in
the SVD or in "discovering generators" as such. Frame the paper around the
coupling and the unification, cite LieGG and the Algebra of Units
prominently, and the claim should survive review.

---

### Citation checklist
- LieGG — arXiv:2210.04345 (NeurIPS 2022)
- The Algebra of Units — arXiv:2606.16737 (2026) — **read in full**
- Bakarji et al. — arXiv:2202.04643 / Nat. Comput. Sci. (2022)
- Xie et al. — Nat. Commun. (2022) — PyDimension base
- LieGAN — Yang et al., ICML 2023
- AI Feynman — Udrescu & Tegmark, Sci. Adv. (2020) — symmetry catalog
- Active Subspaces — Constantine (2015) — active/inactive subspace split
- Symmetry-Informed Governing Equation Discovery — NeurIPS 2024

> Note: arXiv PDFs could not be retrieved in the drafting session (proxy
> 403s); the Valorani (2606.16737) characterization rests on its
> abstract/HTML snippet and should be confirmed against the full text.
