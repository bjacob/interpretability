# Survey 01 — The Landscape of Mechanistic Interpretability

> **Series note.** This is the first survey in `literature/survey/`. It is a *map*: where
> the 22 archived papers sit, how they group by approach, and — the part written for you
> specifically — what mathematics they **share** versus where they **fundamentally diverge**.
> It is deliberately high-altitude; per-paper deep dives and a focused critique of
> mathematical blind spots are for later surveys (Part VI sketches the agenda).
>
> **Audience & conventions.** Written for a mathematician + systems engineer who is not an
> ML specialist. ML jargon is offloaded to [MLConcepts.md](MLConcepts.md); the first use of
> a term defined there is *italicised*. Paper links point into `../papers/`. All 22 papers
> were read in full for this survey.
>
> **Notation.** For a linear map $W:A\to B$ we distinguish (per [MLConcepts.md](MLConcepts.md)'s
> convention) the **dual map** $W^{\vee}:B^{*}\to A^{*}$ — canonical, metric-free (the symbol is
> non-standard; some write ${}^{t}W$ or $W'$) — from the **adjoint** $W^{*}:B\to A$, which exists
> only once a metric is chosen. $W^{*}$ always signals an invoked metric; $W^{\vee}$ never does.

---

## 0. The one-paragraph orientation

Mechanistic interpretability (MI) tries to reverse-engineer trained neural networks into
human-understandable algorithms. The corpus you collected is almost entirely the
**Anthropic / "Transformer Circuits" lineage** (16 of 22 papers) plus a few external
landmarks (Redwood's IOI circuit; Nanda et al.'s grokking; Park–Choe–Veitch's linear-
representation geometry). Strikingly, this body of work is **mathematically monolithic at
its foundation and pluralistic at its methods**: nearly every paper stands on the same four
linear-algebraic commitments (Part I), but they split into six methodological camps that
operationalise those commitments very differently (Part II). The deepest intellectual action
— and the place a mathematician can contribute — is at two seams: (a) the **inner-product /
gauge** question that the field mostly leaves implicit (Part III), and (b) the **linear
direction vs. nonlinear manifold** fork that has only just opened (Part IV).

---

## Part I — The shared mathematical substrate

Four commitments recur in essentially every paper. They are not independent; they interlock
into a single worldview. Understanding these four is most of understanding the field.

### I.1 The Linear Representation Hypothesis (LRH): *features are directions*

The foundational modelling assumption: a high-level concept ("this text is in French", "the
subject is a person") is encoded as a **direction** $d_i$ (a unit vector, sometimes a ray or
cone to fix sign) in some activation space, and the activation vector decomposes as a sparse
linear combination
$$ x \;\approx\; b + \sum_i f_i(x)\, d_i, \qquad f_i(x)\ge 0,\ \text{few nonzero.}$$
"Linear" means two things at once: features **superpose additively**, and a feature's
**intensity** is the scalar coefficient $f_i$. This single hypothesis underwrites
[toy-models-superposition](../papers/toy-models-superposition.md),
[towards-monosemanticity](../papers/towards-monosemanticity.md),
[scaling-monosemanticity](../papers/scaling-monosemanticity.md),
[linear-representation-hypothesis-geometry](../papers/linear-representation-hypothesis-geometry.md),
and the entire SAE program. It is stated as an explicit, falsifiable hypothesis — and
Part IV is about the first serious crack in it.

### I.2 Superposition: *$\gg d$ features in $d$ dimensions, paid for in interference*

If features are directions and there are far more candidate features than dimensions, the
model packs them as an **overcomplete, almost-orthogonal frame**. Two facts of different
*character* are in play, and separating them pays off later (Part III):
- *Existence is metric-free.* That you can have more features than dimensions at all is just
  linear dependence. Writing the **embedding map** $W: e_i \mapsto d_i$ (feature-coordinate
  space $\to$ the width-$d$ activation space), [toy-models-superposition](../papers/toy-models-superposition.md)'s
  crispest statement is: superposition is present **iff $W$ is non-injective** ($\ker W\neq 0$;
  the directions $\{d_i\}$ linearly dependent) — a kernel-and-rank claim needing no inner product,
  forced the moment #features $> d$. (The familiar form "$W^{*}W$ singular" says the same through
  the adjoint.)
- *The cost is metric-dependent.* What superposition *pays* is **interference**: the
  off-diagonal of the Gram matrix $G = D^{*}D$ (built from the adjoint, hence metric-laden).
  That this stays small — the frame is
  *almost*-orthogonal — is a claim about an inner product, underwritten by *Johnson–Lindenstrauss*
  (exponentially many near-orthogonal unit vectors exist in $\mathbb{R}^d$) and *compressed
  sensing* (sparse codes are recoverable from low-dimensional linear maps). *Which* inner
  product is left implicit — exactly the question Part III says the field rarely confronts.

Everything downstream is a negotiation with that Gram matrix.

### I.3 The transformer is *almost* linear: residual stream, path expansion, QK/OV

[framework](../papers/framework.md) is the keystone. Its move: the *residual stream* is a
purely **linear communication channel** (blocks add their outputs), so the forward pass can
be **path-expanded** into a sum of end-to-end linear paths, with the *softmax* attention
pattern $A$ as the only nonlinearity. **Freeze $A$** and the model is an exact linear function
of the input tokens. Two low-rank objects fall out of every attention head:
$$ W_{QK} : V\to V^{*} \;\;(\text{a bilinear form — } \textit{where}\text{ to attend; coordinates } W_Q^{\top}W_K), \qquad
   W_{OV} = W_O W_V : V\to V \;\;(\text{an endomorphism — } \textit{what}\text{ to write}).$$
The type signature *is* the where/what split: that $W_{QK}$ lands in the **dual** $V^{*}$ (a
pairing of two stream vectors into a score) is exactly why a transpose appears in its coordinate
form $W_Q^{\top}W_K$, while $W_{OV}$ — a map of stream vectors to stream vectors — needs none; see
[MLConcepts.md](MLConcepts.md) §1.4 and the adjoint/dual convention there. "*Virtual weights*" $W^{(2)}_{\text{in}} W^{(1)}_{\text{out}}$ connect any two blocks through the
stream. This algebra — tensor/Kronecker products, low-rank factorisations, eigenanalysis of
the OV "copying" matrix — is the literal skeleton of Groups 1 and 4 below, and reappears as
the QK bilinear form in Group 5.

### I.4 No privileged basis (a $GL(V)$ gauge symmetry) — except where a nonlinearity breaks it

Because the residual stream is read and written only by linear maps, you can conjugate the
whole network by any invertible $M\in GL(V)$ — replace each read map $W_{\text{in}}$ by
$W_{\text{in}}M^{-1}$ and each write map $W_{\text{out}}$ by $M W_{\text{out}}$ — and get a
**bit-identical function**. The residual stream therefore has **no privileged basis**: its
coordinates are individually meaningless. (Two things *do* survive this gauge: a read map's
**kernel** and a write map's **image** transform covariantly, $\ker\mapsto M\ker$,
$\operatorname{im}\mapsto M\operatorname{im}$, so *containments between them* — and hence
inter-block **virtual weights** $W_{\text{in}}^B W_{\text{out}}^A$ — are invariant, whereas any
complement, norm, or angle is not; see [MLConcepts.md](MLConcepts.md) §1.3.) A *pointwise
nonlinearity* (ReLU on MLP neurons) breaks the symmetry and singles out a **privileged basis** —
which is why neuron-level interpretability is even conceivable.
[mech-interp-variables-bases](../papers/mech-interp-variables-bases.md) names this;
[privileged-bases-residual-stream](../papers/privileged-bases-residual-stream.md) shows the
symmetry is *empirically* broken in the residual stream too (outlier coordinates, kurtosis
$\gg 3$) and fingers the *Adam* optimiser's per-coordinate normalisation as the culprit.

> **Why these four are really one idea.** I.1 says meaning lives in directions; I.4 says the
> ambient space has no canonical directions; I.2 says we cram more directions than dimensions;
> I.3 says the dynamics shuttling those directions around are (frozen-)linear. Put together:
> *the model computes by moving an overcomplete frame of direction-encoded features through a
> gauge-symmetric linear channel punctuated by sparse nonlinearities.* Hold that sentence;
> the whole field is footnotes to it.

---

## Part II — The six approaches

| # | Approach (camp) | Core question | Papers | Signature mathematics |
|---|---|---|---|---|
| 1 | **Algebraic circuit analysis** | Rewrite the forward pass into readable terms | [framework](../papers/framework.md), [induction-heads](../papers/induction-heads.md), [interpretability-in-the-wild-ioi](../papers/interpretability-in-the-wild-ioi.md) | tensor/Kronecker products, low-rank $W_{QK},W_{OV}$, path expansion, OV eigenvalues, causal patching |
| 2 | **Superposition geometry** | How do $\gg d$ features pack into $d$ dims? | [toy-models-superposition](../papers/toy-models-superposition.md), [superposition-memorization-double-descent](../papers/superposition-memorization-double-descent.md), [distributed-representations-composition-superposition](../papers/distributed-representations-composition-superposition.md), [privileged-bases-residual-stream](../papers/privileged-bases-residual-stream.md), [toy-model-of-interference-weights](../papers/toy-model-of-interference-weights.md) | Thomson problem, uniform polytopes / tegum products, JL, compressed sensing, Gram matrix $D^{*}D$, phase transitions, kurtosis test |
| 3 | **Dictionary learning (SAEs)** | *Recover* the feature frame from activations | [towards-monosemanticity](../papers/towards-monosemanticity.md), [scaling-monosemanticity](../papers/scaling-monosemanticity.md), [softmax-linear-units](../papers/softmax-linear-units.md), [sparse-crosscoders-model-diffing](../papers/sparse-crosscoders-model-diffing.md) | overcomplete dictionary learning, $\ell_1$/$\ell_0$ sparse coding, encoder/decoder frames, scaling laws |
| 4 | **Circuit tracing / attribution** | Read off causal computation graphs on a real model | [circuit-tracing-computational-graphs](../papers/circuit-tracing-computational-graphs.md), [on-the-biology-of-a-large-language-model](../papers/on-the-biology-of-a-large-language-model.md), [tracing-attention-feature-interactions](../papers/tracing-attention-feature-interactions.md), [progress-on-attention](../papers/progress-on-attention.md) | local linearization (Jacobian/Taylor), stop-gradients, Neumann series $(I-A)^{-1}$, transcoders, interference-corrected virtual weights |
| 5 | **Representation geometry** | Take the *geometry* (metric, curvature, group structure) seriously | [linear-representation-hypothesis-geometry](../papers/linear-representation-hypothesis-geometry.md), [when-models-manipulate-manifolds-counting](../papers/when-models-manipulate-manifolds-counting.md), [progress-measures-for-grokking](../papers/progress-measures-for-grokking.md) | non-Euclidean inner products, Riesz duality, differential geometry of curves, Fourier analysis / characters of $\mathbb{Z}/n\mathbb{Z}$ |
| 6 | **Epistemics / vision** | What makes an interpretation *correct*? Where is this going? | [interpretability-dreams](../papers/interpretability-dreams.md), [toy-model-of-mechanistic-unfaithfulness](../papers/toy-model-of-mechanistic-unfaithfulness.md), [mech-interp-variables-bases](../papers/mech-interp-variables-bases.md) | faithfulness vs. function-equality, Jacobian matching, the program's axioms |

### Group 1 — Algebraic circuit analysis
*The exact-rewrite tradition.* [framework](../papers/framework.md) treats the (attention-only)
transformer as a tensor-algebraic object and multiplies it out; 0-layer models are bigrams
$W_U W_E$, 1-layer models are skip-trigrams readable from $W_U W_{OV} W_E$ and
$W_E^{\vee} W_{QK} W_E$ (the dual map $W_E^{\vee}$ pulls the QK form back to token space), and 2-layer models gain power through **composition** of heads,
yielding the **induction head**. [induction-heads](../papers/induction-heads.md) elevates that
single circuit to a theory of *in-context learning* and ties its sudden formation to a
**phase change** in training. [interpretability-in-the-wild-ioi](../papers/interpretability-in-the-wild-ioi.md)
(the one Redwood paper) reverse-engineers a 26-head circuit in GPT-2 and contributes the
field's first **formal validation criteria** — *faithfulness, completeness, minimality* — plus
**path patching**. Mathematically this group is the most classical: it is linear algebra and
causal-intervention bookkeeping. Its open wound, stated in [framework](../papers/framework.md)
itself, is that **MLPs resist the rewrite** (the nonlinearity won't freeze) — which is the
cue for Group 3.

### Group 2 — Superposition geometry
*The forward-geometry tradition: given features and sparsity, what packing does training
choose?* [toy-models-superposition](../papers/toy-models-superposition.md) is the gem here:
in controlled ReLU autoencoders it shows superposition is governed by a sparsity-vs-importance
**phase transition**, and that the learned feature directions self-organise into **solutions
of a generalized Thomson problem** — antipodal pairs, pentagons, tetrahedra, square
antiprisms — often decomposing as **tegum (orthogonal) products** of low-dimensional uniform
polytopes. The capacity question gets a compressed-sensing answer: #features is linear in $d$
but modulated by sparsity. The companion notes specialise it:
[superposition-memorization-double-descent](../papers/superposition-memorization-double-descent.md)
(finite data → memorised *datapoints* live in superposition as polygon vertices; double
descent falls out);
[distributed-representations-composition-superposition](../papers/distributed-representations-composition-superposition.md)
(separates **composition** from **superposition** as two ways to spend the same
$\exp(d)$ representational volume — and argues they are *in tension*);
[privileged-bases-residual-stream](../papers/privileged-bases-residual-stream.md) (the
symmetry-breaking diagnostic); and
[toy-model-of-interference-weights](../papers/toy-model-of-interference-weights.md) (the
*weights* between superposed features are themselves forced into superposition, producing
spurious "interference weights" = noise). This group is where the most beautiful, most
classical mathematics already lives.

### Group 3 — Dictionary learning (Sparse Autoencoders)
*The inverse problem to Group 2.* If the model stores an overcomplete sparse code, **recover
the dictionary**. [towards-monosemanticity](../papers/towards-monosemanticity.md) trains
*SAEs* on a one-layer model's MLP and extracts thousands of monosemantic features;
[scaling-monosemanticity](../papers/scaling-monosemanticity.md) scales this to Claude 3 Sonnet
(millions of features, scaling laws for SAE training, a sigmoid law relating a concept's
training-frequency to whether it gets a dedicated feature). [softmax-linear-units](../papers/softmax-linear-units.md)
is the contrarian: instead of *recovering* features post-hoc, *change the architecture*
($\operatorname{SoLU}(x)=x\odot\operatorname{softmax}(x)$) so features align to neurons in the
first place — but a needed LayerNorm "smuggles superposition back", which the authors read as
evidence *for* the superposition hypothesis. [sparse-crosscoders-model-diffing](../papers/sparse-crosscoders-model-diffing.md)
generalises SAEs to **crosscoders** (one dictionary across many layers/models). The whole
group is **dictionary learning / sparse coding** with a learned overcomplete dictionary and an
$\ell_1$ surrogate for $\ell_0$; the recurring anxiety is that there is **no ground-truth
objective** — reconstruction-MSE-plus-sparsity is only a *proxy* for interpretability.

### Group 4 — Circuit tracing / attribution
*Scale Group 1's exact rewrite to real MLP-bearing models by replacing "exact" with
"locally linear".* [circuit-tracing-computational-graphs](../papers/circuit-tracing-computational-graphs.md)
swaps each MLP for a **cross-layer transcoder** (a Group-3-style sparse feature model that
predicts *computation*, not just representation), freezes attention and norm denominators,
adds per-token **error nodes** to match the model exactly on one prompt, and reads off a
**linear attribution graph**. The math is *local linearization*: stop-gradients make each
feature's pre-activation an exact linear sum of upstream contributions (a Taylor expansion,
exact at the prompt, divergent away from it); node influence is the **Neumann series**
$(I-A)^{-1}-I$. [on-the-biology-of-a-large-language-model](../papers/on-the-biology-of-a-large-language-model.md)
applies it to Claude 3.5 Haiku across ~10 case studies (planning, multi-hop reasoning,
multilingual circuits, refusals). [tracing-attention-feature-interactions](../papers/tracing-attention-feature-interactions.md)
and [progress-on-attention](../papers/progress-on-attention.md) attack the part the method
*freezes* — the QK bilinear form $V\to V^{*}$ — by decomposing the attention score into
**feature–feature pairings** $\langle W_{QK}x_k, x_q\rangle$. This group is methodologically dominant today and is
where "*mechanistic faithfulness*" becomes the central worry (Group 6).

### Group 5 — Representation geometry
*The mathematically deepest group, and the one most aligned with your interests.* It treats
the geometry as geometry rather than as bookkeeping.
[linear-representation-hypothesis-geometry](../papers/linear-representation-hypothesis-geometry.md)
formalises "feature = direction" via counterfactual concept pairs, distinguishes the
**embedding** ($\Lambda$, input/context side) and **unembedding** ($\bar\Gamma$, output/word
side) spaces — which are *canonically dual*, $\Lambda\cong\bar\Gamma^{*}$, the softmax pairing
$\langle\lambda,\gamma\rangle$ needing no metric — and — crucially — observes that the Euclidean inner product is **not canonical**
(see Part III), introducing the **causal inner product** — the metric
$g_C=\operatorname{Cov}(\gamma)^{-1}:\bar\Gamma\xrightarrow{\sim}\bar\Gamma^{*}$, so
$\langle\bar\gamma,\bar\gamma'\rangle_C=\langle g_C\,\bar\gamma,\,\bar\gamma'\rangle$ — and a
**Riesz-duality** theorem unifying the two spaces.
[when-models-manipulate-manifolds-counting](../papers/when-models-manipulate-manifolds-counting.md)
finds that a scalar *count* is represented not as a direction but as a **curved 1-D manifold**
(a rippling helix in a ~6-D subspace), and that an attention head **compares** two counts by
a $W_{QK}$ that **rotates/twists one manifold onto another** — differential geometry, not just
linear algebra. [progress-measures-for-grokking](../papers/progress-measures-for-grokking.md)
reverse-engineers modular addition as **Fourier multiplication**: inputs are placed on circles
$(\cos\omega_k a,\sin\omega_k a)$ (the **characters of $\mathbb{Z}/n\mathbb{Z}$**) and addition
becomes angle-addition, read out by trig identities — representation theory of a finite abelian
group. Note the resonance: **circular / periodic structure appears independently** in grokking
(group characters) and in counting (the "ringing"/Gibbs geometry of the count manifold). That
is a genuine shared mathematical phenomenon hiding under two different tasks.

### Group 6 — Epistemics / vision
*What does it mean to be right?* [mech-interp-variables-bases](../papers/mech-interp-variables-bases.md)
sets the program's vocabulary (network ≈ compiled binary; features ≈ variables).
[interpretability-dreams](../papers/interpretability-dreams.md) is the manifesto: once
superposition is solved, "for small enough circuits, statements about their behaviour become
questions of *mathematical reasoning*" — an explicitly **axiomatic** ambition.
[toy-model-of-mechanistic-unfaithfulness](../papers/toy-model-of-mechanistic-unfaithfulness.md)
is the cold shower: a transcoder can be monosemantic, sparse, and low-error **yet implement a
different algorithm** that generalises differently off-distribution. Its proposed fix —
**Jacobian matching** (penalise $\|J_{\text{MLP}}-J_{\text{transcoder}}\|_F^2$) — is mathematically
the most pointed idea in the group, and it quietly concedes that matching input–output
behaviour (Group 3's objective) is *not* matching mechanism.

---

## Part III — What the groups SHARE (the connective tissue)

This is the section you asked to be emphasised. Under the methodological diversity, the
groups are bound by a small number of mathematical objects. Seeing the corpus this way is,
I think, the single most useful reframe.

### III.1 Almost everything is a bilinear pairing — of two kinds: metric-free vs. metric-dependent

Track the inner products through the corpus and nearly every quantity is a **bilinear pairing of
two vectors**. The distinction that matters — and that the field rarely flags — is *which kind*,
because the two behave oppositely under the $GL(V)$ gauge of I.4.

- **Metric-free (gauge-invariant) pairings** are built only from the model's own maps, or from
  the natural contraction of a space with its dual; they need no choice of inner product:
  - the **canonical pairing** $\langle\lambda(x),\gamma(y)\rangle$ the *softmax* consumes (III.2)
    — embedding contracted with unembedding, legitimate because the embedding space is the dual of
    the unembedding space, $\Lambda\cong\bar\Gamma^{*}$ (no metric needed);
  - the **attention score** $\langle W_{QK}x_k,\, x_q\rangle$ — a *learned* bilinear form
    $W_{QK}:V\to V^{*}$, invariant because its query- and key-sides transform contragrediently
    ([tracing-attention-feature-interactions](../papers/tracing-attention-feature-interactions.md)
    expands it into feature–feature terms);
  - a **linear probe** $\langle w,x\rangle$ — a learned covector $w\in V^{*}$ evaluated on a vector.
- **Metric-dependent pairings** pair two vectors *of the same space through the standard inner
  product* — equivalently they apply the **adjoint** ($W^{*}$, not the dual $W^{\vee}$; see
  [MLConcepts.md](MLConcepts.md)'s convention), silently inserting $G=I$ as a metric the
  architecture never supplied:
  - **interference / coherence** — the off-diagonal of the feature Gram matrix $G=D^{*}D$
    (Group 2); capacity, adversarial vulnerability, and the Thomson geometry are read off $G$;
  - **SAE reconstruction** $\|x-\hat x\|_2^2$ and the near-orthogonality $G\approx I$ that makes
    recovery work (Group 3); *feature splitting* is $G$ revealing finer block structure;
  - **cosine similarity** between two feature directions (the default tool for clustering and
    comparing features).

The operational core of Groups 2–4 lives in the *metric-dependent* column — yet I.4 says the
residual stream has no canonical metric. The one paper that treats the metric as something to
**derive** rather than default to $I$ is the LRH paper, whose **causal inner product** answers
"which $G$?" with $\operatorname{Cov}(\gamma)^{-1}$ (a whitening / Mahalanobis form). So:

> *Mathematically, the field is the study of one metric-free pairing (embedding·unembedding, plus
> the learned form $W_{QK}$) alongside a host of metric-dependent Gram computations that quietly
> assume the very Euclidean structure the architecture declares arbitrary.*

### III.2 The gauge symmetry, and the quantity that actually survives it

Combine I.4 (no privileged basis) with the LRH paper's identifiability law and you get a
genuinely important — and, in the corpus, almost entirely **unexploited** — observation.

The softmax output depends only on the canonical pairing $\langle\lambda(x),\gamma(y)\rangle$,
and is invariant under the simultaneous change (LRH paper, eq. 3.1)
$$ \gamma(y)\ \mapsto\ A\,\gamma(y)+\beta, \qquad \lambda(x)\ \mapsto\ (A^{\vee})^{-1}\lambda(x), \qquad A\in GL(\bar\Gamma) $$
(in coordinates $(A^{\vee})^{-1}=A^{-\top}$). This is the **readout instance of the I.4 gauge**
($A$ acting on the final-layer spaces): the unembedding side transforms by $A$, the embedding side
by the inverse **dual map** $(A^{\vee})^{-1}$ — *contragrediently* — so the pairing
$\langle\lambda,\gamma\rangle$ is invariant (the additive $\beta$ merely shifts every logit by the
same $y$-independent amount, which softmax discards — §1.2). That $\lambda$ transforms as a
covector is just the statement $\Lambda\cong\bar\Gamma^{*}$. The general principle, across every activation space:

> **Gauge-invariant means built only from the model's own maps and the natural dual pairing.**
> Quantities assembled from learned weights and the contraction of a space with its dual — the
> embedding–unembedding pairing $\langle\lambda,\gamma\rangle$, an attention score $\langle W_{QK}x_k, x_q\rangle$, a
> probe $\langle w,x\rangle$ — are invariant. Quantities that pair two vectors of the **same** space through
> the **standard** inner product — the cosine of two feature directions, $\|x-\hat x\|_2^2$, the
> Gram off-diagonal — secretly insert the identity as a metric, and so are **gauge-dependent**:
> not canonically meaningful without first *choosing* that metric.

This has a sharp, slightly uncomfortable consequence for Group 3. An SAE minimises
$\|x-\hat x\|_2^2$ in the residual stream and compares decoder directions by Euclidean cosine
similarity — both gauge-dependent quantities — while Group 2/4 insist the same space has *no
privileged basis*. The practical escape hatch is empirical: training + Adam *do* break the
symmetry ([privileged-bases-residual-stream](../papers/privileged-bases-residual-stream.md)),
so the data-induced metric isn't arbitrary. But that is an empirical crutch, not a principled
choice of metric — and exactly one paper (the LRH paper) treats the metric as something to be
*derived* rather than assumed. **This is the central seam of the whole corpus** and I flag it
as the leading candidate for mathematical contribution (Part VI).

The constructive flip side is worth stating, because the gauge problem is not a dead end: there
*is* a gauge-invariant way to ask whether two components interact — not the cosine of their
directions, but whether one's **image lands in the other's kernel**, equivalently whether their
**virtual weight** $W_{\text{in}}^B W_{\text{out}}^A$ vanishes (I.4, [MLConcepts.md](MLConcepts.md) §1.3).
Re-expressing interference and feature interaction in those metric-free terms — rather than
through the $G=I$ Gram matrix — is one concrete shape the contribution in Part VI(1) could take.

### III.3 The same low-rank attention algebra, reused five times

$W_{QK}:V\to V^{*}$ (the bilinear form; coordinates $W_Q^\top W_K$) and $W_{OV}=W_O W_V:V\to V$
(the endomorphism) are introduced in [framework](../papers/framework.md) and then *never leave*: induction-head prefix-matching is a positive-eigenvalue OV plus a
K-composition QK; the IOI circuit is classified entirely in these terms; circuit tracing
freezes the QK and linearises the OV; the attention papers decompose QK into features; the
manifold paper finds QK implementing a rotation. **If you learn one piece of transformer
algebra, learn the QK/OV factorisation** — it is the corpus's universal coordinate system.

### III.4 Three regularizers, one Lagrangian shape

Groups 2–4 and 6 all minimise *reconstruction + sparsity*:
$\ \mathbb{E}\|x-\hat x\|^2 + \lambda\,\rho(f)$, with $\rho$ an $\ell_1$ surrogate for $\ell_0$.
Toy models, SAEs, transcoders, crosscoders, the interference-weight model, the unfaithfulness
model — same objective, different $x$, different decoder constraints. The shared pathologies
(shrinkage from $\ell_1$, dead features, the $\ell_0$/$\ell_1$ gap exploited by "gradient
masking") are therefore *shared* across the whole tooling stack, not quirks of one method.

### III.5 Compressed sensing as the two-way bridge

Group 2 uses compressed sensing to *upper-bound* how much superposition is possible (a forward
statement about $W$); Group 3 *is* the recovery algorithm compressed sensing promises (an
inverse statement about recovering $f$ from $x=Wf$). Same theorem (RIP / incoherence / sparse
recovery), read in both directions. This is the cleanest example of two camps sharing one
piece of mathematics.

---

## Part IV — Where the math FUNDAMENTALLY diverges

Genuine forks, not just different tools. These are the fault lines worth your attention.

### IV.1 Feature-as-direction (linear) vs. feature-as-manifold (nonlinear) — the big one

Groups 1–4 assume a feature is a **1-D ray**: meaning is a direction, intensity is a scalar.
[when-models-manipulate-manifolds-counting](../papers/when-models-manipulate-manifolds-counting.md)
breaks this for *continuous* quantities: a count is a **curved 1-dimensional manifold** with
deliberately high curvature ("ringing"), embedded in ~6 dimensions, and the model exploits the
curvature (a ray could only be scaled/compared by a monotone product, whereas a curve admits a
genuine **rotation** that implements thresholded comparison). This is a *category* difference:
linear subspace vs. nonlinear submanifold; linear algebra vs. **differential geometry**. The
question it raises is pointed and, as far as the corpus shows, open: **is the Linear
Representation Hypothesis simply false for continuous/ordered variables, and right only for
discrete categorical ones?** If so, the field needs a geometry of feature *manifolds* (intrinsic
dimension, curvature, how attention acts as a near-isometry between them) that currently does
not exist in worked-out form. The LRH paper and the manifold paper are, in a precise sense,
**mathematically incompatible accounts of what a feature is** — and nobody has reconciled them.
(Note the entanglement with III: the manifold's "curvature" and "ringing" are themselves read
through the ambient inner product — cosine-similarity matrices of activations — so a properly
metric-free account of feature *manifolds* is doubly open.)

### IV.2 Modelling representation vs. modelling computation

SAEs (Group 3) model the **representation** — a basis for activation *space*. Transcoders and
attribution graphs (Group 4) model the **computation** — the map between activations. The
[unfaithfulness](../papers/toy-model-of-mechanistic-unfaithfulness.md) paper shows these are
**not interchangeable**: an SAE "can't really mislead you" (it just describes a vector), but a
transcoder that reproduces the input–output map can still implement the *wrong algorithm*. The
mathematical content of the distinction: representation-modelling constrains a single space;
computation-modelling must match a **Jacobian** (a linear map between spaces), and matching a
function is strictly weaker than matching its derivative everywhere. This is a real divergence
in what "correct" even means, and it cuts the toolkit in half.

### IV.3 Exact algebra vs. local linearization vs. learned approximation

Three incompatible notions of "linear" coexist (see [MLConcepts.md](MLConcepts.md) §1.4 and §3):
1. **Exact, by freezing the nonlinearity** ([framework](../papers/framework.md)): attention
   frozen ⇒ the forward pass is *honestly* linear in tokens. No approximation.
2. **Local linearization** ([circuit-tracing](../papers/circuit-tracing-computational-graphs.md),
   attribution patching): a Taylor/Jacobian expansion, *exact at one prompt, wrong away from it.*
3. **Learned sparse approximation** (SAEs/transcoders): a trained model that is neither exact
   nor a derivative, only a low-error fit — and possibly mechanistically unfaithful.
These are routinely all called "linear", but they have different error semantics, and conflating
them is where overclaiming creeps in. A mathematician will want to keep them strictly separate.

### IV.4 Discrete-algebraic vs. continuous-geometric task structure

[progress-measures-for-grokking](../papers/progress-measures-for-grokking.md) finds **finite
group representation theory** ($\mathbb{Z}/n\mathbb{Z}$ characters); the manifold paper finds
**continuous differential geometry**; the superposition papers find **discrete polytopes**
(Thomson). These are three different branches of mathematics describing "the geometry of
representations", and the field has **not** unified them. The tantalising hint (III, and the
grokking↔counting circle resonance) is that **harmonic analysis / periodicity** might be the
common generalisation — Fourier on a group, Fourier features on a manifold, and the eigen-
structure of a circulant Gram matrix are not unrelated — but this is conjecture, not established.

---

## Part V — A few honest observations about the corpus

- **Provenance bias.** 16/22 papers are one lab's (Anthropic) thread; they share authors,
  assumptions, and a house style (many are explicitly "preliminary lab-meeting notes"). The
  shared substrate of Part I is partly a sociological fact, not only a mathematical necessity.
  The three external papers (IOI, grokking, LRH-geometry) are also the three that contribute
  the most *formal* machinery (validation metrics; Fourier reverse-engineering; the metric
  question) — worth noting.
- **Rigour gradient.** Formality decreases as models scale: toy models get closed-form loss
  analysis and named polytopes; frontier-model papers get qualitative case studies with
  ~25% success rates and heavy hedging. The mathematically tractable results are mostly in
  Group 2 (toy models); the high-stakes claims are mostly in Group 4 (frontier circuits),
  where faithfulness is weakest.
- **Proxy objectives everywhere.** No one has a ground-truth definition of "feature" or a
  principled SAE objective; MSE+$\ell_1$ is admitted to be a stand-in. Several "definitions"
  (copying matrix via positive eigenvalues; interference weights; faithfulness) are explicitly
  heuristic and acknowledged as not-yet-formalisable.

---

## Part VI — Agenda for later surveys (where the math might be)

Signposts, not conclusions. Each is a candidate "mathematical blind spot → contribution"
to develop in a focused follow-up.

1. **A gauge theory of representations (III.2).** Make the $GL(V)$ symmetry explicit;
   characterise the invariants (the embedding–unembedding pairing, learned forms like $W_{QK}$,
   and image-in-kernel / virtual-weight relations); re-express superposition/SAE/probing results
   in gauge-invariant form; ask what an SAE *should* minimise if not the gauge-dependent $\ell_2$
   loss. The LRH paper's $\operatorname{Cov}(\gamma)^{-1}$ is one answer for the output space;
   the residual-stream interior is wide open.
2. **A geometry of feature manifolds (IV.1).** Reconcile direction-features with manifold-
   features. What is the right invariant description of a count/colour/date manifold (intrinsic
   dimension, curvature, the action of $W_{QK}$ as a near-isometry)? Is LRH a categorical-
   variable special case of a manifold theory?
3. **The interference Gram matrix as the central object (III.1).** A unified treatment of
   $G=D^{*}D$ — spectrum, block structure (feature splitting), the relation between its
   off-diagonal (interference) and downstream "interference weights", and bounds linking
   sparsity, dimension, and the operator norm of $G-I$.
4. **Error semantics of the three "linearities" (IV.3).** Make precise when local
   linearization / learned approximation incurs how much error, and connect Jacobian-matching
   (mechanistic faithfulness) to a quantitative bound.
5. **Harmonic analysis as a possible unifier (IV.4).** Is there a single framework
   (Fourier on groups / spectral geometry / circulant structure) underlying the polytope,
   character, and manifold geometries? Likely the most speculative, possibly the most fruitful.

---

### Appendix — paper → group quick index

- **framework** — G1 (substrate for all)
- **induction-heads** — G1
- **interpretability-in-the-wild-ioi** — G1 (+ validation metrics)
- **toy-models-superposition** — G2 (substrate for G3)
- **superposition-memorization-double-descent** — G2
- **distributed-representations-composition-superposition** — G2
- **privileged-bases-residual-stream** — G2 (substrate I.4)
- **toy-model-of-interference-weights** — G2 (bridges to G4)
- **towards-monosemanticity** — G3
- **scaling-monosemanticity** — G3
- **softmax-linear-units** — G3 (architectural variant)
- **sparse-crosscoders-model-diffing** — G3 (bridges to G4)
- **circuit-tracing-computational-graphs** — G4 (method)
- **on-the-biology-of-a-large-language-model** — G4 (application)
- **tracing-attention-feature-interactions** — G4/G1 (QK)
- **progress-on-attention** — G4/G1 (QK)
- **linear-representation-hypothesis-geometry** — G5 (the metric question)
- **when-models-manipulate-manifolds-counting** — G5 (the manifold fork)
- **progress-measures-for-grokking** — G5 (group/Fourier)
- **interpretability-dreams** — G6
- **mech-interp-variables-bases** — G6 (substrate I.4)
- **toy-model-of-mechanistic-unfaithfulness** — G6 (faithfulness)
