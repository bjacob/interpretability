# A Toy Model of Interference Weights

> Source: https://transformer-circuits.pub/2025/interference-weights/index.html
> literature/README.md line 1
> Format: markdown (converted from HTML); images in `images/toy-model-of-interference-weights/`

---

This note explores the phenomenon of "interference weights" and "weight superposition", an idea that we've discussed briefly in previous papers and updates. We've come to believe [they are a central issue](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#limitations-local-v-global) if one wants to move from attribution graphs which describe why the model behaves the way it does for a specific example, to global circuit analysis where one can reason about the model more broadly. (In fact, avoiding interference weights was our primary motivation for studying attribution graphs.)

We study interference weights in the context of toy models and preliminarily find:

Takeaway 1: Interference weights can be demonstrated in toy models. We just need to slightly modify our interpretation of the setup from the original [Toy Models](https://transformer-circuits.pub/2022/toy_model/index.html)[paper](https://transformer-circuits.pub/2022/toy_model/index.html). The resulting interference weights exhibit distinctive phenomenology we saw in [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), suggesting they also occur in real models. For many purposes we probably want to filter interference weights out in our analysis.

Takeaway 2: There are a number of plausible definitions of interference weights. Some are principled, while others are less principled heuristics which are more practical. We can compare these in toy models. Using some clever tricks, it should also be tractable to apply the expensive, principled definitions to small numbers of weights in real models, which might be useful for baseline comparisons of heuristics.

Takeaway 3: A lot more toy model work could be done to better understand interference weights.

This note is very preliminary. We're sharing because it might be of interest to researchers working actively in this space, and we believe there's value in sharing thoughts earlier. We'd ask you to treat these results like those of a colleague sharing some thoughts or preliminary experiments for a few minutes at a lab meeting, rather than a mature paper. It is not intended for a broad audience. All claims should be taken with significant caveats and low confidence throughout.

### [Introduction](#introduction)

When we think of a model in superposition, we tend to think of how the features are arranged in superposition. However, there's a second aspect to superposition which is easy to neglect: [weight superposition](https://transformer-circuits.pub/2023/may-update/index.html#weight-superposition).

If two layers have features in superposition, the weights between the features are forced into superposition as well:

![](images/toy-model-of-interference-weights/fig-001.png)

This is a big problem for circuit analysis because it causes "interference weights". Even if we can uncover the correct features, when we "lift" the model weights to connect those features, many of those weights will correspond to feature interference.

![](images/toy-model-of-interference-weights/fig-002.png)

### [Should we care about interference weights?](#should-we-care)

These interference weights are "real" in the sense that they do genuinely describe connections between features in the model we observe. In fact, there's a [hypothesis](https://transformer-circuits.pub/2022/toy_model/index.html#adversarial) that they're one of the causes of adversarial examples!

However, they're essentially noise and as a result don't make sense. The model doesn't "want to have them". They make the loss worse, or at least don't help it. If we finetune the lifted model (impractical) they go away.This is actually much more subtle than it sounds, and is a bit simplified as written.  In practice, there are at least three issues that prevent this statement from being strictly true. (1) Features are often not perfectly monosemantic, and so the model loss has some tiny preference for interference weights, requiring a small penalty. (2) If one naively optimizes more than one matrix in the "upstairs feature model", gradient descent may try to change those features – for example, it might introduce new superposition and polysemanticity, taking advantage of the added capacity. (3) Even setting aside the new polysemanticity issues of the previous point, the model may just learn new weights that it didn't try to represent when in superposition, or undo shrinkage (see scatter plot of virtual weights vs ideal weights later); this is expected under Defn (1) but counter intuitive. See [Appendix 4](#appendix-4) for related discussion. As a result of all of these issues, in practice one might do something like learn a mask on the virtual weights with a small penalty, along the lines of [Drori, 2025](https://www.lesswrong.com/posts/NAQpcNz9WGSJ8WH2A/sparsely-connected-cross-layer-transcoders).

A natural question, then, is whether we should care about them for safety. It seems like this may depend on our goals:

* Robustness – They Matter! We can think of robustness as being the set of safety concerns arising from an adversarial / worst case environment (including user inputs). In this case, the interference weights exist and are an object the adversary can exploit. (Again, see the idea that they might literally cause adversarial examples!)
* Alignment - They Likely Don't Matter. We can think of alignment as being the safety concerns arising from an adversarial optimization or learning process. If we define interference weights as being those which harm, or at least don't help, the optimization objective, they in some sense definitionally don't matter for alignment concerns.

This split suggests that we want to find a way to factor apart the "interference weights" and "real weights", so that we can only consider the interference weights when relevant. Since alignment is interpretability's priority, we care most about the "real weight" analysis, although of course we also want to keep the interference weights in mind.

Note that this claim – that we can ignore interference weights for alignment – warrants serious scrutiny. We're not fully convinced of it at this point! But it does seem quite likely to us, or at least likely that it's directionally true (perhaps interference weights matter much less than real ones for alignment).

It seems possible that whether interference weights can be ignored may be the pivotal question in whether global mechanistic analysis is possible.

### [Revisiting Toy Models](#revisiting-tms)

[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) introduced a simple toy model for studying feature superposition:

h=Wx

x' = \text{ReLU}(W^Th+b)

Or:

x'=\text{ReLU}(W^TWh+b)

This toy model actually has two different interpretations. We can think of it as studying the geometry of how features are encoded in superposition (this is how Toy Models frames it), but we can also see it as an extremely simple case of one-layer identity circuits being put in superposition:

![](images/toy-model-of-interference-weights/fig-003.png)

By revisiting the toy model with this second interpretation, we can do some very basic empirical exploration of weight superposition.

When we train these toy models, we'll be interested in U=W^TW – the observed, "downstairs", [virtual weights](https://transformer-circuits.pub/2021/framework/index.html#virtual-weights). Even though in actuality, the toy model has a smaller set of weights which project down to a lower dimensional space and back up, these are the effective weights between the features.

### [Reproducing Phenomenology from Towards Monosemanticity](#basic-phenomenology-reproduction)

In [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), we observed a number of phenomena that seemed suggestive of interference weights. They were in fact the origin of much of our thinking on this topic! The goal of this section will be to find a simple toy model that matches the originally motivating phenomena, and compare them side-by-side.

But before we dive in, it's worth understanding why Towards Monosemanticity's experiments would have interference weights in the first place! At its core, the paper studied superposition in the MLP output of a one-layer transformer language model, using a sparse autoencoder to extract features. The key idea to understand is the notion of "logit weights", which connected the discovered features to the logits. Our claim will be that these logit weights should have interference weights.

Note that in such a setup, the features have a linear effect on the logits (modulo a rescaling by the final layer norm). We can get the virtual weights connecting the features and logits by tracing the path from the features, to the MLP output, to the residual stream, and then back up to the logits.

![](images/toy-model-of-interference-weights/fig-004.png)

However, along this path, the features are forced into superposition, first in the MLP output, then more densely in the residual steam. As a result, when we expand these weights, we expect to see interference weights.

And indeed, Towards Monosemanticity did see many indications of such interference weights! We'll explore this very shortly. But before we do, we'll briefly introduce a toy model, and give a formal definition for interference weights. This will allow us to examine the real world phenomena from Towards Monosemanticity side-by-side with examples from toy models, where we can definitively say the analogous behavior comes from interference weights.

#### [Toy Model Setup](#basic-phenomenology-reproduction-setup)

Our goal is to reproduce the same basic phenomenology from Towards Monosemanticity, concretely demonstrating how weight superposition could explain those observations. We'll try to start with the simplest model that can produce roughly correct results to illustrate these ideas. In a later section, we'll produce a more complex model that matches things better.

We consider a standard toy model with `n_features=100`, `n_residual=20`, and `feature_density=0.02`. It will also be undertrained (this reproduces the phenomenology of Towards Monosemanticity better; we'll explore fully converged examples later).

![](images/toy-model-of-interference-weights/fig-005.png)

#### [Interference Weights and Decomposition](#basic-phenomenology-reproduction-decomposition)

Because of superposition, the actual weights we observe are "noisy". Informally, this noise is the interference weights.

One way we could try to formalize this with the following definitions:

* Ideal Weights. The ideal weights, U^\*, are the weights we would get if we finetuned the feature-feature weight matrix in the lifted space with respect to the training loss, with a small weight regularization term to get rid of weights where the loss is indifferent.Note that this notion of ideal weights is pretty awkward to operationalize outside of toy models, and that the corresponding notion of interference weights may not be what we want. We discuss the various definitions one could have, and some of their pros and cons, in [Appendix 4](#appendix-4).
* Interference Weights (defn 1). The interference weights are U-U^\*, the difference between the observed virtual weights and ideal weights.

We can then think of the observed weights as decomposing into ideal weight and interference weights:

![](images/toy-model-of-interference-weights/fig-006.png)

Another definition is to think about the loss contribution of each weight, \Delta L(U\_{ij}). That is, the difference between the toy model loss, and its loss if we ablate a given weight.

* Interference Weights (defn 2). The interference weights are the subset of the virtual weights U which don't improve the loss. That is, those which have \Delta L(U\_{ij}) < \epsilon.

This offers an alternative decomposition:

![](images/toy-model-of-interference-weights/fig-007.png)

For now, we'll focus our investigation on the loss contribution of each weight, \Delta L(U\_{ij}). Later, we'll return to these two definitions, along with a few others which are easier to operationalize.

#### [Weight Histogram](#basic-phenomenology-reproduction-histogram)

We can now return to our earlier goal of showing how our toy model can recover some of the interesting phenomenology of Towards Monosemanticity.

Firstly, let's consider a histogram of our weights, U. We'll color the histogram by \Delta L(U\_{ij}) to allow us to distinguish interference weights from real weights.

![](images/toy-model-of-interference-weights/fig-008.png)

This plot might seem familiar – it's qualitatively similar to the logit weight plots from Towards Monosemanticity, such as this one for the [Hebrew feature](https://transformer-circuits.pub/2023/monosemantic-features/index.html#feature-hebrew):Note that there's a difference in the color scheme: the coloring above estimates a weight's average effect on the loss while the coloring below represents a token's connection with our interpretation of the feature. Another difference is the toy model histogram shows weights from all features while the graph from Towards Monosemanticity shows weights from a single feature to the logits.

![](images/toy-model-of-interference-weights/fig-009.png)

#### [Comparing Two Models with a Scatter Plot](#basic-phenomenology-scatter)

We can also train a second toy model (with a different random seed) and make a scatter plot of the weights against each other.The random seed specifies the weight initialization as well as how data is sampled from the generating distribution. The interference weights are independent, but the real weights are all significantly positive. Again, this is quite reminiscent of what we saw in Towards Monosemanticity!

![](images/toy-model-of-interference-weights/fig-010.png)

It's also interesting to look at a non-undertrained model here. We can see that in this configuration the interference weights become more structured, but we can also see that the real weights become perfectly agreed, while the interference weights don't.

![](images/toy-model-of-interference-weights/fig-011.png)

### [Getting More Realistic Phenomenology](#better-phenomenology-reproduction)

While the phenomenology of the above toy model is similar to Towards Monosemanticity, it is different in important ways. Consider the example of the [base64 feature](https://transformer-circuits.pub/2023/monosemantic-features/index.html#feature-base64) from Towards Monosemanticity:

![](images/toy-model-of-interference-weights/fig-012.png)

The base64 feature has overlap between its "real weights" and "interference weights". This is a critical part of why interference weights are such a hard problem for us. If we could just ignore small weights, things would be much easier! So we'd like an example that demonstrates why the problem is hard.

#### [A More Sophisticated Toy Model](#better-phenomenology-reproduction-model)

We'd like a more realistic toy model, but how can we get one that exhibits the relevant phenomenology? In this section, we'll generalize our toy model to one that does.

To start, note that we can think of our previous toy model as mimicking an identity circuit y = ReLU(Id ~x). In the toy model, we imagine the x is compressed into superposition (h=Wx) and then taken out of superposition (y' = ReLU(W^Th+b)).

We're now going to consider a different toy model where the circuit we're approximating in superposition is more complex:

y = ReLU(A x + v)

For some random matrix A, instead of an identity. Since A isn't necessarily symmetric, we need to "untie" our toy model, using different weights to project into and out of superposition.

h = W\_{down} x

y' = ReLU(W\_{up} h+b)

This toy model introduces new degrees of freedom – we need to specify how to generate A and v.

Different choices here can yield very different phenomenology, and it turns out to be somewhat tricky to find a regime in which (1) real weights and interference weights strongly overlap; (2) training two models doesn't always lead to the same "ideal superposition configuration", collapsing the scatter plot; (3) training two models doesn't collapse into two superposition solutions which make very different binary choices for weights, also making the scatter plot uninteresting; and (4) this continues to hold as one trains to convergence. One can achieve (1-3), but not (4), by having a block diagonal matrix, where each block is itself sparse (probability 0.5), and otherwise uniformly sampled between [0,1]. The block diagonal structure seems to really help with (2). We have v be a constant vector of -0.1.

If we put 128 features in superposition in 16 dimensions, with 8 blocks, 0.1 weight density within those blocks, and input feature density of 0.3, and train two models we get the following weight scatter plot:

![](images/toy-model-of-interference-weights/fig-013.png)

And if we focus on one model, we get the following weight histogram:

![](images/toy-model-of-interference-weights/fig-014.png)

It's also interesting to compare the learned weights to the ideal weights:

![](images/toy-model-of-interference-weights/fig-015.png)

### [What Should We Do about Interference Weights?](#filtering-interference-weights)

Ideally, we'd like to be able to separate the real weights and the interference weights, so we at least have the option to do circuit analysis only on the real weights.

Previously, we introduced two different definitions of interference weights:

* Interference Weights (defn 1). The interference weights are U-U^\*, the difference between the observed weights and ideal weights.
* Interference Weights (defn 2). The interference weights are the subset of the virtual weights U which don't improve the loss. That is, those which have \Delta L(U\_{ij}) < \epsilon.

(Note that these definitions are genuinely different, and are not the only possible definitions. See discussion in [Appendix 4](#appendix-4).)

In theory, these definitions could allow us to separate real weights from interference weights in any model. Unfortunately, both of these definitions are expensive to compute, and naively intractable to compute for all weights in a large model.For the first approach, and variants related to it, we naively need to materialize `n_features^2` matrices. Today we train CLTs with tens of millions of features, but we worry we will ultimately need billions. And then we need to optimize them over large amounts of data. Conversely, for the second approach, we need to test the loss when we ablate each of the `n_features^2` virtual weights, which avoids memory issues, but is perhaps worse in terms of compute… For smaller models, with smaller numbers of features, it might be possible to brute force things however.

Instead, we'll consider cheap heuristics which can be more computable proxies for these principled definitions. For now, let's consider four heuristic metrics: the original weights (big weights are likely to be real), expected attribution (weights with big average effects are likely to be real), target weighted attribution (weights with big effects on big things are more likely to be real), frequency (weights that often do something are more likely to be real), and an ideal baseline of the actual loss weight effect.

We can then try to do binary classification of real weights based on this, with the ground truth real weights defined as those with loss effect \Delta L(U\_{ij}) > \epsilon = 0.0001. We can then look at precision-recall curves:

![](images/toy-model-of-interference-weights/fig-016.png)

But we probably don't actually care about recall per se. Among real weights, there's a lot of variance in how important they are, and it's much worse to lose some important weights than others. Similarly, among interference weights, some are worse than others. So perhaps we should instead think about "loss gain" vs precision.Here "loss gain" is just the sum of \Delta L(U\_{ij}) for each individual weight, rather than evaluated on each ablated model. This is much more promising!

![](images/toy-model-of-interference-weights/fig-017.png)

Can we beat this? One tempting approach might be to take inspiration from compressed sensing – afterall, we're imagining the weights as actually living in a higher-dimensional space, and being compressed down via superposition. However, this requires that the map be linear, which may not be the case (see [Appendix 1](#appendix-1)).

### [Conclusion](#conclusion)

Interference weights may be the fundamental bottleneck preventing us from global circuit analysis of models. (Our recent work on attribution graphs was significantly designed to avoid interference weights!) The most ambitious vision for mechanistic interpretability requires global analysis, so addressing this seems quite important.

The naive approaches to dealing with interference weights are not scalable to large models with large numbers of features, but alternative heuristics may be. We can test these heuristics on toy models, and also test them by computing a ground truth for smaller numbers of weights in large models.
