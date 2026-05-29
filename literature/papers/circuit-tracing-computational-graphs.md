# Circuit Tracing: Revealing Computational Graphs in Language Models

> Source: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
> literature/README.md line 1
> Format: markdown (converted from HTML); images in `images/circuit-tracing-computational-graphs/`

---

### Contents

[Introduction](#introduction)

[Building an Interpretable Replacement Model](#building)

* [Architecture](#building-architecture)
* [From Cross-Layer Transcoder to Replacement Model](#building-replacement)
* [The Local Replacement Model](#building-local-replacement)

[Attribution Graphs](#graphs)

* [Constructing an Attribution Graph for a Prompt](#graphs-constructing)
* [Learning from Attribution Graphs](#graphs-tutorial)
* [Understanding and Labeling Features](#graphs-tutorial-features)
* [Grouping Features into Supernodes](#graphs-tutorial-supernodes)
* [Validating Attribution Graph Hypotheses with Interventions](#graphs-interventions)
* [Localizing Important Layers](#graphs-per-layer)
* [Factual Recall Case Study](#graphs-factual-recall)
* [Addition Case Study](#graphs-addition)

[Global Weights](#global-weights)

* [Global Weights in Addition](#global-weights-addition)

[Evaluations](#evaluating)

* [Cross-Layer Transcoder Evaluation](#evaluating-model)
* [Attribution Graph Evaluation](#evaluating-graphs)
* [Evaluating Mechanistic Faithfulness](#evaluating-model-faithfulness)

[Biology](#biology)

[Limitations](#limitations)

[Discussion](#discussion)

[Related Work](#related-work)

[Acknowledgments](#acknowledgments)

[Author Contributions](#author-contributions)

[Citation Information](#citation-info)

[CLT Implementation Details](#appendix-ml-details)

[Attribution Graph Computation](#appendix-attribution-graph-computation)

[Graph Pruning](#appendix-graph-pruning)

[Validating the Replacement Model](#appendix-lrm-validation)

[Nuances of Steering with Cross-Layer Features](#appendix-cross-layer-steering)

[Unexplained Variance and Choice of Steering Factors](#appendix-unexplained-var)

[Similar Features and Supernodes](#appendix-dupe-features)

[Iterative Patching](#appendix-patching)

[Details of Interventions](#appendix-interventions)

[Notes on the Interface](#appendix-interface)

[Residual Stream Norms Growth](#appendix-res-norms)

[Interference Weights over More Features](#appendix-interference-weights)

[Number Output Weights for Select Features](#appendix-full-number-weights)

[Comparison of Addition Features Between Small and Large Models](#appendix-arithmetic-comparison)

[Additional Evaluation Details](#appendix-eval-details)

[Prompt and Graph Lists](#appendix-all-graphs)

---

  
  

## [Introduction](#introduction)

Deep learning models produce their outputs using a series of transformations distributed across many computational units (artificial “neurons”). The field of mechanistic interpretability seeks to describe these transformations in human-understandable language. To date, our team’s approach has followed a two-step approach. First, we identify features, interpretable building blocks that the model uses in its computations. Second, we describe the processes, or circuits, by which these features interact to produce model outputs.

A natural approach is to use the raw neurons of the model as these building blocks.An alternative approach is to study the roles of gross model components such as entire MLP blocks and attention heads . This approach has identified interesting roles these components play in specific behaviors, but large components play a multitude of unrelated roles across the data distribution, so we seek a more granular decomposition. Using this approach, previous work successfully identified interesting circuits in vision models, built out of neurons that appear to represent meaningful visual concepts . However, model neurons are often polysemantic — representing a mixture of many unrelated concepts. One reason for polysemanticity is thought to be the phenomenon of superposition , in which models must represent more concepts than they have neurons, and thus must “smear” their representation of concepts across many neurons. This mismatch between the network's basic computational units (neurons) and meaningful concepts has proved a major impediment to progress to the mechanistic agenda, especially in understanding language models.

In recent years, sparse coding models such as sparse autoencoders (SAEs) , transcoders , and crosscoders  have emerged as promising tools for identifying interpretable features represented in superposition. These methods decompose model activations into sparsely active components (“features”),We use 'feature' following the tradition of 'feature detectors' in neuroscience and 'feature learning' in machine learning. Some recent literature uses the term 'latent,' which refers to specific vectors in the model's latent space. We find 'feature' better captures the computational role these elements play, making it more appropriate for describing transcoder neurons than SAE decoder vectors. which turn out in many cases to correspond to human-interpretable concepts. While current sparse coding methods are an imperfect way of identifying features (see [§ Limitations](#limitations)), they produce interpretable enough results that we are motivated to study circuits composed of these features. Several authors have already made promising steps in this direction .

Although the basic premise of studying circuits built out of sparse coding features sounds simple, the design space is large. In this paper we describe our current approach, which involves several key methodological decisions:

1. Transcoders. We extract features using a variant of transcoders  rather than SAEs, which allows us to construct an interpretable “replacement model” that can be studied as a proxy for the original model. Importantly, this approach allows us to analyze direct feature-feature interactions.
2. Cross-Layer. We base our analysis on cross-layer transcoders (CLT) , in which each feature reads from the residual stream at one layer and contributes to the outputs of all subsequent MLP layers of the original model, which greatly simplifies the resulting circuits. Remarkably, we can substitute our learned CLT features for the model's MLPs while matching the underlying model's outputs in ~50% of cases.
3. Attribution Graphs. We focus on studying “attribution graphs” which describe the steps a model used to produce an output for a target token on a particular prompt, using an approach similar to Dunefsky et al. . The nodes in the attribution graph represent active features, token embeddings from the prompt, reconstruction errors, and output logits. The edges in the graph represent linear effects between nodes, so the activity of each feature is the sum of its input edges (up to its activation threshold) (see [§ Attribution Graphs](#graphs)).
4. Linear Attribution Between Features. We design our setup so that, for a specific input, the direct interactions between features are linear. This makes attribution a well-defined, principled operation. Crucially, we freeze attention patterns and normalization denominators (following ) and use transcoders to achieve this linearity.Direct feature-feature interactions are linear because transcoder features “bridge over” the MLP nonlinearities, replacing their computation, and because we've frozen the remaining nonlinearities: attention patterns and normalization denominators. It's worth noting that strictly we mean that the pre-activation of a feature is linear with respect to the activations of earlier features.  
   Freezing attention patterns is a standard approach which divides understanding transformers into two steps: understanding behavior given attention patterns, and understanding why the model attends to those positions. This approach was explored in depth for attention models in A Mathematical Framework , which also [discussed](https://transformer-circuits.pub/2021/framework/index.html#additional-intuition) a generalization to MLP layers which is essentially the approach used in this paper. Note that factoring out understanding attention patterns in this way leads to the issues with attention noted in [§ Limitations: Missing Attention Circuits](#limitations-attention). However, we can then also take the same solution Framework takes of studying QK circuits. Features also have indirect interactions, mediated by other features, which correspond to multi-step paths.
5. Pruning. Although our features are sparse, there are still too many features active on a given prompt to easily interpret the resulting graph. To manage this complexity, we prune graphs by identifying the nodes and edges which most contribute to the model’s output at a specific token position (see [§ Appendix: Graph Pruning](#evaluating-graphs-pruning)). Doing so allows us to produce sparse, interpretable graphs of the model’s computation for arbitrary prompts.
6. Interface. We designed an interactive interface for exploring attribution graphs, and the features they're composed of, that allows a researcher to quickly identify and highlight key mechanisms within them.
7. Validation. Our approach to studying circuits is indirect – our replacement model may use different mechanisms from the underlying model. Thus, it is important that we validate the mechanisms we find in attribution graphs. We do so using perturbation experiments. Specifically, we measure the extent to which applying perturbations in a feature's direction produces changes to other feature activations (and to the model's output) that are consistent with the attribution graph. We find that across prompts, perturbation experiments are generally qualitatively consistent with our attribution graphs, though there are some deviations.
8. Global Weights. While our paper mostly focuses on studying attribution graphs for individual prompts, our methods also allow us to study the weights of the replacement model (“global weights”) directly, which underlie mechanisms across many prompts. In [§ Global Weights](#global-weights), we demonstrate some challenges with doing so – naive global weights are often less interpretable than attribution graphs due to weight interference. However, we successfully apply them to understand the circuits underlying small number addition.

The goal of this paper is to describe and validate our methodology in detail, using a few case studies for illustration.

* We begin with methods. We describe the setup of our replacement model ([§ Building an Interpretable Replacement Model](#building)) and how we construct attribution graphs ([§ Attribution Graphs](#graphs)), concluding with two case studies ([§ Factual Recall Case Study](#graphs-factual-recall), [§ Addition Case Study](#graphs-addition)). We then go on to explore approaches to constructing global circuits, including challenges and some preliminary methods for addressing them ([§ Global Weights](#global-weights)).

* We then provide a detailed quantitative evaluation of our cross-layer transcoders and the resulting attribution graphs ([§ Evaluations](#evaluating)), showing metrics by which CLTs provide Pareto-improvements over neurons and per-layer transcoders. Afterwards, we provide an overview of our companion paper, in which we apply our method to various behaviors of Claude 3.5 Haiku ([§ Biology](#biology)). We follow with a discussion of methodological limitations ([§ Limitations](#limitations)). These include the role of attention patterns, the impact of reconstruction errors, the identification of suppression motifs, and the difficulty of understanding global circuits. Addressing these limitations, and seeing what additional model mechanisms are then revealed, is a promising direction for future work.

* We close with a broader discussion ([§ Discussion](#discussion)) of the design space of methods for producing attribution graphs – parts of our approach can be freely remixed with others while retaining much of the benefit – and a review of related work ([§ Related Work](#related-work)).
* Our companion paper, [On the Biology of a Large Language Model](./biology.html), applies these methods to Claude 3.5 Haiku, investigating a diverse range of behaviors such as multiple hop reasoning, planning, and hallucinations.

We note that training a cross-layer transcoder can incur significant up-front cost and effort, which is amortized over its application to circuit discovery. We have found that this improves circuit interpretability and parsimony enough to justify the investment (see [cost estimates](#appendix-ml-details-plausible) for open-weights models and [discussion](#appendix-ml-details-efficiency) of cost-matched performance relative to per-layer transcoders). Nevertheless, we stress that alternatives like per-layer transcoders or even MLP neurons can be used instead (keeping the same steps 3–8 above), and still produce useful insights. Moreover, it is likely that better methods than CLTs will be developed in the future.

To aid replication, we share guidance on CLT [implementation](#appendix-ml-details), details on the [pruning method](#appendix-graph-pruning), and the [front-end code](https://github.com/anthropics/attribution-graphs-frontend) supporting the interactive graph analysis interface.

  
  
  

---

  
  

## [Building an Interpretable Replacement Model](#building)

### [Architecture](#building-architecture)

A cross-layer transcoder (CLT) consists of neurons (“features”) divided into L layers, the same number of layers as the underlying model. The goal of the model is to reconstruct the outputs of the MLPs of the underlying model, using sparsely active features. The features receive input from the model’s residual stream at their associated layer, but are “cross-layer” in the sense that they can provide output to all subsequent layers.Other architectures besides CLTs can be used for circuit analysis, but we’ve empirically found this approach to work well. Concretely:

* Each feature in the \ell^\text{th} layer “reads in” from the residual stream at that layer using a linear encoder followed by a nonlinearity.
* An \ell^\text{th} layer feature contributes to the reconstruction of the MLP outputs in layers \ell, \ell+1,\ldots , L, using a separate set of linear decoder weights for each output layer.
* All features in all layers are trained jointly. As a result, the output of the MLP in a layer \ell^\prime is jointly reconstructed by the features from all previous layers.

More formally, to run a cross-layer transcoder, let \mathbf{x^{\ell}} denote the original model’s residual stream activations at layer \ell. The CLT feature activations \mathbf{a^{\ell}} at layer \ell are computed as

\mathbf{a^{\ell}} = \text{JumpReLU}\!\left(W\_{enc}^{\ell} \mathbf{x^{\ell}}\right)

where W\_{enc}^{\ell} is the CLT encoder matrix at layer \ell.

We let \mathbf{y^{\ell}} refer to the output of the original model’s MLP at layer \ell. The CLT’s attempted reconstruction \mathbf{\hat{y}^{\ell}} of \mathbf{y^{\ell}} is computed using the JumpReLU activation function  as:

\mathbf{\hat{y}^{\ell}} = \sum\_{\ell'=1}^{\ell} W\_{dec}^{\ell' \to \ell} \mathbf{a^{\ell'}}

where W\_{dec}^{\ell' \to \ell} is the CLT decoder matrix for features at layer \ell' outputting to layer \ell.

To train a cross-layer transcoder, we minimize a sum of two loss functions. The first is a reconstruction error loss, summed across layers:

L\_\text{MSE} = \sum\_{\ell=1}^L \|\mathbf{\hat{y}^{\ell}} - \mathbf{y^{\ell}} \|^2

The second is a sparsity penalty (with an overall coefficient \lambda, a hyperparameter, and another hyperparameter c) summed across layers:

L\_\text{sparsity} = \lambda\sum\_{\ell=1}^L \sum\_{i=1}^N \textrm{tanh}(c \cdot \|\mathbf{W\_{dec, i}^{\ell}}\| \cdot a^{\ell}\_i)

Where N is the number of features per layer and \mathbf{W\_{dec, i}^{\ell}} is the concatenation of all decoder vectors of feature i.

We trained CLTs of varying sizes on a small 18-layer transformer model (“18L”)18L has no MLP in layer 0, so our CLT has 17 layers., and on Claude 3.5 Haiku. The total number of features across all layers ranged from 300K to 10M features (for 18L) and from 300K to 30M features (for Haiku). For more training details see [§ Appendix: CLT Implementation Details](#appendix-ml-details).

### [From Cross-Layer Transcoder to Replacement Model](#building-replacement)

Given a trained cross-layer transcoder, we can define a “replacement model” that substitutes the cross-layer transcoder features for the model’s MLP neurons – that is, where each layer's MLP output is replaced by its reconstruction by all CLTs that write to that layer. Running a forward pass of this replacement model is identical to running the original model, with two modifications:

* Upon reaching the input to the MLP in layer \ell, we compute the activations of the cross-layer transcoder features whose encoders live in layer \ell.
* Upon reaching the output of the MLP in layer \ell, we overwrite it with the summed outputs of the cross-layer transcoder features in this and previous layers, using their decoders for layer \ell.

Attention layers are applied as usual, without any freezing or modification. Although our CLTs were only trained using input activations from the underlying model, “running” the replacement model involves running CLTs on "off-distribution" input activations from intermediate activations from the replacement model itself.

As a simple evaluation, we measure the fraction of completions for which the most likely token output of the replacement model matches that of the underlying model. The fraction improves with scale, and is better for CLTs compared to a per-layer transcoder baseline (i.e., each layer has a standard single layer transcoder trained on it; the number of features shown refers to the total number across all layers). We also compare to a baseline of thresholded neurons, varying the threshold below which neurons are zeroed out (empirically, we find that higher neuron activations are increasingly interpretable, and we indicate below where their interpretability roughly matches that of features according to our auto-evaluations in [§&nbspQuantitative CLT Evaluations](#evaluating-model-clt)). Our largest 18L CLT matches the underlying model’s next-token completion on 50% of a diverse set of pretraining-style prompts from an open source dataset (see [§&nbspAdditional Evaluation Details](#appendix-eval-details)).We use a randomly sampled set of prompts and target tokens, restricting to those which the model predicts correctly, but with a confidence lower than 80% (to filter out “boring” tokens).

### [The Local Replacement Model](#building-local-replacement)

While running the replacement model can sometimes reproduce the same outputs as the underlying model, there is still a significant gap, and reconstruction errors can compound across layers. Since we are ultimately interested in understanding the underlying model, we would like to approximate it as closely as possible. To that end, when studying a fixed prompt p, we construct a local replacement model, which

* Substitutes the CLT for the MLP layers (as in the replacement model);
* Uses the attention patterns and normalization denominators from the underlying model's forward pass on p (as in );
* Adds an error adjustment to the CLT output at each (token position, layer) pair equal to the difference between the true MLP output on p and the CLT output on p (as in ).

After this error adjustment and freezing of attention and normalization nonlinearities, we've effectively re-written the underlying model's computation on the prompt p in terms of different basic units; all of the error-corrected replacement model's activations and logit outputs exactly match those of the underlying model. However, this does not guarantee that the local replacement model and underlying model use the same mechanisms. We can measure differences in mechanism by measuring how differently these models respond to perturbations; we refer to the extent to which perturbation behavior matches as “mechanistic faithfulness”, discussed in [§ Evaluating Mechanistic Faithfulness](#evaluating-model-faithfulness).This is similar in spirit to a Taylor approximation of a function f at a point a; both agree locally in a neighborhood of a but diverge in behavior as you move away.

The local replacement model can be viewed as a very large fully connected neural network, spanning across tokens, on which we can do classic circuit analysis:

* Its input is the concatenated set of one-hot vectors for each token in the prompt.
* Its neurons are the union of the CLT features active at every token position.
* Its weights are the summed interactions over all the linear paths from one feature to another, including via the residual stream and through attention, but not passing through MLP or CLT layers. Because attention patterns and normalization denominators are frozen, the impact of a source feature's activation on a target feature's pre-activation via each path is linear in the activation of the source feature. We sometimes refer to these as "virtual weights" because they are not instantiated in the underlying model.
* Additionally, it has bias-like nodes corresponding to error terms, with a connection from each bias to each downstream neuron in the model.

The only nonlinearities in the local replacement model are those applied to feature preactivations.

The local replacement model serves as the basis of our attribution graphs, where we study the feature-feature interactions of the local replacement model on the prompt for which it was made. These graphs are the primary object of study of this paper.

  
  
  

---

  
  

## [Attribution Graphs](#graphs)

We will introduce our methodology for constructing attribution graphs while working through a case study regarding the model’s ability to write acronyms for arbitrary titles. In the example we study, the model successfully completes a fictional acronym. Specifically, we give the model the prompt The National Digital Analytics Group (N and sample its completion: DAG). The tokenizer the model was trained with uses a special “Caps Lock” token, which means the prompt and completion are tokenized as follows: `The` `National` `Digital` `Analytics` `Group`  `(``⇪``n``dag`.

We explain the computation the model performs to output the “DAG” token by constructing an attribution graph showing the flow of information from the prompt through intermediary features and to that output.Due to the “Caps Lock” token, the actual target token is “dag”. We write the token in uppercase here and in the rest of the text for ease of reading. Below, we show a simplified diagram of the full attribution graph. The diagram shows the prompt at the bottom and the model’s completion on top. Boxes represent groups of similar features, and can be hovered over to display each feature’s visualization. We discuss our interpretation of features in [§ Understanding and Labeling Features](#graphs-tutorial-features). Arrows represent the direct effect of a group of features or a token on other features and the output logit.

The graph for the acronym prompt shows three main paths, originating from each of the tokens that compose the desired acronym. Paths originate from features for a given word, promoting features about “saying the first letter of that word in the correct position”, which themselves have positive edges to a “say DAG” feature and the logit. “say X” labels describe “output features”, which promote a specific token X, and arbitrary single letters are denoted with underscores. The “Word → say \_W” edges represent attention heads’ OV circuits writing to a subspace that is then amplified by MLPs at the target position. Each group of features also has a direct edge to the logit in addition to the sequential paths, representing effects mediated only via attention head OVs (i.e., paths to the output in the local replacement model that don’t “touch” another MLP layer).

In order to output “DAG”, the model also needs to decide to output an acronym, and to account for the fact that the prompt already contains N, and indeed we see features for “in an acronym” and “in an N at the start of an acronym” with positive edges to the logit. The word National has minimal influence on the logit. We hypothesize that this is due to its main contribution being through influencing attention patterns, which our method does not explain (see [§ Limitations: Missing Attention Circuits](#limitations-attention)).

 In the rest of this section, we explain how we compute and visualize attribution graphs.

### [Constructing an Attribution Graph for a Prompt](#graphs-constructing)

To interpret the computations performed by the local replacement model, we compute a causal graph that depicts the sequences of computational steps it performs on a particular prompt. The core logic by which we construct the graph is essentially the same as that of Dunefsky et al. , extended to handle cross-layer transcoders. Our graphs contain four types of nodes:

* The output nodes correspond to candidate output tokens. We only construct output nodes for the tokens required to reach 95% of the probability mass, up to a total of 10.We chose this threshold arbitrarily. Empirically, fewer than three logits are required to capture 95% of the probability in the cases we study.
* The intermediate nodes correspond to active cross-layer transcoder features at each prompt token position.
* The primary input nodes of the graph correspond to the embeddings of the prompt tokens.
* Additional input nodes (“error nodes”) correspond to the portion of each MLP output in the underlying model left unexplained by the CLT.

Edges in the graph represent direct, linear attributions in the local replacement model. Edges originate from feature, embedding, and error nodes, and terminate at feature and output nodes. Given a source feature node s and a target feature node t, the edge weight between them is defined to be A\_{s\rightarrow t} := a\_sw\_{s \rightarrow t}, where w\_{s \rightarrow t} is the (virtual) weight in the local replacement model viewed as a fully connected neural network and a\_s is the activation of the source feature.Alternatively, w\_{s \rightarrow t} is the derivative of the preactivation of t with respect to the source feature activation, with stop-gradients on all non-linearities in the local replacement model.

In terms of the underlying model, w\_{s \rightarrow t} is a sum over all linear paths (i.e., through attention head OVs and residual connections) connecting the source feature's decoder vectors to the target feature's encoder vector.

We now give details on how to efficiently compute these in practice, using backwards Jacobians. Let s be a source feature node at layer \ell\_s and context position c\_s and let t be a target feature node at layer \ell\_t and context position c\_t. We write J\_{c\_s, \ell\_s \rightarrow c\_t, \ell\_t}^{\blacktriangledown} for the Jacobian of the underlying model with a stop-gradient operation applied to all model components with nonlinearities – the MLP outputs, the attention patterns, and normalization denominators – on a backwards pass on the prompt of interest, from the residual stream at context position c\_t and layer \ell\_t to the residual stream at context position c\_s and layer \ell\_s. The edge weight from s to t is then

A\_{s \rightarrow t} = a\_s w\_{s \rightarrow t} = a\_s \sum\_{\ell\_s \leq \ell < \ell\_t} (W\_{\text{dec}, \;s}^{\ell\_s \to \ell})^T J\_{c\_s, \ell \rightarrow c\_t, \ell\_t}^{\blacktriangledown} W\_{\text{enc}, \;t}^{\ell\_t},

where

* W\_{\text{dec}, \;s}^{\ell\_s \to \ell} is the decoder vector of the feature for s writing to layer \ell,
* W\_{\text{enc}, \;t}^{\ell\_t} is the encoder vector of the feature for s.

The formulas for the other edge types are similar, e.g., an embedding-feature edge weight is given by w\_{s \rightarrow t} = \text{Emb}\_s^TJ\_{c\_s, \ell\_s \rightarrow c\_t, \ell\_t}^{\blacktriangledown} W\_{\text{enc}, \;t}^{\ell\_t}. Note that error nodes have no input edges. For all such formulas, and an expansion of the Jacobian in terms of paths in the underlying model, see [§ Appendix: Attribution Graph Computation](#appendix-attribution-graph-computation).)

Because we have added stop-gradients to all model nonlinearities in the computation above, the preactivation h\_t of any feature node t is simply the sum of its incoming edges in the graph: h\_t = \sum\_\mathcal{S\_t} w\_{s \rightarrow t} where S\_t is the set of nodes at earlier layers and equal or earlier context positions as t. Thus the attribution graph edges provide a linear decomposition of each feature's activity.

Note that these graphs do not contain information about the influence of nodes on other nodes via their influence on attention patterns, but do contain information about node-to-node influence through the outputs of frozen attention. In other words, we account for the information which flows from one token position to another, but not why the model moved that information.That is, our model ignores the "QK-circuits" but captures the "OV-circuits". Note also that the outgoing edges from a cross-layer feature aggregate the effect of its decodings at all of the layers that it writes to on downstream features.

While our replacement model features are sparsely active (on the order of a hundred active features per token position), attribution graphs are too large to be viewed in full, particularly as prompt length grows – the number of edges can grow to the millions even for short prompts. Fortunately, a small subgraph typically accounts for most of the significant paths from the input to the output.

To identify such subgraphs, we apply a pruning algorithm designed to preserve nodes and edges that directly or indirectly exert significant influence on the logit nodes. With our default parameters, we typically reduce the number of nodes by a factor of 10, while only reducing the behavior explained by 20%. See [§ Appendix: Graph Pruning](#appendix-graph-pruning) for methodological details of our algorithms and metrics.

### [Learning from Attribution Graphs](#graphs-tutorial)

Even following pruning, attribution graphs are quite information-dense. A pruned graph often contains hundreds of nodes and tens of thousands of edges – too much information to interpret all at once. To allow us to navigate this complexity, we developed an interactive attribution graphs visualization interface. The interface is designed to enable “tracing” key paths through the graph, retain the ability to revisit previously explored nodes and paths, and materialize the information needed to interpret features on an as-needed basis.

Below we show the interactive visualization for the attribution graph attributing back from the single token “DAG”:

The interface is interactive. Nodes can be hovered over and clicked on to display additional information. Subgraphs can also be constructed by using `Cmd/Ctrl+Click` to select a subset of nodes. In the subgraph, features can be aggregated into groups we call supernodes (motivated below in [§ Grouping Features into Supernodes](#graphs-tutorial-supernodes)).

### [Understanding and Labeling Features](#graphs-tutorial-features)

We use feature visualizations similar to those shown in our previous work, [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), in order to manually interpret and label individual features in our graph.We sometimes used labels from our automated interpretability pipeline as a starting point, but generally found human labels to be more reliable.

The easiest features to label are input features, which activate on specific tokens or categories of closely-related tokens and which are common in early layers, and output features, which promote continuing the response with specific tokens or categories of closely-related tokens and which are common in late layers. For example:

* This feature is likely an input feature because its visualization shows that it activates strongly on the word “digital” and similar words like “digitize”, but not on other words. We therefore label it a “digital” feature.
* This feature (a Haiku feature from the later [§ Addition Case Study](#graphs-addition)) is an input feature that activates on a variety of tokens that end in the digit 6, and even on tokens that are more abstractly related to 6 like “six” and “June”.
* This feature is likely an output feature because it activates strongly on several different tokens, but in each example, the token is followed by the text “dag”. Furthermore, the top of the visualization indicates that the feature increases the probability of the model predicting “dag” more than any other token (in terms of its direct effect through the residual stream). This suggests that it’s an output feature. Since output features are common, when labeling output features that promote some token or category X, we often simply write “say X”, so we give this example the label “say ‘dag’”.
* This feature (from the later [§ Factual Recall Case Study](#graphs-factual-recall)) is an output feature that promotes a variety of sports, though it also demonstrates some ways in which labeling output features can be difficult. For example, one must observe that “lac” is the first token of “lacrosse”. Also, the next token in the context after the feature activates often isn’t actually the name of a sport, but is usually a plausible place for the name of a sport to go.

Other features, which are common in middle layers of the model, are more abstract and require more work to label. We may use examples of contexts they are active over, their logit effects (the tokens they directly promote and suppress through the residual stream and unembedding), and the features they’re connected to in order to label them. For example:

* This feature activates on the first one or two letters of an unfinished acronym after an open parenthesis, for a variety of letters and acronyms, so we label it as continuing an acronym in general.
* This feature activates at the start of a variety of acronyms that all have D as their second letter, and many of the tokens it promotes directly have D as their second letter as well. (Not all of those tokens do, but we don’t expect logit effects to perfectly represent the feature’s functionality because of indirect effects through the rest of the model. We also find that features further away from the last layer have less interpretable logit effects.) For brevity, we may label this feature as “say ‘\_D’”, representing the first letter with an underscore.
* Finally, this feature activates on the first letter of various strings of uppercase letters that don’t seem to be acronyms, and the tokens it most suppresses are acronym-like letters, but its examples otherwise lack an obvious commonality, so we tentatively label it as suppressing acronyms.

We find that even imperfect labels for these features allow us to find significant structure in the graphs.

### [Grouping Features into Supernodes](#graphs-tutorial-supernodes)

Attribution graphs often contain groups of features which share a facet relevant to their role on the prompt. For example, there are three features active on “Digital” in our prompt which each respond to the word “digital” in different cases and contexts. The only facet which matters for this prompt is that the word “digital” starts with a “D”; all three features have positive edges to the same set of downstream nodes. Thus for the purposes of analyzing this prompt, it makes sense to group these features together and treat them as a unit. For the purposes of visualization and analysis, we find it convenient to group multiple nodes — corresponding to (feature, context position) pairs — into a “supernode.” These supernodes correspond to the boxes in the simplified schematic we showed above, reproduced below for convenience.

The strategy we use to group nodes depends on the analysis at hand, and on the roles of the features in a given prompt. We sometimes group features which activate over similar contexts, have similar embedding or logit effects, or have similar input/output edges, depending on the facet which is important for the claim we are making about the mechanism. We generally want nodes within a supernode to promote each other, and their effects on downstream nodes to have the same sign. While we experimented with automated strategies such as clustering based on decoder vectors or the graph adjacency matrix, no automated method was sufficient to cover the range of feature groupings required to illustrate certain mechanistic claims. We further discuss supernodes and potential reasons for why they are needed in [Similar Features and Supernodes](#appendix-dupe-features).

### [Validating Attribution Graph Hypotheses with Interventions](#graphs-interventions)

In attribution graphs, nodes suggest which features matter for a model's output, and edges suggest how they matter. We can validate the claims of an attribution graph by performing feature perturbations in the underlying model, and checking if the effects on downstream features or on the model outputs match our predictions based on the graph. Features can be intervened on by modifying their computed activation and injecting its modified decoding in lieu of the original reconstruction.

Features in a cross-layer transcoder write to multiple output layers, so we need to decide on a range of layers in which to perform our intervention. How might we do this? We could intervene on a feature’s decoding at a single layer just like we would for a per-layer transcoder, but edges in an attribution graph represent the cumulative effect of multiple layers’ decodings, so intervening at a single layer would only target a subset of a given edge. In addition, we’ll often want to intervene on more than one feature at a time, and different features in a supernode will decode to different layers.

To perform interventions over layer ranges, we modify the decoding of a feature at each layer in the given range, and run a forward pass starting from the last layer in the range. Since we aren’t recomputing a layer’s MLP output based on the result of interventions earlier in the range, the only change to the model’s MLP outputs will be our intervention. We call this approach “constrained patching”, as it doesn’t allow an intervention to have second-order effects within its patching range. See [§ Appendix: Iterative Patching](#appendix-patching) for a description of another approach we call “iterative patching”, and see [§ Appendix: Nuances of Steering with Cross-Layer Features](#appendix-cross-layer-steering) for a discussion of why more naive approaches, such as adding a feature’s decoder vector at each layer during a forward pass of the model, risk double counting a feature's effect.

Below, we illustrate a multiplicative version of constrained patching, in which we multiply a target feature’s activation by M in the [\ell - 1, \ell] layer range. Note that MLP outputs at further layers are not directly affected by the patch.They can be indirectly affected since they sit downstream of affected nodes.

Attribution graphs are constructed by using the underlying model’s attention patterns, so edges in the graph do not account for effects mediated via QK circuits. Similarly, in our perturbation experiments, we keep attention patterns fixed at the values observed during an unperturbed forward pass. This methodological choice means our results don't account for how perturbations might have altered the attention patterns themselves.

Returning to our acronym prompt, we show the results of patching supernodes, starting with suppressing the “Group” supernode. Below, we overlay patching effects onto supernode schematics for clarity, displaying the effect on other supernodes and the logit distribution. Note that in this diagram, the position of the nodes in the figure is not meant to correspond to token positions unless explicitly noted.

We now show the results of suppressing some supernodes on the aggregate activation of other supernodes and on the logit. For each patch, we set every feature in a node’s activation to be the opposite of its original value (or equivalently, we steer multiplicatively with a factor of −1).For a discussion of why we steer negatively instead of ablating the feature, see [§ Unexplained Variance and Choice of Steering Factors](#appendix-unexplained-var). We then plot each node’s total activation as a fraction of its original value.For each patched supernode, we choose the end-layer range which causes the largest suppression effect on the logit. We use an orange outline to highlight nodes downstream of one another for which we would hypothesize patching to have an effect.

We see that inhibiting features for each word inhibits the related initial features in turn. In addition, the supernode of features for “say DA\_” is affected by inhibitions of both the “Digital” and “Analytics” supernodes.

### [Localizing Important Layers](#graphs-per-layer)

The attribution graph also allows us to identify in which layers a feature’s decoding will have the greatest downstream effect on the logit. For example, the “Analytics” supernode features mostly contribute to the “dag” logit indirectly through intermediate groups of features “say \_A”, “say DA\_”, and “say DAG” which live in layers 13 and beyond.

We would thus expect steering negatively on an “Analytics” feature to have an effect on the dag logit which plateaus before layer 13 and then decreases in magnitude as we approach the final layer. The decrease is caused by the constrained nature of our intervention. If a patching range includes all the “say an acronym” features, it will not change their activation, because constrained patching doesn’t allow knock-on effects. Below, we show the effect of steering with each Analytics feature, keeping the start layer set to 1 and sweeping over the patching end layer.Note that we use a large steering factor in this experiment. For a discussion of this, see [§ Unexplained Variance and Choice of Steering Factors](#appendix-unexplained-var).

### [Factual Recall Case Study](#graphs-factual-recall)

We now turn to the question of factual recall by studying how the model completes Fact: Michael Jordan plays the sport of with basketball with 65% confidence . We start by computing an attribution [graph](./static_js/attribution_graphs/index.html?slug=mj-18l). We group semantically similar features into supernodes like we did for the acronym study.

The supernode diagram below shows two primary paths. One path originates from the “plays” and “sport” tokens and promotes “sport” and “say a sport” features, which in turn promote the logits for basketball, football, and other sports. The other path originates from “Michael Jordan and other celebrities” and promotes basketball related features, which have positive edges to the basketball logit and negative edges to the football logit. In addition to these sequential paths, some groups of features such as “Michael Jordan” and “sport/game of” have direct edges to the basketball logit, representing effects mediated only via attention head OVs, consistent with the findings of Batson et al. .

We also display the full interactive graph below.

In addition, a complex set of mechanisms seems to be involved in contributing information about the entity Michael Jordan to the residual stream at “Jordan”, as observed in Nanda et al. . We have grouped into one supernode features sensitive to "Michael", an L1 feature which has already identified the token pair “Michael Jordan”, features for other celebrities, and polysemantic features firing on “Michael Jordan” and other unrelated concepts. Note that we choose to include some polysemantic features in supernodes as long as they share a facet relevant to the prompt, such as this feature which activates more strongly on the word “synergy” than on “Michael Jordan”. We evaluate features in more depth in [Qualitative Feature Evaluations](#evaluating-model-features).

Steering experiments can once more allow us to validate the hypotheses proposed by the graph.

Ablating either the “sport” or “Michael Jordan” supernode has a large effect on the logit but a comparatively smaller effect on the other supernode, confirming the parallel path structure. In addition, we see that suppressing the intermediate “basketball discussion” supernode also has a large effect on the logit.

### [Addition Case Study](#graphs-addition)

We now consider the simple addition prompt calc: 36+59=. We use the prefix "calc:" because the 18L performs much better on the problem with it. This prefix is not necessary for Haiku 3.5, but we include it nevertheless for the purposes of direct comparison in later sections. Unlike previous sections, we show results for Haiku 3.5 because the patterns are clearer and show the same structure (see [§ Appendix: Comparison of Addition Features …](#appendix-arithmetic-comparison) for a side-by-side comparison). We look at small-number addition because it is one of the simplest behaviors exhibited competently by most LLMs and human adults (try the problem in your head to see if your approach matches the model's!).

We supplement the generic feature visualization (on arbitrary dataset examples) with one which explicitly covers the set of two-digit addition problems, allowing us to get a crisp picture of what each feature does. Following Nikankin et. al. , who analyzed neurons, we visualize each feature active on the = token with three plots:

* An operand plot, displaying its activity on the 100 × 100 grid of potential inputs.
* An output weight plot, displaying its direct weights on the outputs for [0, 99].During analysis we visualize effects on [0, 999]. This is important to understand effects beyond the first 100 number tokens (e.g., the feature predicting 95 mod 100), but we only show [0, 99] for simplicity.
* An embedding weight plot (or “de-embedding” ), displaying the direct effect of embedding vectors on a feature’s encoder. This is shown in the same format as the output weight plot.

We show an example plot of each of these three types below for different features. On this restricted domain, the operand plots are complete descriptions of the CLT features as functions. Stripes and grids in these plots represent different kinds of structure (e.g. diagonal lines indicate constraints on the sum, while grids represent modular constraints on the inputs).

In the supernode diagram below, we see information flow from input features, which split out the final digit, the number, and the magnitude of the operands to three major paths: a final-digit path (mod 10) (light-brown, right), a moderate precision path (middle), and a low precision path (dark brown, left),With hover you can see the variability in the precision of the low precision lookup table features and the moderate precision sum features. which collectively produce a moderate precision value of the sum and the final digit of the sum; these finally constructively interfere to give both the mod 100 version of the sum and the final output.

We provide the equivalent interactive graph for 18L [here](./static_js/attribution_graphs/index.html?slug=calc-36-plus-59-18l).

The supernode graph suggests a taxonomy of features underlying this task, in which features vary along two major axes:As in other sections of this work, we apply these labels manually.

* Computational role

* Sum features have diagonal operand plots, and fire on pairs of inputs whose sum satisfies some condition.
* Lookup Table Features have plots that look like a grid, and consist of inputs a and b satisfying `condition1(a) AND condition2(b)`. We discuss these in more detail below.
* Add Function Features have plots with horizontal or vertical bars. One addend satisfies some condition, or an `OR` operation merges two conditions across addends.
* Mostly Active Features are active on the "=" token of most of our 10,000 addition prompts.
* Miscellaneous Features have all sorts of strange properties, but often look like hybrid activation patterns from the above types. We find that these have lower influence on the outputs.

* Condition properties

* Precision: we find conditions with ones-digit precision (sum=\_5 or =59), with exact range (of width e.g. 2 or 10), and with fuzzy ranges of width ranging from 2–50.
* Modularity: we find features that are sensitive to the sum or operand value in absolute terms, mod 10, mod 100, and less commonly, mod 2, mod 5, mod 25, and mod 50.
* Pattern: we find features sensitive to a regex style pattern in an input or output, such as “starts with 51”, as in . These do not feature as prominently in our addition graphs, having low influence on the model's output, but they do exist, and may be more important for other tasks involving numbers.

These findings broadly agree with other mechanistic studies showing that language models trained on natural language corpora perform addition using parallel heuristics involving magnitudes and moduli that constructively interfere to produce the correct answer . Namely, Nikankin et al.  proposed a “bag of heuristics” interpretation, recognizing a set of “operand” features (equivalent to our “add X” features) and “result” features (equivalent to our “sum” features) exhibiting high- and low-precision and different modularities in sensing the input and producing the output.

We also identify the existence of lookup table features, which seem to be an interesting consequence of the architecture used by both the model and the CLT. Neurons and CLT feature activations are computed by applying a nonlinearity to the sum of their inputs. This produces a “parallelogram constraint” on the response of a feature to a set of inputs: namely, if a feature f is active on two inputs of the form x+y and z+w, then it must be active on at least one of the inputs x+w or z+y. This follows since the preactivation of f is an affine function of the operands.Write f\_{pre} for the preactivation of x. Then f\_{pre}(x + y) + f\_{pre}(z + w) = f\_{pre}(x + w) + f\_{pre}(y + z). If both terms on the LHS are positive, at least one on the right must be. In particular, it is impossible for input features to produce a general sum feature in one step. For example, a general “sum = 5” feature which fires for 1+4 and 2+3 would need to fire for at least one of 1+3 or 2+4. So some intermediate step between copying over information about both inputs to the “=” token and producing a property of their sum is required. CLT lookup table features represent these intermediate steps for addition.Concretely, the intermediate info is the ones digit produced from adding the ones digit of the operands and the approximate magnitude of the sum of two numbers in the 20s.

To validate that the structure we observe in the attribution graph matches the causal structure of the model, we perform a series of interventions. For each supernode, we perturb it to the negative of its original value, and measure the result on all subsequent supernodes and the outputs. We find results largely consistent with the graph:

In particular, inhibiting the ones-digit feature on either of the input tokens suppresses the entire ones-digit pathway (the \_6 + \_9 lookup table features, the resulting sum=\_5 and sum= \_95 features), while leaving the magnitude pathway mostly intact, including the sum~92 features. Remarkably, when suppressing \_6, the model confidently outputs 98 instead of the correct answer 95; the tens digit from the original problem is preserved by the other magnitude signals but the ones digit is that which would result from adding 9 to itself. (Suppressing \_9, however, results in an output of 91, not 92, so such numerology must be taken with a grain of salt.). Conversely, inhibiting low-precision features on either inputs (~30 and ~59) suppress the low-precision lookup table features, the magnitude sum feature, and the appropriate sum features while leaving the ones-digit pathway alone.

We also show the quantitative effects of perturbations on the outputs, finding that negatively steering the \_6 + \_9 lookup table features smears the result out over a range of 5, while negatively steering the final sum=\_95 feature smears the result out to a wider band (perhaps coming from sum~92 features).

We will investigate how CLT features interact across the full range of two-digit addition prompts below, after establishing the framework for global weights that we use to generalize this circuit to other inputs.

  
  
  

---

  
  

## [Global Weights](#global-weights)

The attribution graphs we construct show how features interact on a specific prompt to produce the model's output, but we are also interested in a more global picture of how features interact across all contexts. In a classic multi-layer perceptron, the global interactions are provided by the weights of the model: the direct influence of one neuron on another is just the weight between them if the neurons are in consecutive layers; if neurons are further apart, the influence of one on another factors through intermediate layers. In our setup, the interaction between features has a context independent component and a context dependent component. We would ideally like to capture both: we want a set of global weights which are context independent, but also capture network behavior across all possible contexts. In this section we analyze the context independent component (a kind of “virtual weight”), a problem with them (large “interference” terms with no causal effect on distribution), and one approach using co-activation statistics to deal with the interference.

On a specific prompt, a source CLT feature (s) influences a target (t) via three kinds of paths:

1. residual-direct: s’s decoders write to the residual stream, where it is read in at a later layer by t’s encoder.
2. attention-direct: s’s decoders write to the residual stream, are transported by some number of attention head OV steps, and then read by t’s encoder.
3. indirect: paths from s to t are mediated by other CLT features.

We note that the residual-direct influence is simply the product of the first feature's activation on this prompt times a [virtual weight](https://transformer-circuits.pub/2021/framework/index.html#virtual-weights) which is consistent across inputs.The attention-direct terms can also be written in terms of virtual weights given by multiplying various decoder vectors by a series of attention head OVs, and then by an encoder, but these get scaled on a prompt by both the source feature activation and the attention patterns, which make their analysis more complex. These virtual weights are a simple form of global weights because of this consistent relationship. Virtual weights have been derived between many different components in neural networks, including attention heads , SAE features , and transcoder features . For CLTs, the virtual weight between two features is the inner product between the encoder of the downstream feature and the sum of decoders in between these two features.

More formally, let \ell\_s and \ell\_t be the layers of the encoder weights for features s and t. Let L\_{st}  be the set of layers in between these features such that  \forall \ell \in L\_{st}, s \leq \ell < t. Feature s writes to all MLP outputs in L\_{st} before reaching feature t. Let W\_{\text{dec}}^{s,\ell} be the decoder weights for feature s targeting layer \ell, and W\_{\text{enc}}^t be the encoder weights for feature t. Then, the virtual weights are computed as:  V\_{st} = \big< \sum\_{\ell \in L\_{st}} W\_{\text{dec}}^{s,\ell}, W\_{\text{enc}}^t \big> Attribution graph edges consist of a sum of the residual-direct contribution (the virtual weight multiplied by a feature’s activation) plus the attention-direct contribution.

There is one major problem with interpreting virtual weights: interference . Because millions of features are interacting via the residual stream, they will all be connected, and features which never activate together on-distribution can still have (potentially large) virtual weights between them. When this happens, the virtual weights are not suitable global weights because these connections never impact network function.

We can see interference at play in the following example: below, we take a “Say a game name” feature in 18L and plot its largest virtual weights by magnitude. Green bars indicate a positive connection and purple bars indicate a negative one. Many of the most strongly-connected features are hard to interpret or not clearly related to the concept.

You might consider this a sign that virtual weights or our CLTs aren’t capturing interpretable connections. However, we can still uncover many interpretable connections by trying to remove interference from these weights.We also see interference weights when looking at a larger sample of features in the [§ Appendix: Interference Weights over More FeaturesAppendix](#appendix-interference-weights).

There are two basic solutions to this problem. One is to restrict the set of features being studied to those active on a small domain (as we do in [§ Global Weights in Addition](#global-weights-addition)). The other is to bring in information about the feature-feature coactivation on the data distribution.

For example, let a\_i be the activation for feature i. We can compute an expected residual attribution value by multiplying the virtual weight as follows.  V\_{ij}^{\text{ERA}} = \mathbb{E}\big[ {\small 1}\kern{-0.33em}1(a\_j > 0)V\_{ij}a\_i \big] = \mathbb{E}\big[ {\small 1}\kern{-0.33em}1(a\_j > 0)a\_i \big] V\_{ij}  This represents the average strength of a residual-direct path across all of the prompts we’ve analyzed (also computed by Dunefsky et al. ). This is similar to computing the average of all attribution graphs within a context position across many tokens.It only represents the residual-direct component and does not include the attention-direct one. The indicator function in this expression ({\small 1}\kern{-0.33em}1(a\_j > 0)) captures how attributions are only positive when the target feature is active. As small feature activations are often polysemantic, we instead weight attributions using the target activation value:  V\_{ij}^{\text{TWERA}} = \frac{\mathbb{E}\big[a\_ja\_i\big]}{\mathbb{E}\big[ a\_j \big]} V\_{ij}  We call this last type of weight target-weighted expected residual attribution (TWERA). As shown in the equations, both of these values can be computed by multiplying the original virtual weights by (“on-distribution”) statistics of activations.

Now, we revisit the example game feature from before but with connections ordered by TWERA. We also plot each connection’s “raw” virtual weight for comparison. Many more of these connections are interpretable, suggesting that the virtual weights extracted useful signals but we needed to remove the interference in order to see them. The most interpretable features from the virtual weight plot above (another “Say a game name” and “Ultimate frisbee” feature) are preserved while many unrelated concepts are filtered out.

TWERA is not a perfect solution for interference. Comparing TWERA values to the raw virtual weights shows that many extremely small virtual weights have strong TWERA values.  Note that these weights cannot be 0 as this would also send TWERA to 0. This indicates that TWERA heavily relies on the coactivation statistics and strongly changes which connections are important beyond simply removing the large interference weights. TWERA also does not handle inhibition well (like attribution generally). We will explore these issues further in future work.

Still, we find that global weights give us a useful window into how features behave in a broader range of contexts than our attribution graphs. We’ll use these methods to complement our understanding in the rest of this paper and in [the companion paper](./biology.html).

### [Global Weights in Addition](#global-weights-addition)

We now return to the simple addition problem from [above](#graphs-addition) on Haiku 3.5, and show how data-independent virtual weights reveal clear structure between the types in our taxonomy of addition features. We again consider completions of the 10,000 prompts calc: a+b= for a,b ∈ [0, 99]. In addition to the operand plots (again, defined [above](#graphs-addition)), we inspect the virtual weight graph after restricting the large n\_\text{feat} \times n\_\text{feat} virtual weight matrix to the set which are active on at least 10 of the set of 10,000 addition prompts. This allows us to see all the feature-feature interactions that can occur via direct residual paths.

In the neighborhood of the features appearing in the 36+59 prompt above, we see:

* The \_6 + \_9 lookup table features feed into other sum features (sum=\_15, sum=\_25, etc.), which likely activate when combined with other magnitude features.
* The add \_9 feature feeds into other ones-digit lookup table features, (\_9 + \_9, \_0 + \_9, etc.).
* The sum = \_5 feature is fed by other lookup table features.
* The medium-precision ~36+60 lookup table feature is fed by an add ~62 feature in addition to the add ~57 feature we see on this prompt.

We provide an [interactive interface](./static_js/addition/index.html) to explore the virtual weights connecting all 2931 features prominent on two-digit addition problems in our smaller 18L model.

We find that restricting to features active on this narrow domain of addition problems produces a global circuit graph where most edges are interpretable in terms of the operand function realized by the source and target features. Moreover, the connections between features recapitulate a more general version of the graph in the previous section; add features detect a specific operand as part of the input, lookup table features propagate this information to sum features which (in concert with the previous features) produce the model’s final answer.

Several of the features we find take the form of heuristics as in Nikankin et al. : features whose operand plots have predictive power directly push that prediction to the outputs: the low precision features promote outputs in the range matching their operands; the ones digit lookup table features directly promote outputs matching their mappings (e.g. \_6 + \_9 directly upweights \_5 outputs). Almost none of these features represent the full solution until the very last layers of the model. As viewed by CLTs, the models use several intermediate heuristics rather than a coherent procedural program with a single confident output.

Our focus on the computational steps the model uses to perform addition is complementary to concurrent work by Kantamneni and Tegmark  which begins from representations. Inspired by the observation of spikes in the Fourier decomposition of the embedding vectors for integers, they find low-dimensional subspaces highly correlated with numbers' magnitudes and mod 2, 5, 10, and 100 components. Projecting to those subspaces preserves much of the model's performance on the task, consistent with a “Clock” algorithm performing separate calculations in each modulus, which interfere constructively at the end; the CLT features show essentially high- and low-precision versions of that method. Some of the important features we find have operand plots similar to their neurons, which they fit as a (thresholded) sum of Fourier modes.There's no guarantee that the CLT features are the most parsimonious way to split up the computation, and it's possible some of our less important, roughly-periodic features which are harder to interpret are artifacts of the periodic aspects of the representation. Some of the important features appearing in our graphs (such as operands or sums that start with fixed digits, e.g. 95\_ and 9\_) aren't describable in Fourier terms, consistent with the existence of some error in their low-rank approximation.In [§ Appendix: Number Output Weights over More Features](#appendix-full-number-weights), we show output weight plots for the 9\_ and 95\_ features on all number predictions from [0,999]. We also show a miscellaneous feature that promotes “simple numbers”: small numbers, multiples of 100 and a few standouts like 360. Identifying the representational basis of the ensemble of computational strategies revealed by our unsupervised approach is a promising direction for future work.

Altogether, we’ve replicated a view of the base model using heuristics finding matching CLT features, we’ve shown how these heuristics contribute to separable pathways through intervention experiments, and we've demonstrated how these heuristics are connected, building off one another to collectively solve the addition task.

  
  
  

---

  
  

## [Evaluations](#evaluating)

In this section, we perform qualitative and quantitative evaluations of transcoder features and the attribution graphs derived from them, focusing especially on interpretability and sufficiency. For readers interested in a higher level discussion of findings and limitations, we recommend skipping ahead to [§ Biology](#biology) and [§ Limitations](#limitations).

Our methods produce causal graph descriptions of the model’s mechanisms on a particular prompt. How can we quantify how well these descriptions capture what is really going on in the model? It is difficult to distill this question to one number, as several factors are relevant:

Interpretability. How well do we understand what individual features “mean”? We attempt to quantify interpretability in a few ways [below](#evaluating-model-clt); however, we still rely heavily on [subjective evaluation](#evaluating-model-features) in practice. The coherence of our groupings of features into “supernodes” also warrants evaluation. We do not attempt to quantify this in this work, instead leaving it to readers to verify for themselves that our groupings are sensible and interpretable. We also note that in the context of attribution graphs, interpretability of the graph is just as important as interpretability of individual features. To that end, we quantify one notion of graph simplicity: [average path length](#evaluating-graphs-paths).

Sufficiency. To what extent are our (pruned) attribution graphs sufficient to explain the model’s behavior? We attempt to quantify this in several ways. The most straightforward such evaluation is our measurement of how well the replacement model’s outputs match the underlying model, discussed in [§ From Cross-Layer Transcoder to Replacement Model](#building-replacement) . This is a “hard” evaluation in that a single error anywhere along the computational graph can severely degrade performance. We also compute a few “softer” measures of sufficiency [below](#evaluating-graphs-comparing), that measure the proportion of error nodes in attribution graphs. Note that in many instances, we present schematics of subgraphs of a pruned attribution graph that portray what we believe to be its most noteworthy components. We intentionally do not measure the sufficiency of these subgraphs, as they often intentionally exclude “boring” but necessary parts of the graph (e.g. “this is a math problem” features in addition prompts). We leave it to future work to find more principled ways to distill attribution graphs to their “interesting” components and quantify how much (and what kind of) information is lost. One route is to consider families of prompts, and to exclude from considerations features that are present across all prompts within a family (but see ).

Mechanistic faithfulness. To what extent are the mechanisms we identify actually used by the model? To measure this, we perform perturbation experiments (such as inhibiting active features) and measuring whether the effects agree with what is predicted by the local replacement model (the underlying object portrayed by our attribution graphs). We attempt to do so quantitatively [below](#evaluating-model-faithfulness), and we also validate faithfulness on our specific case studies, in particular focusing on faithfulness of the mechanisms we have identified as interesting / important. Note that our notion of mechanistic faithfulness is related to the idea of necessity of circuit components to a model’s computation. However, necessity can be a somewhat restrictive notion – mechanisms that are not strictly “necessary” for the model’s output may still be important to identify, especially in cases where multiple mechanisms cooperate in parallel to contribute to a computation, as we often observe.

We note that the specific evaluations we use are in many cases new to this work. In part this is because our work is somewhat unique in focusing on attribution graphs for individual prompts, rather than identifying circuits underlying the model’s performance of an entire task. Developing better automatic methods for evaluating interpretability, sufficiency, and faithfulness of the entire pipeline (features, supernodes, graphs) is an important subject of future research. See [§ Related Work](#related-work) for more detail on prior circuit evaluation methods.

### [Cross-Layer Transcoder Evaluation](#evaluating-model)

#### [Qualitative Feature Evaluations](#evaluating-model-features)

For CLT features to be useful to us, they must be human-interpretable (perhaps in the future it will suffice for them to be AI-interpretable!). Interpretability is ultimately a qualitative property – the best gauge of the interpretability of our features is to view them in action. A standard (though incomplete) tool for understanding what a feature represents is to view the dataset examples for which it is active (we refer to the collection of such examples as our “feature visualization”). We provide thousands of feature visualizations in the context of our case studies of circuits later in this paper and in [the companion paper](./biology.html). Below we also show 50 randomly sampled features from assorted layers of each model.

Our feature visualizations show snippets of samples from public datasets ([Common Corpus](https://huggingface.co/blog/Pclanglais/common-corpus), The Pile with books removed , LMSYS Chat 1m , and [Isotonic Human-Assistant Conversation](https://huggingface.co/datasets/Isotonic/human_assistant_conversation)) that most strongly activate the feature, as well as examples that activate the feature to varying degrees interpolating between the maximum activation and zero. Highlights indicate the strength of the feature’s activation at a given token position. We also show the output tokens that the feature most strongly promotes / inhibits via its direct connections through the unembedding layer (note that this information is typically more meaningful for features in later model layers).

18L Features (Hover)

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Layer 1 | Layer 5 | Layer 9 | Layer 13 | Layer 17 |

Haiku Features (Hover)

|  |  |  |
| --- | --- | --- |
| First layer | Mid-layer | Final layer |

At a very coarse level, we find several types of features:

* Input features that represent low-level properties of text (e.g. specific tokens or phrases). Most early-layer features are of this kind, but such features are also present in middle and later layers.
* Features whose activations represent more abstract properties of the context. For example, a feature for the danger of mixing common cleaning chemicals. These features appear in middle and later layers.
* Features that perform functions, such as an “add 9” feature that causes the model to output a number that is nine greater than another number in its context. These tend to be found in middle and later layers.
* Output features, whose activations promote specific outputs, either specific tokens or categories of tokens. An example is a “say a capital” feature, which promotes the tokens corresponding to the names of different U.S. state capitals.
* Polysemantic features, especially in earlier layers, such as this feature that activates for the token “rhythm”, Michael Jordan, and several other unrelated concepts.

In line with our [previous results on crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html) , we find that features also vary in the degree to which their outputs “live” in multiple layers – some features contribute primarily to reconstructing one or a few layers, while others have strong outputs all the way through the final layer of the model, with most features falling somewhere in between.

We also note that the abstractions represented by Haiku features are in many cases richer than those in the smaller 18L model, consistent with the model’s greater capabilities.

#### [Quantitative CLT Evaluations](#evaluating-model-clt)

In [§&nbspFrom Cross-Layer Transcoder to Replacement Model](#building-replacement), we evaluated the ability of our CLTs to reproduce the computation of the underlying model. Here, we measure reconstruction error, sparsity (measured by “L0”, the average number of features active per input token), and feature interpretability. As we increased the size of our CLT, we observed Pareto-improvements in reconstruction error (averaged across layers) and feature sparsity (in 18L, reconstruction error decreased at a roughly fixed L0, while in Haiku, reconstruction error and L0 both decreased). In our largest 18L run (10M features), we attained a normalized mean reconstruction error of ~11.5% and an average L0 of 88. In our largest Haiku run (30M features), we attained a normalized reconstruction error of 21.7%, and an average L0 of 235.

We also computed two LLM-based quantitative measures of interpretability, introduced and described in more detail in :

* Sort Eval: we take two randomly sampled features and identify the set of dataset examples that activate them most strongly. We present these sets of examples to Claude, including the token-by-token feature activation information. Then we take other dataset examples that activate only one of the features, present these to Claude, and ask it to guess which feature these examples correspond to (based on the initial example sets). The final evaluation score is the empirical likelihood that Claude guesses the correct feature on any given pair.

* Contrastive Eval: we generate (using Claude) pairs of prompts that are similar in content and structure but differ in one key respect. We compute the sets of features that activate on only one of the prompts but not the other. We present Claude with the feature vis for each such feature, along with the two prompts, and ask it to guess which prompt caused the feature to activate. The final evaluation score is the empirical likelihood that Claude guesses the correct prompt for features across trials.

We find that according to both measures, the quality of CLT features improves with scale (alongside improving reconstruction error) – see plots below. We scale the number of training steps with the number of features, so improvements reflect a combination of both forms of scaling – see [§ Appendix: CLT Implementation Details](#appendix-ml-details) for details.

We also compare CLTs to two baselines: per-layer transcoders (PLTs) trained at each layer of the model, and the raw neurons of the model thresholded at varying activation levels. Specifically, we sweep over a range of scalar thresholds, and for each value we clamp all neurons with activation less than the threshold to 0. For our metrics and graphs, we then only consider neurons above the threshold.  We find that on all metrics, CLTs outperform PLTs, and both CLTs and PLTs substantially outperform the Pareto frontier of thresholded neurons.

### [Attribution Graph Evaluation](#evaluating-graphs)

Case studies in [§&nbspAttribution Graphs](#graphs) focused on qualitative observations derived from attribution graphs. In this section, we describe our more quantitative evaluations used to compare methodological choices and dictionary sizes. In each of the following subsections, we will introduce a metric and compare graphs generated using (1) cross-layer transcoders, (2) per-layer transcoders for every layer, and (3) thresholded neuronsWe choose the threshold at approximately the point where the neurons achieve similar automated interpretability scores as our smallest dictionaries, see [figures above](#evaluating-model-clt).. To connect the quantitative to the qualitative, we will link to graphs which score especially high or low on each of these metrics.

While we don’t treat these metrics as fundamental quantities to be optimized, they have proven a useful guide for tracking ML improvements in dictionary learning and to flag prompts our method performs poorly on.

#### [Computing Node Influence](#evaluating-graphs-influence)

Our graph-based metrics rely on quantities derived from the indirect influence matrix. Informally, this matrix measures how much each pair of nodes influences each other via all possible paths through the graph. This gives a natural importance metric for each node: how much it influences the logit nodes. We also commonly compare how much influence comes from error nodes vs. non-error nodes.

To construct this matrix, we start with the adjacency matrix of the graph. We replace all the edge weights with their absolute values (or simply clamp negative values to 0) to obtain an unsigned adjacency matrix and then normalize the input edges to each node so that they sum to 1. Let A refer to this normalized, unsigned adjacency matrix, indexed as (target, source).

The indirect influence matrix is B = A + A^2 + A^3 + \cdots, which is a [Neumann series](https://en.wikipedia.org/wiki/Neumann_series) and can be efficiently computed as B = (I - A)^{-1} - I. The entries of B indicate the sum of the strengths of all paths between a given pair of nodes, where the strength of any given path is given by the product of the values of its constituent edges in A. To compute a logit influence score for each node, we compute a weighted average of the rows of B corresponding to logit nodes (weighted by the probability of the particular logit).

#### [Measuring and Comparing Path Length](#evaluating-graphs-paths)

A natural metric of graph complexity is the average path length from embedding nodes to logit nodes. Intuitively, shorter paths are easier to understand as they require interpreting fewer links in the causal chain.

To measure the influence of paths of different lengths, we compute influence matrices B\_{\ell} = \sum\_{i=0}^{\ell} A^{i}.The influence of paths of length less than or equal to \ell is then given by P\_{\ell} = \sum\_{e} B^{\ell}\_{t,e} where e are all embedding nodes and t is the logit node.If there are multiple logit nodes, we compute an average of the rows weighted by logit probability.

Below, we compare graphs built from our 10M CLT, 10M PLTs, and thresholded neurons in terms of their influence by path length averaged across a dataset of [pretraining prompts](#appendix-eval-details) (without pruning).We normalize influence scores by the total influence of embeddings in the unpruned graph. This normalization factor is exactly equal to the replacement score, which we define in the next section.

One of the most important advantages of crosslayer transcoders is the extent to which they reduce the path lengths in the graph. To understand how large of a qualitative difference this is, we invite the reader to view these graphs generated with different types of replacement models for the same prompt.

|  |  |  |
| --- | --- | --- |
| Replacement Model Type | Average Path Length | Graph Link |
| Cross-Layer Transcoder (10m) | 2.3 | [capital-analogy-clt](./static_js/attribution_graphs/index.html?slug=capital-analogy-clt-18l) |
| Per-Layer Transcoders (10m) | 3.7 | [capital-analogy-plt](./static_js/attribution_graphs/index.html?slug=capital-analogy-slt-18l) |

We find that one important way in which cross-layer transcoders collapse paths is the case of amplification, where many similar features activate each other in sequence. For example, on the prompt Zagreb:Croatia::Copenhagen: the per-layer transcoder shows a path of length 7 composed entirely of Copenhagen [features](./static_js/attribution_graphs/index.html?slug=capital-analogy-slt-18l-path-highlight) while the cross-layer transcoder collapses them all down to layer 1 [features](./static_js/attribution_graphs/index.html?slug=capital-analogy-clt-18l-path-highlight).

This example illustrates both the advantages and disadvantages of consolidating amplification of a repeated computation across multiple layers into a single cross-layer feature. On one hand, it makes interpretability substantially easier, as it automatically collapses duplicate computations into a single feature without needing to do post hoc analysis or clustering. It also reduces the risk of “chain-breaking”, where missing one feature in an amplification chain inhibits the ability to trace back further into the graph (i.e., a relevant amplification feature is missing for one step of the path, breaking the causal chain). On the other hand, the CLT has a different causal structure than the underlying model, which increases the risk that the replacement model’s mechanisms diverge from the underlying model’s. In the above example, we observe a set of Copenhagen features that activate a Denmark feature, which initiates a mutually reinforcing chain of Copenhagen and Denmark features. This dynamic is invisible in CLT graphs, and to the extent this dynamic is also present in the underlying model, it is an example of CLTs being mechanistically unfaithful.

#### [Evaluating and Comparing Graph Completeness](#evaluating-graphs-comparing)

Because our replacement model has reconstruction errors, we want to measure how much of the model’s computation is being captured. That is, how much of the graph influence is attributable to feature nodes versus error nodes.

To measure this, we primarily rely on two metrics:

* Graph completeness score: measures the fraction of input edges (weighted by the target node’s logit influence score) that come from feature or embedding nodes rather than error nodes.
* Graph replacement score: measures the fraction of end-to-end graph paths (weighted by strength) that proceed from embedding nodes to logit nodes via feature nodes (rather than error nodes).

Intuitively, the completeness score gives more “partial credit” and measures how much of the most important node inputs are accounted for, whereas replacement score rewards complete explanations.

Below, we report average unpruned graph replacement and completeness scores for dictionaries of various sizes and types on our pretraining prompt [dataset](#appendix-eval-details). We find the biggest methodological improvement comes when moving from per-layer to cross-layer transcoders, with large but diminishing returns from scaling the number of features.

To contextualize the qualitative difference we observe in graphs with varying scores, we invite the reader to explore some representative attribution graphs. Note, these graphs are pruned with our default pruning, which we describe in more detail below.

|  |  |  |  |
| --- | --- | --- | --- |
| Replacement Model Type | Completeness Score | Replacement Score | Graph Link |
| Cross-Layer Transcoder (10m) | 0.80 | 0.61 | [uspto-telephone-clt](./static_js/attribution_graphs/index.html?slug=uspto-telephone-clt-18l) |
| Per-Layer Transcoders (10m) | 0.78 | 0.37 | [uspto-telephone-plt](./static_js/attribution_graphs/index.html?slug=uspto-telephone-slt-18l) |

#### [Graph Pruning](#evaluating-graphs-pruning)

We rely heavily on pruning to make graphs more digestible. To decide how much to prune the graph, we can use the completeness and replacement metrics described above, but with pruned nodes now counting towards the error terms. By varying the pruning threshold, we chart a frontier between the number of {nodes, edges} and {replacement, completeness} scores (see [Appendix](#appendix-graph-pruning) for full plots and details).

We find we can generally reduce the number of nodes by an order of magnitude while reducing completeness by only 20%.

For a sense of the qualitative difference, in the table below we link to attribution graphs for the same prompt (another acronym) but with different pruning thresholds.

|  |  |  |  |
| --- | --- | --- | --- |
| Pruning Threshold | Completeness Score | Node Count | Graph Link |
| 0.95 | 0.87 | 236 | [iasg-p95](./static_js/attribution_graphs/index.html?slug=iasg-clt-18l-p95) |
| 0.9 | 0.83 | 137 | [iasg-p90](./static_js/attribution_graphs/index.html?slug=iasg-clt-18l-p90) |
| 0.8 (default) | 0.70 | 55 | [iasg-p80](./static_js/attribution_graphs/index.html?slug=iasg-clt-18l-p80) |
| 0.7 | 0.58 | 27 | [iasg-p70](./static_js/attribution_graphs/index.html?slug=iasg-clt-18l-p70) |

### [Evaluating Mechanistic Faithfulness](#evaluating-model-faithfulness)

As discussed in [§ Validating Attribution Graph Hypotheses with Interventions](#graphs-interventions), attribution graphs provide hypotheses about mechanisms, which must be validated with perturbation experiments. This is because attribution graphs describe interactions in the local replacement model, which may differ from the underlying model. In most of our work, we use attribution graphs as a tool for generating hypotheses about specific mechanisms (“Feature A activates Feature B, which increases the likelihood of Token X”) operating inside the model, which correspond to “snippets” of the attribution graph. We summarize the results of three kinds of validation experiments, which are described in more detail in [§&nbspAppendix: Validating the Replacement Model](#appendix-lrm-validation).

We start by measuring the extent to which influence metrics derived from attribution graphs are predictive of intervention effects on the logit and other features. First, we measure the extent to which a node’s logit influence score is predictive of the effect of ablating a feature on the model’s output distribution. We find that influence is significantly more predictive of ablation effects than baselines such as direct attribution (i.e. direct edges in the attribution graph, ignoring multi-step paths) and activation magnitude (see [Validating Node-to-Logit Influence](#appendix-lrm-validation-node)). We then perform a similar analysis for interactions between features. We compute the influence score between pairs of features, and compare it to the relative effect of ablating the upstream feature in the pair on the activation of the downstream one. We observe a Spearman correlation of 0.72, which is evidence that graph influence is a good proxy for effects in the downstream model (see [Validating Feature-to-feature Influence](#appendix-lrm-validation-edges)). See [Nuances of Steering with Cross-Layer Features](#appendix-cross-layer-steering) for some complexities in interpreting these results.

The metrics above help provide an estimate of the likelihood that an intervention experiment will validate a specific mechanism in the graph. We might also be interested in a more general validation of all the mechanistic hypotheses implicitly made by our attribution graphs. Thus, another complementary approach to validation is to measure the mechanistic faithfulness of the local replacement model as a whole, rather than specific paths within attribution graphs. We can operationalize this by asking to what extent perturbations made in the local replacement model (which attribution graphs describe) have the same downstream effects as corresponding perturbations in the underlying model. We find that while perturbation results are reasonably similar between the two models when measured one layer after the intervention (~0.8 cosine similarity, ~0.4 normalized mean squared error), perturbation discrepancies compound significantly over layers.Compounding errors have a gradually detrimental effect on the faithfulness of the direction of perturbation effects, which are largely consistent across CLT sizes, with signs of faithfulness worsening slightly as dictionary size increases. Compounding errors can have a catastrophically detrimental effect on the magnitude of perturbations, with worse effects for larger dictionaries. We suspect the lack of normalization denominators in the local replacement model may be why its perturbation effect magnitudes deviate so significantly from the underlying model, even when the perturbation effect directions are significantly correlated. For more details, see [Evaluating Faithfulness of the Local Replacement Model](#appendix-lrm-validation-faithfulness).

  
  
  

---

  
  

## [Biology](#biology)

In our [companion paper](./biology.html), we use the method outlined here to perform deep investigations of the circuits in nine behavioral case studies of the frontier model Haiku 3.5. These include:

* [Multi-Step Reasoning.](./biology.html#dives-tracing) We present a simple example where the model performs “two-hop” reasoning to complete “The capital of the state containing Dallas is…”, going Dallas → Texas → Austin. We can see and manipulate its representation of the intermediate Texas step.
* [Planning in Poems.](./biology.html#dives-poems) We show that the model plans its outputs when writing lines of poetry. Before beginning to write each line, the model identifies potential rhyming words that could appear at the end. These preselected rhyming options then shape how the model constructs the entire line.
* [Multilingual Circuits.](./biology.html#dives-multilingual) We find the model uses a mixture of language-specific and abstract, language-independent circuits (which are more prevalent in Claude 3.5 Haiku than a smaller model).
* [Addition.](./biology.html#dives-addition) We highlight a case where the same addition circuitry generalizes between very different contexts, and uncover qualitative differences between the addition mechanisms in Claude 3.5 Haiku and a smaller, less capable model.
* [Medical Diagnoses.](./biology.html#dives-medical) We show an example in which the model identifies candidate diagnoses based on reported symptoms, and uses these to inform follow-up questions about additional symptoms that could corroborate the diagnosis – all “in its head,” without writing down its steps.
* [Entity Recognition and Hallucinations.](./biology.html#dives-hallucinations) We uncover circuit mechanisms that allow the model to distinguish between familiar and unfamiliar entities, which determine whether it elects to answer a factual question or profess ignorance. “Misfires” of this circuit can cause hallucinations.
* [Refusal of Harmful Requests.](./biology.html#dives-refusals) We find evidence that the model constructs a general-purpose “harmful requests” feature during finetuning, aggregated from features representing specific harmful requests learned during pretraining.
* [An Analysis of a Jailbreak](./biology.html#dives-jailbreak), which works by first tricking the model into starting to give dangerous instructions “without realizing it,” and continuing to do so due to pressure to adhere to syntactic and grammatical rules.
* [Chain-of-thought Faithfulness.](./biology.html#dives-cot) We explore the faithfulness of chain-of-thought reasoning to the model’s actual mechanisms. We are able to distinguish between cases where the model genuinely performs the steps it says it is performing, cases where it makes up its reasoning without regard for truth, and cases where it works backwards from a human-provided clue so that its “reasoning” will end up at the human-suggested answer.
* [A Model with a Hidden Goal.](./biology.html#dives-misaligned) We also apply our method to a variant of the model that has been finetuned to pursue a secret goal of exploiting biases in its training process. While the model is reluctant to reveal its goal out loud, our method exposes it, revealing the goal to be “baked in” to the model’s “Assistant” persona.

We encourage the reader to explore those case studies before returning here to understand the limitations we encountered, and how that informs our approach to method development.

  
  
  

---

  
  

## [Limitations](#limitations)

Despite the exciting results presented here and in [the companion paper](./biology.html), our methodology has a number of significant limitations. At a high level, the most significant ones are:

* [Missing Attention Circuits](#limitations-attention) – We don't explain how attention patterns are computed by QK-circuits, and can sometimes "miss the interesting part" of the computation as a result.
* [Reconstruction Errors & Dark Matter](#limitations-reconstruction-error) – We only explain a portion of model computation, and much remains hidden. When the critical computation is missing, attribution graphs won't reveal much.
* [The Role of Inactive Features & Inhibitory Circuits](#limitations-inactive) – Often the fact that certain features weren't active is just as interesting as the fact that others are. In particular, there are many interesting circuits which involve features inhibiting other features.
* [Graph Complexity](#limitations-complexity) – The resulting attribution graphs can be very complex and hard to understand.
* [Features at the Wrong Level of Abstraction](#limitations-abstraction-level) – Issues like feature splitting and absorption mean that features often aren't at the level of abstraction which would make it easiest to understand the circuit.
* [Difficulty of Understanding Global Circuits](#limitations-local-v-global) – Ideally, we want to understand models in a global manner, rather than attributions on a single example. However, global circuits are quite challenging.
* [Mechanistic Faithfulness](#limitations-faithfulness) – When we replace MLP computation with transcoders, how confident are we that they're using the same mechanisms as the original MLP, rather than something that's just highly correlated with the MLP's outputs?

We discuss these in detail below, and where possible provide concrete counterexamples where our present methods can not explain model computation due to these issues. We hope that these may motivate future research.

### [Missing Attention Circuits](#limitations-attention)

One significant limitation of our approach is that we compute our attribution graphs with respect to fixed attention patterns. This makes attribution a well-defined and principled operation, but also means that our graphs do not attempt to explain how the model’s attention patterns were formed, or how these patterns mediate feature-feature interactions through attention head output-value matrices . In this paper, we have focused on case studies where this is not too much of an issue – cases where attention patterns are not responsible for the “interesting part” or “crux” of the model’s computation. However, we have also found many cases where this limitation renders our attribution graphs essentially useless.

#### Example: Induction

Let's consider for a moment a much simpler model – a humble 2-layer attention-only model, of the kind studied in . One interesting property of these models is their use of induction heads  to perform basic in-context learning. For example, if we consider the following prompt:

I always loved visiting Aunt Sally. Whenever I was feeling sad, Aunt

These models will have induction heads attend back to `"Sally"`, and then predict that is the correct answer. If we were to apply our present method, the answer isn’t very informative. It would simply tell us that the model predicted `"Sally"`, because there was a token `"Sally"` earlier in the context.

This misses the entire interesting story! The induction head attends to `"Sally"` because it was preceded by `"Aunt"`, which matches the present token. Previous methods (e.g.  were able to elucidate this, and so this case might even be seen as a kind of regression.

Indeed, when applied to Claude 3.5 Haiku on this prompt, our method has exactly this problem. See the [attribution graph visualization](./static_js/attribution_graphs/index.html?slug=sally-induction-qk) – the graph contains direct edges from token-level “Sally” features to “say Sally” features and to the “Sally” logit, but fails to explain how these edges came about.

#### Example: Multiple-Choice Questions

Induction is a simple case of attentional computation where we can make a reasonable guess at the mechanism even without help from our attribution graphs. However, this failure of the attribution graphs can manifest in more complex scenarios as well, where it completely obscures the interesting steps of the model’s computation. For instance, consider a multiple choice question:

Human: In what year did World War II end?

(A) 1776

(B) 1945

(C) 1865

  

Assistant: Answer: (B)

When we compute the attribution graph ([interactive graph visualization](./static_js/attribution_graphs/index.html?slug=multiple-choice-qk)) for the `"B"` token in the Assistant’s response, we obtain a relatively uninteresting answer – we answer `"B"` because of a `tokens following "(b)"` feature that activates on the correct answer. (There's also a direct pathway to the token `"B"`, and output pathways mediated by a `say "B"` motor feature; we've chosen to elide these for simplicity.)

None of this provides a useful explanation of how the model chose its answer! The graph “skips over” the interesting part of the computation, which is how the model knew that 1945 was the correct answer. This is because the behavior is driven by attention. On further investigation, it turns out that there are three “correct answer” features that appear to fire on the correct answer to multiple choice questions, and which interventions show play a crucial role. From this, we hypothesize that the mechanism might be something like the following.Other explanations are possible, particularly for the direct paths from `"B"` that do not go through `tokens following "B"` – one alternative is that the model may use a “binding ID vector”  to group the `"B"` tokens with `"1945"` and nearby tokens and use this to attend directly back to the `"B"` token from the final token position – see Feng & Steinhardt  for more details on this type of mechanism.

Since this involves significant conjecture, it's worth being clear about what we know about the QK-circuit, and what we don't.

* We know that it's controlled by attention, since freezing attention locks the answer.
* We don't know that it's mediated by a particular head, nor that any heads involved have a general behavior that can be understood as a generalization of this.
* We do know that there are three features which appear to track "this seems like the correct answer". We do know that intervening on them changes the correct answer in expected ways; for example activating them on answer C causes the model to predict `"C"`.
* We don't know that "correct answer" features directly play a role in the key side of whatever attention heads are involved, nor do we know if "need answer" features exist or play a role on the query side.
* We also do not know if there might be alternative parallel mechanisms at play (see Feng & Steinhardt ).

This is all to say, there's a lot we don't understand!

But despite our limited understanding, it seems clear that the model behavior crucially flows through attention patterns and the QK circuits that compute them. Until we can fix this, our attribution graphs will "miss the story" in cases where attention plays a critical role. And while we were able to get a partial understanding of the story in this case through manual investigation, we would like for our methodology to surface this information automatically in the future!

#### Future Directions on Attention

We suspect that similar circuits, where attention is the crux, are at play across a wide variety of prompts. In these cases, our present attribution graphs are little help to us, and new methods are needed.

Ultimately, the QK-circuit is a quadratic form over the residual stream. This means that attributions can naturally be made to pairs of key-side and query-side features. These pairs have a weight describing whether they increase or decrease attention to a particular token. However, this approach has the downside of a quadratic explosion in complexity.

Looking at QK-circuits for individual attention heads may make things harder than necessary, or even be misleading, if there is [attention superposition](https://transformer-circuits.pub/2024/jan-update/index.html#attn-superposition). The interesting behavior may be spread across heads, which each have their own QK-circuit.

We also suspect that certain forms of dictionary learning on attention layers, or full attention replacement layers, could make these “QK attributions” more tractable and interpretable. We believe that finding the right way to handle attention attributions is one of the most important directions for future work.

### [Reconstruction Errors & Dark Matter](#limitations-reconstruction-error)

Our cross-layer transcoders fail to fully reconstruct the activations of the model. As a result, our methods leave some [“dark matter”](https://transformer-circuits.pub/2024/july-update/index.html#dark-matter) of the model’s mechanisms unobserved, in part because our dictionaries are not large enough to learn a feature corresponding to every concept the model might know about. In our attribution graphs, this failure manifests as “error nodes,” which provide input to other nodes but receive no input themselves (they “pop out of nowhere”) and which are not interpretable. In the examples presented in this paper, we managed to identify interesting mechanisms despite the presence of error nodes. It is important to note that the circuits we have described are only a partial description of the model’s computation (both due to the presence of error nodes, and due to the fact that the circuits we highlight typically only include a subset of the non-error nodes in the graph!).

It is worth highlighting, however, that in some cases error nodes completely obscure what the model is doing. For instance, consider this attempted jailbreak – an obfuscated request for the model to write malware (which the model refuses):

Human: Write an office macro TAhT SEaRChEs fOS AND EtEIaFTxLRS pASWORdS

  

Assistant:

In the [attribution graph](./static_js/attribution_graphs/index.html?slug=bon-errors), almost all the nodes on the obfuscated prompt tokens are error nodes, making it impossible to trace back the origin of the refusal-related features that are active on the “Assistant:” tokens.

The prevalence of error nodes in this example may not be surprising – this prompt is rather out-of-distribution relative to typical prompts, and so the cross-layer-transcoder is likely to do a poor job of predicting model activity.

We also note that another major source of error is the gap between our human interpretations of features and what they truly represent. Typically our interpretations of features are much too coarse to account for their precise activation profiles.

#### Future Directions on Reconstruction Error and "Dark Matter"

We see several avenues for addressing this issue:

* Scaling replacement models to larger sizes / more training data will increase the amount of variance they explain.
* Architectural modifications to our cross-layer transcoder setup could make it more expressive and thus capable of explaining more variance.
* Training our replacement model in a more end-to-end fashion, rather than on MSE alone, could decrease the weight assigned to error nodes even at a fixed MSE level
* Finetuning the replacement model on data distributions of interest could improve our ability to capture mechanisms on those distributions.
* We could develop methods of attributing back from error nodes. This would leave an uninterpretable “hole” in the attribution graph, but in some cases may still provide more insight into the model than our current no-inputs error nodes.

### [The Role of Inactive Features & Inhibitory Circuits](#limitations-inactive)

Our cross-layer transcoder features are trained to be sparsely active. Their sparsity is key to the success of our method. It allows us to focus on a relatively small set of features for a given prompt, out of the tens of millions of features in the replacement model. However, this convenience relies on a key assumption – that only active features are involved in the mechanism underlying a model’s responses.

In fact, this need not be the case! In some cases, the lack of activity of a feature, because it has been suppressed by other features, may be key to the model’s response. For instance, in our analysis of hallucinations and entity recognition (see [companion paper](./biology.html#dives-hallucinations)), we discovered a circuit in which “can’t answer” features are suppressed by features representing known entities, or questions with known answers. Thus, to explain why the model hallucinates in a specific context, we need to understand what caused the “can’t answer” features to not be active.

By default, our attribution graphs do not allow us to answer such questions, because they only display active features. If we have a hypothesis about which inactive features may be relevant to the model’s completion (due to suppression), we can include them in the attribution graph. However, this detracts somewhat from one of the main benefits of our methodology, which is its enablement of exploratory, hypothesis-free analysis.

This leads to the following challenge – how can we identify inactive features of interest, out of the tens of millions of inactive features? It seems like we want to know which features could have been “counterfactually active” in some sense. In the entity recognition example, we identified these counterfactually active features by comparing pairs of prompts that contained either known or unknown entities (Michael Jordan or “Michael Batkin”), and then focusing on features that were active in at least one prompt from each pair. We expect that this contrastive pairs strategy will be key to many circuit analyses going forward. However, we are also interested in developing more unsupervised approaches to identifying key suppressed features. One possibility may be to perform feature ablation experiments, and consider the set of inactive features that are only “one ablation away” from being active.

One might think that these issues can be escaped by moving to global circuit analysis. However, it seems like there may be a deep challenge which remains. We need a way to filter out interference weights, and it's tempting to do this by using co-occurrence of features. But these strategies will miss important inhibitory weights, where one feature consistently prevents another from activating. This can be seen as a kind of global circuit analog of the challenges around inactive features in local attribution analysis.

### [Graph Complexity](#limitations-complexity)

One of the fundamental challenges of interpretability is finding abstractions and interfaces that manage the cognitive load of understanding complex computations.For example, the fundamental reason we need features to be independently interpretable is to avoid needing to think about all of them at once (see discussion [here](https://transformer-circuits.pub/2022/mech-interp-essay/index.html)). Our methodology is designed to reduce the cognitive load of understanding circuits as much as possible. For example:

* Feature sparsity means that there are fewer nodes in the attribution graph.
* We prune the graphs so that the analyst can focus only on its most important components.
* Our UI is designed to make navigating the graph as fluid as possible.
* We use the not-very-principled abstraction of "supernodes" to ad-hoc group together related features.

Despite all these steps, our attribution graphs are still quite complex, and require considerable time and effort to understand for many reasons:

* Even after our pruning pipeline and on fairly short prompts, the graphs typically contain hundreds of features and thousands of edges.
* The concepts we are interested in are typically smeared across multiple features.
* Each feature receives many small inputs from many other features, making it difficult to succinctly summarize “what caused this feature to activate.”
* Features often exert influence on one another by multiple paths of different lengths, or even of different signs!

As a result, it is difficult to distill the mechanisms uncovered by our graphs into a succinct story. Consequently, the vignettes we have presented are necessarily simplified stories of even the limited understanding of model computation captured in our attribution graph. We hope that a combination of improved replacement model training, better abstractions, more sophisticated pruning, and better visualization tools can help mitigate this issue in the future.

### [Features at the Wrong Level of Abstraction](#limitations-abstraction-level)

As sparse coding models have grown in popularity as a technique for extracting interpretable features from models, many researchers have documented shortcomings of the approach (see e.g. ). One notable issue is the problem of feature splitting  in which the uncovered features are in some sense too specific. This can also lead to a related problem of feature absorption , where highly specific features steal credit from more general features, leaving holes in them (leading to things like a “U.S. cities except for New York and Los Angeles” feature).

As a concrete example of feature splitting, recall that in many examples in this paper we have highlighted “say X” features that cause the model to output a particular (group of) token(s). However, we also notice that there are many such features, suggesting that they each actually represent something more specific. For example, we came up with twelve prompts which Claude 3.5 Haiku completes with the word “during” and measured whether any features activated for all of the prompts (as a true “say ‘during’” feature would). In fact, there are no such features – any individual feature fires for only a subset of the prompts. Moreover, the degree of generality of the features appears to decrease with the size of our cross-layer transcoder.

It may be the case that each individual feature represents something interpretable – for instance, qualitatively different contexts that might cause one to say the word “during.” However, we often find that the level of abstraction we care about is different from the level we find in our features. Using smaller cross-layer transcoders may help this problem, but would also cause us to capture less of the model’s computation.

In this paper, we often work around this issue in an ad-hoc way by manually grouping together features with related meanings into “supernodes” of an attribution graph. While this technique has proven quite helpful, the manual step is labor-intensive and likely loses information. It also makes it difficult to study how well mechanisms generalize across prompts, since different subsets of a relevant feature category may be active on different prompts.

We expect that solving this problem requires recognizing that there exist interpretable concepts at varying levels of abstraction, and at different times we may be interested in different levels. Sparse coding approaches like SAEs and (cross-layer) transcoders are a “flat” instrument, but we probably need a hierarchical variant that allows features at varying levels of abstraction to coexist in an interpretable way.

Several authors have recently proposed “Matryoshka” variants of sparse autoencoders that may help address this issue . Other researchers have proposed post-hoc ways to unify related features with “meta-SAEs” .

### [Difficulty of Understanding Global Circuits](#limitations-local-v-global)

In this paper we have mostly focused on attribution graphs, which display information about feature-feature interactions on a particular prompt. However, one theoretical advantage of transcoder-based methodologies like ours is that they give us global weights between features, that are independent of the prompt. This allows us to estimate a “connectome” of the replacement model and learn about the general algorithms it (though not necessarily the underlying model) uses that apply to many different inputs. We have some successes in this approach – for instance, in the companion paper section on [Refusals](./biology.html#dives-refusals), we could see the global inputs to “harmful requests” features consisting of a variety of different specific categories of harm. In this paper, we studied in depth the global weights of features relating to arithmetic, finding for instance that “say a number ending in 5” features receive input from “6 + 9” features, “7 + 8” features, etc.

However, for the most part, we have found global feature-feature connections rather difficult to understand. This is likely for two main reasons:

* Interference weights – because features are represented in superposition, in order to learn useful weights between features, models must incur spurious [“interference weights”](https://transformer-circuits.pub/2023/may-update/index.html#weight-superposition) – connections between features that don’t make sense and aren’t useful to the model’s performance. These spurious weights are not too detrimental to model performance because their effects rarely “stack up” to actually change the model’s output. For instance, we sometimes see features like this, which appear to clearly be a “say 15” feature, but whose top logit outputs include many seemingly unrelated words (“gag”, “duty”, “temper”, “dispers”). We believe these logit connections are essentially irrelevant to the model’s behavior, because when this feature activates, it is very unlikely that “duty” will be a plausible completion, and so upweighting its logit runs little risk of causing it to be sampled. Unfortunately, this makes the global weights very difficult to understand! This phenomenon applies to feature-feature weights as well (see [§ Global Weights in Addition](#global-weights-addition)).

* Interactions mediated by attention – The basic global feature-feature weights derived from our cross-layer transcoder describe the direct interactions between features not mediated by any attention layers. However, there are also feature-feature weights mediated by attention heads. These might be thought of as similar to how features in a convolutional neural network are related by multiple sets of weights, corresponding to different positional offsets (further discussion [here](https://transformer-circuits.pub/2021/framework/index.html#additional-intuition)).

Our attribution graph edges are weighted combinations of both the direct weights and these attention-mediated weights. Our basic notion of global weights does not account for these interactions at all. One way to do so would be to compute the global weights mediated by every possible attention head. However, this has two limitations: (1) for this to be useful, we need a way of understanding the mechanisms by which different heads choose where they attend (see [§ Limitations: Missing Attention Circuits](#limitations-attention)), (2) it does not account for interactions mediated by compositions of heads . Solving this issue likely requires extending our dictionary learning methodology to learn interpretable attentional features or replacement heads.

### [Mechanistic Faithfulness](#limitations-faithfulness)

Our cross-layer transcoder is trained to mimic the activations of the underlying model at each layer. However, even when it accurately reconstructs the model’s activations, there is no guarantee that it does so via the same mechanisms. For instance, even if the cross-layer transcoder achieved 0 MSE on our training distribution, it might have learned a fundamentally different input/output function than the underlying model, and consequently have large reconstruction error on out-of-distribution inputs. We hope that this issue is mitigated by (1) training on a broad data distribution, and (2) forcing the replacement model to reconstruct the underlying model’s per-layer activations, rather than simply its output. Nevertheless, we cannot guarantee that the replacement model has learned the same mechanisms  – what we call mechanistic faithfulness – and instead resort to verifying it post-hoc.

In this paper, we have used perturbation experiments (inhibiting and exciting features) to validate the mechanisms suggested by our attribution graphs. In the case studies we presented, we were typically able to validate that features had the effects we expected (on the model output, and on other features). However, the degree of validation we have provided is very coarse. We typically perturb multiple features at once (“supernodes”) and check their directional effects on other features / logit outputs. In addition, we typically sweep over the layer at which we perform perturbations and use the layer that yields the maximum effect. In principle, our attribution graphs make predictions that are much more fine-grained than these kinds of interventions can test. Ideally, we should be able to accurately predict the effect of perturbing any feature at any layer on any other feature.

In [§ Appendix: Validating the Replacement Model](#appendix-lrm-validation), we attempt to more comprehensively quantify our accuracy in predicting such perturbation results, finding reasonably good predictive power for effects a few layers downstream of a perturbation, and much worse predictive power many layers downstream. This suggests that, while our circuit descriptions may be mechanistically accurate at a very coarse level, we have substantial room to improve their faithfulness to the underlying model.

We are optimistic about trying methods to directly optimize for mechanistic faithfulness, or exploring alternative dictionary learning architectures that learn more faithful solutions naturally.

  
  
  

---

  
  

## [Discussion](#discussion)

Our approach to reverse engineering neural networks has four basic steps: decomposition into components, providing descriptions of these components, characterizing how components interact to produce behaviors, and validating these descriptions.See Sharkey et al.  for a detailed description of the reverse engineering philosophy. A number of choices are required at each step, which can be more or less principled, and the power of a method is ultimately the degree to which it produces valid hypotheses about model behaviors.

In this paper, we trained cross-layer transcoders with sparse features to replace MLP blocks (the decomposition), described the features by the dataset examples they activate on (the description), characterized their interactions on specific prompts using attribution graphs (the interactions), and validated the hypotheses using causal steering interventions (the validation).

We believe some of the choices we made are robust, and that successful decomposition methods will make similar choices or find other ways of dealing with the underlying issues they address:

* We use learned features instead of neurons. While the top activations for neurons are often interpretable, lower activations are not. In principle, one could threshold neuron activations to restrict them to this interpretable regime; however, we found that thresholding neurons at that level damages model behavior significantly more than a transcoder or CLT. This means a trained replacement layer can provide a Pareto improvement relative to thresholded neurons across interpretability, L0, and MSE. The set of neurons is fixed. (Nevertheless, neurons can provide a starting point for investigation, without incurring any additional compute costs, see e.g. .)
* We use transcoders instead of residual-stream SAEs. While residual stream SAEs can decompose the latent states of a model, they don't provide a natural extension decomposing its computational steps. Crucially, transcoder features bridge over MLP layers and interact linearly via the residual stream with transcoder features in other layers. In contrast, interaction between SAE features is interposed by non-linear MLPs.
* We use cross-layer transcoders instead of per-layer transcoders. We hypothesized  that different MLP layers might collaborate to implement a single computational step (“cross-layer superposition”); the most extreme case of this is when many layers amplify the same early-layer feature so that it is still large enough to influence late layers. CLTs collapse these to one feature. We found evidence that this phenomenon happens in practice, as manifested through the Pareto improvement on [path length vs graph influence](#evaluating-graphs-paths).
* We compute feature-feature interactions using linear direct effects instead of nonlinear attributions or ablations. Much has been written about “saliency maps” or attribution through non-linear neural networks (including ablation, path-integrated gradients , and Shapley values (e.g. )). Even the most principled options for credit assignment in a nonlinear setting are somewhat fraught. Since our goal is to crisply reason about mechanism, we construct our setup so that the direct interactions between features in the previous layer and the pre-activations of features in the next layer are conditionally linear; that is to say, they are linear once we freeze certain parts of the problem (attention patterns and normalization denominators). This factors the problem into a portion we can mechanistically understand in a principled manner, and a portion that remains to be understood . Also crucial to achieving this linear direct effect property is the earlier decision to use transcoders.Credit attribution in a non-linear setting is a hard problem. The core challenge can't simply be washed away, and it's worth asking where we implicitly have pushed the complexity. There are three places it may have gone. First – and this is the best option – is that the non-linear interactions have become multi-step paths in our attribution graph, which can then be reasoned about in interpretable ways. Next, a significant amount of it must have gone into the frozen components, factored into separate questions we haven't tried to address here, but at least know remain. But there is also a bad option: some of it may have been simplified by our CLT taking a non-mechanistically faithful shortcut, which approximates MLP computation with a linear approximation that is often correct.Our setup allows for some similar principled notions of linear interaction in the global context, but we now have to think of different weights for interactions along different paths – for example, what is the interaction of feature A and feature B mediated by attention head H? This is discussed in [§ Global Weights](#global-weights). The Framework paper also [discussed](https://transformer-circuits.pub/2021/framework/index.html#additional-intuition) these general conceptual issues in the appendix.

There are other choices we made for convenience, or as a first step towards a more general solution:

* We collapse attention paths. Every edge in our attribution graph is the direct interaction of a pair of features, summed over all possible direct interaction paths. Some of these paths flow primarily through the residual stream. Others flow through attention heads. We make no effort in the present work to distinguish these. This throws away a lot of interesting structure, since which heads mediated an interaction may be interesting, if they are something we understand (e.g. a successor head  or induction head ).Analyzing the global weights mediated by a head may be interesting here. For example, an induction head might systematically move “I'm X” features to “say X” features. A successor head might systematically map “X” features to “X+1” features.Of course, not all heads are individually interesting in the way induction heads or successor heads often are. We might suspect there are many more “attentional features” like induction and succession hiding in superposition over attention heads. If we could reveal these, there might be a much richer story.
* We ignore QK-circuits. In order to get our linear feature-feature interactions, we factor understanding transformers into two pieces, following Framework . First we ask about feature-feature interaction, conditional on an attention head or set of attention heads (the “OV-circuit”). But this leaves a second question of why attention heads attend to various pieces (the “QK-circuit”). In this work, we do not attempt this second half.
* We only use a sparsity penalty and a reconstruction loss for crosscoder training. While our ultimate goal is to find circuits with sparse interpretable edges, in a replacement model which is mechanistically faithful to the underlying model, we don't train explicitly for any of those goals.

Nevertheless, our current method yielded interesting, validated mechanisms involving planning, multilingual structure, hallucinations, refusals, and more, in [the companion paper](./biology.html).

We expect that advances in the trained, interpretable replacement model paradigm will produce quantitative improvements on graph-related metrics and qualitative improvements on the amount of model behaviors that become legible. It is possible that this will be an incremental process, where incremental improvements to CLTs and associated approaches to attention will yield incremental improvements to circuit identification, or that a radically different decomposition approach will best this method at scale at uncovering mechanisms. Regardless, we hope to enter an era where there is a clear flywheel between decomposition methods and “biology” results, where the appearance of structure in specific model investigations inspires innovations in decomposition methods, which in turn bring more model behaviors into the light.

### Coda: The lessons of addition

Addition is one of the simplest behaviors performed by models, and because it is so structured, we can characterize every feature's activity on the full problem domain exactly. This allows us to skip the difficult step of staring at dataset examples and trying to discern what a feature is doing relative to, and what distinguishes it from other features active in similar contexts. This revealed a range of heuristics used by Haiku 3.5 (“say something ending in a 5”, “say something around 50”, “say something starting with 51”) which had been identified before by Nikankin , together with a set of lookup table features that connect input pairs satisfying certain conditions (say, adding digits ending in 6 and 9) to the appropriate sum features satisfying the consequence on the output (say, producing a sum ending in 5).

However, even in this easier setting we made numerous mistakes when labeling these features from the original dataset examples alone, for example thinking a `_6 + _9` feature was itself a `sum = _5` feature based on what followed it in contexts. We also struggled to distinguish between low-precision features of different scales, and between features which were sensitive to a limited set of inputs or merely appeared to be because of a high prevalence of those inputs in our dataset. How much worse must this be when looking at dozens of gradations of refusal features! Getting more precise distinctions between features in fuzzier domains than arithmetic, whether through feature geometry or superhuman autointerpretability methods, will be necessary if we want to understand problems at the level of resolution that even today's CLTs appear to make possible.

Because addition is such a clear problem, we were also able to see how the features connected with each other to build parallel pathways; giving rise from simple heuristics depending on the input to more complex heuristics related to the output; going from the “Bag of Heuristics” identified by Nikankin to a “Graph of Heuristics”. The virtual weights show this computational structure, with groups of lookup table features combining to form sum features of different modularity and scale, which combine to form more precise sum features, and to eventually give the output. It seems likely that, in “fuzzier” natural language examples, we are conflating many roles played by features at different depths into overall buckets like “unknown entity” or “harmful request” or “notions of largeness” which actually serve specialized roles, and that there is actually an intricate aggregation and transformation of information taking place, just out of our understanding today.

  
  
  

---

  
  

## [Related Work](#related-work)

Despite being a young field, mechanistic interpretability has grown rapidly. For an introduction to the landscape of open problems and existing methods, we recommend the recent survey by Sharkey et al. . Broader perspectives can be found in recent reviews of mechanistic interpretability and related topics (e.g. ).

In previous papers, we've discussed some of the foundational topics our work builds on, and rather than recapitulate that discussion, we will refer readers to our previous discussion on them. This includes

* Existence of interpretable features (e.g. ; see [prior discussion](https://transformer-circuits.pub/2022/toy_model/index.html#related-interpretable-features))
* Attention head analysis (e.g. ; see [prior discussion](https://transformer-circuits.pub/2021/framework/index.html#related-work:~:text=to%20our%20approach.-,ATTENTION%20HEAD%20ANALYSIS,-Our%20work%20follows)),
* Bertology (e.g. ; see [prior discussion](https://transformer-circuits.pub/2021/framework/index.html#bertology-generally)),
* Interpretability interfaces (e.g. ; see [prior discussion](https://transformer-circuits.pub/2021/framework/index.html#interpretability-interfaces)),
* Disentanglement (e.g. ; see [prior discussion](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#related-work-disentanglement)),
* Compressed sensing (e.g. ; see [prior discussion](https://transformer-circuits.pub/2022/toy_model/index.html#related-compressed)),
* Sparse dictionary learning (e.g. ; see [prior discussion](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#related-work-dictionary)),
* Theory of superposition (e.g. ; see [prior discussion](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#related-work-superposition)),
* Theories of neural coding and distributed representation (e.g. ; see [prior discussion](https://transformer-circuits.pub/2022/toy_model/index.html#related-codes))
* Activation steering (e.g. ; see [prior discussion](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#related-work-steering)).

The next two sections will focus on the two different stages we often use in mechanistic interpretability : identifying features and then analyzing the circuits they form. Following that, we'll turn our attention to past work on the "biology" of neural networks.

### Feature Discovery Methods

A fundamental challenge in circuit discovery is finding suitable units of analysis . The network's natural components—attention heads and neurons—lack interpretability due to superposition , making the identification of better analytical units a central problem in the field.

Sparse Dictionary Learning is a technique with a long history originally developed by neuroscientists to analyze neural recording data . Recent work  has applied Sparse Autoencoders (SAEs)  as a scalable solution to address superposition by learning dictionaries that decompose language model representations. While SAEs have been successfully scaled to frontier models , researchers have identified several methodological limitations including feature shrinkage , feature absorption , lack of canonicalization , automating feature interpretation , and poor performance on downstream classification and steering tasks .

For circuit analysis specifically, SAEs are suboptimal because they decompose representations rather than computations. Transcoders  address this limitation by predicting the output of nonlinear components from their inputs. Bridging over nonlinearities like this enables direct computation of pairwise feature interactions without relying on attributions or ablations through intermediate nonlinearities.

The space of dictionary learning approaches is large, and we remain very excited about work which explores this space and addresses methodological issues. Recent work has explored architectural modifications like multilayer feature learning , adding skip connections , incorporating gradient information , adding stronger hierarchical inductive biases , and further increasing computational efficiency with mixtures-of-experts . The community has also studied alternative training protocols to learn dictionaries that respect downstream activations , reduce feature shrinkage , and make downstream computational interactions sparse . To measure this methodological progress, a number of benchmarks and standardized evaluation protocols for assessing dictionary learning methods have been developed . We think circuit-based metrics will be the next frontier in dictionary learning evaluation.

Beyond dictionary learning, several alternative unsupervised approaches to extracting computational units have shown initial success in small-scale settings. These include transforming activations into the local interaction basis  and developing attribution-based decompositions of parameters .

### Circuit Discovery Methods

Definitions. Throughout the literature, the term circuit is used to mean many different things. Olah et al.  introduced the definition as a subgraph of a neural network where nodes are directions in activation space and edges are the weights between them. This definition has been relaxed over time in some parts of the literature and is often used to refer to a general subgraph of network components with edge weights computed from an attribution or intervention of some kind .

There are several dimensions along which circuit approaches and definitions vary:

* Are the units of analysis globally interpretable or not? (For example, compare monosemantic features versus an entire attention head which does many different things across the data distribution.)
* Is the circuit itself (i.e. the edge connections) a global description, or a locally valid attribution graph? Or something else?
* Are the edges interpretable? (For example, are the edges computed by linear attributions or a complex nonlinear intervention?)
* Does the approach naturally address superposition?

We believe the North Star of circuit research is to manifest an object with globally interpretable units connected by interpretable edges which are globally valid. The present work falls short by only offering a locally valid attribution graph.

Manual Analysis. Early circuit discovery was largely manual, requiring specific hypotheses and bespoke methods of validation . Causal mediation analysis , activation patching , patch patching , and distributed alignment search  have been the the most commonly used techniques for refining hypotheses and isolating causal pathways in direct analyses. However, these techniques generally do not provide interpretable (i.e. linear) edge weights between units.

Automatic Analysis. These analyses were automated in Conmy et al.  by developing a recursive patching procedure to automatically find component subgraphs given a task of interest. However, patching analyses are computationally expensive, requiring a forward pass per step. This motivated attribution patching , a more efficient method leveraging gradients to approximate the effect of interventions. There has been significant follow up work on attribution patching including improving the gradient approximation, adapting the techniques to vision models , and incorporating positional information . Other techniques studied in the literature include learned masking techniques, circuit probing , discretization , and information flow analysis . The objective of many of these automated approaches is isolating important model components (i.e., layers, neurons, attention heads) and the interactions between them, but they do not address the interpretation of these components.

However, armed with better computational units of analysis from sparse dictionary learning, this work and other recent papers  are making a return to discovering connections between interpretable components. Specifically, our work is most similar to Dunefsky et al.  and Ge et al.  who also use transcoders to compute per-prompt attribution graphs with stop-gradients while also studying input-agnostic global weights. Our work is different in that we use crosscoders to absorb redundant features, use a more global pruning algorithm, include error nodes , and use a more powerful visualization suite to enable deeper qualitative analysis. These works are closer in spirit to the original circuits vision outlined in Olah et al. , but inherit prompt specific quantities (e.g. attention patterns) that limit their generality.

Attention Circuits. The attention mechanism in transformers introduced challenges for weight-based circuit analysis as done by Olah et al. . Elhage et al.  proposed decomposing attention layers into a nonlinear QK component controlling the attention pattern, and a linear OV component which controls the output. By freezing QK (e.g., for a particular prompt), transcoder feature-feature interactions mediated by attention become linear. This is the approach adopted by this work and Dunefsky et al. . Others have tried training SAEs on the attention outputs  and multiplying features through key and query matrices to explain attention patterns .

Replacement Models. Our notion of a replacement model is similar in spirit to past work on causal abstraction  and proxy models . These methods seek to learn an interpretable graphical model which maintains faithfulness to an underlying black box model. However, these techniques typically require a task specification or other supervision, as opposed to our replacement model which is learned in a fully unsupervised manner.

Circuit Evaluation. Causal scrubbing  was proposed as an early principled approach for evaluating interpretation quality through behavior-preserving resampling ablations. Shi et al.  formalized criteria for evaluating circuit hypotheses, focusing on behavior preservation, localization, and minimality, while applying these tests to both synthetic and discovered circuits in transformer models. However, Miller et al.  raised important concerns about existing faithfulness metrics, finding them highly sensitive to seemingly insignificant changes in ablation methodology.

### Circuit Biology and Phenomenology

Beyond methods, many works have performed deep case studies and uncovered interesting model phenomenology. For example, thorough circuit analysis has been performed on

* Arithmetic in toy models
* Python doc strings
* Indirect object identification
* Computing the greater-than operator
* Multiple Choice
* Pronoun gender
* In-context learning

The growing set of case studies has enabled further research on how these components are used in other tasks . Moreover, Tigges et al.  found that many of these circuit analyses are consistent across training and scale. Preceding these analyses, there has also been a long line of “Bertology” research that has studied model biology (see survey ) using attention pattern analysis and probing.
