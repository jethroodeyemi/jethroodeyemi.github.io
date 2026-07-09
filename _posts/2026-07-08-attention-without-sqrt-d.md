---
title: "Can Attention Work Without the sqrt(d) Scaling?"
date: 2026-07-08
permalink: /posts/2026/07/attention-without-sqrt-d/
tags:
  - machine learning
  - transformers
  - attention
  - deep learning
  - softmax
  - optimization
---

The division by $\sqrt{d_k}$ inside the attention softmax might be the most-copied magic constant in deep learning. It appears in essentially every Transformer implementation ever written, and yet the original paper justifies it in a single footnote, prefaced in the main text by a hedge: *"We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients"* [1]. A suspicion, not a theorem. So I wanted to work through it properly: why is the $\sqrt{d_k}$ there, what actually breaks if you delete it, and — since several production models have effectively deleted it — what does it take for attention to work without it? I ran a few quick experiments along the way, and every number below is measured, not hypothetical.

## The Equation

Scaled dot-product attention is defined as

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
$$

where $Q \in \mathbb{R}^{n \times d_k}$ are the queries, $K \in \mathbb{R}^{n \times d_k}$ the keys, $V \in \mathbb{R}^{n \times d_v}$ the values, $d_k$ the head dimension, and the softmax is applied row-wise. Because each row normalizes independently, everything can be analyzed for a single query $q \in \mathbb{R}^{d_k}$ against keys $k_1, \dots, k_n$ and values $v_1, \dots, v_n$:

$$
z_i = \frac{q^{\top} k_i}{\sqrt{d_k}}, \qquad p = \mathrm{softmax}(z), \qquad o = \sum_{i=1}^{n} p_i v_i
$$

The question is why the logit $z_i$ carries that divisor.

## The Variance Argument

Assume the components of $q$ and $k$ are independent random variables with mean 0 and variance 1, and that $q$ and $k$ are independent of each other — a reasonable model at random initialization. The dot product $q^{\top}k = \sum_i q_i k_i$ then has mean 0, and its variance expands as

$$
\mathrm{Var}(q^{\top}k) = \sum_{i,j} \mathbb{E}[q_i q_j]\,\mathbb{E}[k_i k_j] = d_k \cdot 1 + (d_k^2 - d_k)\cdot 0 = d_k
$$

because the $d_k$ diagonal terms each contribute $\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$ while every cross term factors into products of zero means. The standard deviation of a raw attention logit is therefore $\sqrt{d_k}$, and dividing by $\sqrt{d_k}$ restores unit variance. This is exactly the footnote in Vaswani et al. [1], and it checks out numerically: sampling 100,000 Gaussian query–key pairs per setting, I measured an empirical standard deviation of 7.9774 at $d_k = 64$ against the predicted 8, and the match stays within 0.5% at every $d_k$ from 4 to 1024.

The cleanest way to read this is as a temperature. For temperature $T$, softmax sharpness is governed by the ratio of logit scale to $T$: as $T \to \infty$ the distribution flattens to uniform, and as $T \to 0$ it collapses to argmax. The $\sqrt{d_k}$ divisor is nothing but a fixed choice $T = \sqrt{d_k}$, calibrated so the logits it sees have unit scale — at initialization.

## What Breaks Without It: Saturation

Delete the divisor and the logit standard deviation becomes $\sigma = \sqrt{d_k}$: 8 at $d_k = 64$, about 22.6 at $d_k = 512$. For $n$ roughly Gaussian logits, the expected maximum grows like $\sigma\sqrt{2\ln n}$, so the gap between the top logit and a typical one grows linearly in $\sigma$ — and softmax exponentiates that gap.

I measured this directly with $n = 256$ keys, averaged over 2000 trials. With scaling, the mean maximum attention weight is essentially constant in $d_k$: 0.0434 at $d_k = 64$ and 0.0431 at $d_k = 1024$, with mean entropy pinned around 5.05 nats, just below the uniform ceiling of $\ln 256 = 5.545$. Without scaling, the maximum weight climbs from 0.170 at $d_k = 4$ to 0.749 at $d_k = 64$ and 0.936 at $d_k = 1024$, while the entropy collapses from 3.97 to 0.75 to 0.159 nats. At production head sizes, unscaled attention puts roughly three quarters or more of its mass on a single key.

![Softmax saturation with and without the sqrt(d_k) scaling](https://jethroodeyemi.github.io/files/2026_07_08_post/fig1_saturation.png)
_**[Figure 1]** With scaling (blue), max attention weight and entropy are flat in $d_k$. Without it (green), one key soaks up nearly all the attention as $d_k$ grows and entropy collapses toward zero._

## The Gradient Argument

Saturation matters because of what it does to the backward pass. Writing $p_i = e^{z_i}/\sum_l e^{z_l}$, the softmax Jacobian is

$$
J = \frac{\partial p}{\partial z} = \operatorname{diag}(p) - p\,p^{\top}, \qquad \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

where $\delta_{ij}$ is the Kronecker delta. Every entry contains a factor $p_i(1-p_i)$ or $p_i p_j$, so every entry is bounded by $1/4$ and — the important part — vanishes as $p$ approaches one-hot: if the top weight is $1 - \varepsilon$, every entry of $J$ is $O(\varepsilon)$.

Measuring the mean Frobenius norm of $J$ over the same 2000 trials produced a curve I did not fully expect. Scaled attention sits flat at about 0.099 for every $d_k$. Unscaled attention is non-monotonic: 0.187 at $d_k = 4$, peaking at 0.288 at $d_k = 16$, then falling through 0.231 and 0.142 before crossing below the scaled curve between $d_k = 256$ and 1024 and reaching 0.083 with no sign of stopping. A mildly peaked softmax actually has a larger Jacobian than a near-uniform one — the norm is small at both extremes — so the scaling buys you *constancy* of gradient flow, not maximality.

![Softmax Jacobian norm versus head dimension](https://jethroodeyemi.github.io/files/2026_07_08_post/fig2_jacobian.png)
_**[Figure 2]** The unscaled Jacobian norm (green) peaks at $d_k = 16$, then goes into free fall as saturation sets in, crossing below the flat scaled curve (blue) between $d_k = 256$ and 1024._

There is a crucial nuance here: saturation does not kill the whole layer. The output $o = \sum_i p_i v_i$ is *linear* in the values, so with upstream gradient $g = \partial L / \partial o$, the value path never touches $J$ at all:

$$
\frac{\partial L}{\partial v_i} = p_i\, g
$$

which stays healthy for the selected key even at full saturation. By contrast, every gradient reaching $W_Q$ and $W_K$ is a linear image of $Ja$ (with $a_i = v_i^{\top}g$), and at exact one-hot that image is exactly zero. Unscaled attention at large $d_k$ therefore degenerates into a hard gather with frozen, random routing but a still-learnable payload through $W_V$ — which is precisely why the failure mode is a silent quality loss rather than a crash.

## A Training Demo

To see the dynamics rather than just the statics, I trained a single-head attention model with $d_k = 256$ on an associative recall task: 16 key–value pairs per sequence over 32 key tokens and 32 value tokens, Adam at learning rate 3e-4, with identical initializations and identical data streams for the scaled and unscaled variants.

The first honest surprise: at step 0 the Frobenius norm of the $W_Q$ gradient was 0.1085 scaled versus **1.0610 unscaled** — about 9.8 times *larger*, not smaller. Removing the $1/\sqrt{d_k}$ also multiplies $\partial z / \partial W_Q$ by $\sqrt{256} = 16$, and with only 16 keys per sequence, saturation claws back just a factor of about 0.61. So the textbook story "gradients vanish at initialization" is wrong for small tasks; the accurate story is ill-conditioned, saturation-throttled *dynamics*.

And that is what the loss curves show. Both models reached 100.0% eval accuracy on 2000 held-out sequences (chance is 3.125%) — the unscaled model does learn. But the scaled run crossed smoothed loss 1.0 at step 224 versus 424 for unscaled, and 0.1 at step 436 versus 822; the unscaled run stalled visibly on a plateau near loss 0.3 between roughly steps 530 and 820, and finished at 0.01374 versus 0.00562 — about 2.4 times higher. Dropping the scaling here cost roughly 2x slower, bumpier optimization, not outright failure.

![Training curves with and without scaling](https://jethroodeyemi.github.io/files/2026_07_08_post/fig3_training.png)
_**[Figure 3]** Identical init, identical data. The unscaled run (green) learns the task but takes roughly twice as long to every loss milestone, with a mid-training stall around loss 0.3._

## So Can It Work Without sqrt(d)? Yes — in Several Senses

### Move it into the initialization

Scale $W_Q$ and $W_K$ by $d_k^{-1/4}$ at initialization and drop the divisor. The logits are then

$$
\tilde q^{\top}\tilde k = \left(d_k^{-1/4} W_Q x\right)^{\top}\!\left(d_k^{-1/4} W_K x'\right) = \frac{q^{\top}k}{\sqrt{d_k}}
$$

identical to scaled attention at step 0. But this is a reparameterization, and optimizers are not reparameterization-invariant: under SGD it is equivalent to a $\sqrt{d_k}$-times-larger learning rate on the query and key projections, Adam breaks the correspondence entirely, and weight norms drift away from their initialization during training. Identical at step 0 is not identical at step 10,000.

### Normalize q and k

QKNorm [2] L2-normalizes queries and keys and replaces $\sqrt{d_k}$ with a learned scalar. Swin Transformer V2 [3] goes further with cosine attention divided by a learned per-head temperature — there is no $\sqrt{d}$ anywhere in it, and it trains at ~3B parameters, which is as clean an existence proof as you could ask for. The ViT-22B paper [4] is the flip side: *even with* the $1/\sqrt{d}$ scaling, scaling ViT up led to divergence at around 8B parameters, caused by "extremely large values in attention logits" producing near-one-hot, near-zero-entropy attention — which they fixed by applying LayerNorm to queries and keys before training the 22B model. Zhai et al. [5] name this failure "attention entropy collapse" and prove the entropy lower bound decays exponentially in the spectral norm of the attention logits. QK normalization is now standard in OLMo 2 [6], Chameleon [7] — whose 7B model diverged about 20% into an epoch without it — and Gemma 3 [8].

### Learn the temperature

If $\sqrt{d_k}$ is just a fixed temperature, make it a parameter: QKNorm's learned scalar [2] and Swin V2's per-head, per-layer temperature [3] both do this. YaRN [9] shows the right temperature is not even constant — it usefully depends on context length when extending the context window.

### Use 1/d instead (muP)

The variance argument assumes $q$ and $k$ are independent — true at initialization, destroyed by training, since queries learn to align with their keys. Once coordinates correlate, $q^{\top}k$ scales like $d_k$ by the law of large numbers, not $\sqrt{d_k}$ by the central limit theorem. Yang and Hu's muP analysis [10] concludes that $1/\sqrt{d_k}$ is the *wrong* width scaling for trained networks and prescribes $q^{\top}k / d_k$, which is one ingredient of zero-shot hyperparameter transfer across model widths.

### Cap the logits

The blunt instrument: Gemma 2 [11] applies a tanh soft-cap to attention logits at 50, and Grok-1's released code [12] caps them at 30. A hard bound on logit magnitude makes the precise divisor much less critical.

### Keep the head small

At $d_k = 1$ the two variants are literally identical. More generally, unscaled logits with $\sigma = \sqrt{d_k}$ stay below the softmax condensation threshold $\sqrt{2\ln n}$ whenever $d_k \lesssim 2\ln n$ — about 11 for $n = 256$ keys. At $d_k = 4$, saturation would require attending over fewer than about 7 keys. Tiny heads simply never enter the regime the scaling protects against.

## Takeaway

The $\sqrt{d_k}$ is a sensible default temperature, calibrated for independent queries and keys at initialization — not a law of nature. What breaks without it is not attention itself but control of the attention-logit scale, and every working alternative restores that control by other means: normalization, a learned temperature, a hard cap, or a stronger $1/d$ exponent. The interesting question is not whether to divide by $\sqrt{d_k}$. It is which mechanism owns the attention temperature in your model — and whether it still holds after the weights have drifted far from wherever they started.

## References

[1] Vaswani et al. "Attention Is All You Need" 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

[2] Henry et al. "Query-Key Normalization for Transformers" 2020. [https://arxiv.org/abs/2010.04245](https://arxiv.org/abs/2010.04245)

[3] Liu et al. "Swin Transformer V2: Scaling Up Capacity and Resolution" 2022. [https://arxiv.org/abs/2111.09883](https://arxiv.org/abs/2111.09883)

[4] Dehghani et al. "Scaling Vision Transformers to 22 Billion Parameters" 2023. [https://arxiv.org/abs/2302.05442](https://arxiv.org/abs/2302.05442)

[5] Zhai et al. "Stabilizing Transformer Training by Preventing Attention Entropy Collapse" 2023. [https://arxiv.org/abs/2303.06296](https://arxiv.org/abs/2303.06296)

[6] Team OLMo et al. "2 OLMo 2 Furious" 2024. [https://arxiv.org/abs/2501.00656](https://arxiv.org/abs/2501.00656)

[7] Chameleon Team. "Chameleon: Mixed-Modal Early-Fusion Foundation Models" 2024. [https://arxiv.org/abs/2405.09818](https://arxiv.org/abs/2405.09818)

[8] Gemma Team. "Gemma 3 Technical Report" 2025. [https://arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786)

[9] Peng et al. "YaRN: Efficient Context Window Extension of Large Language Models" 2023. [https://arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)

[10] Yang, Hu et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" 2022. [https://arxiv.org/abs/2203.03466](https://arxiv.org/abs/2203.03466)

[11] Gemma Team. "Gemma 2: Improving Open Language Models at a Practical Size" 2024. [https://arxiv.org/abs/2408.00118](https://arxiv.org/abs/2408.00118)

[12] xAI. "Grok-1 open release (model.py)" 2024. [https://github.com/xai-org/grok-1](https://github.com/xai-org/grok-1)