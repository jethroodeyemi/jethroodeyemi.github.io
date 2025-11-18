---
title: "A Technical Deep Dive into Exploding Gradients"
date: 2025-11-17
permalink: /posts/2025/11/exploding-gradients-problem/
tags:
  - machine learning
  - neural networks
  - exploding gradients
  - data normalization
  - deep learning
  - backpropagation
---

I remember one of the experiences I had duing my MS in Computer Science at Georgia Tech while working on a CNN for protein data. I was feeding raw protein data as an image, with pixel values in the standard 0-255 range, directly into the network. My model's accuracy was stuck below 20%, and the loss was oscillating wildly. After hours of debugging, I traced the issue to its source: I had neglected to normalize my input data, leading to a classic case of "exploding gradients."

In this post, I'll offer a more technical deep dive into this phenomenon, exploring the mathematical questions of why gradients explode and discussing several strategies to mitigate it.

## The Chain Rule and the Root of the Explosion

To understand why gradients explode, we must first revisit the mechanism of backpropagation. During training, we compute the gradient of the loss function $L$ with respect to the network's weights $W$. For a deep network, this involves the chain rule, propagating the gradient from the final layer back to the initial ones.

Consider a deep network with $n$ layers. The gradient of the loss with respect to the weights $W_i$ in an early layer $i$ is a product of many terms:

$$
\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdot \ldots \cdot \frac{\partial a_{i+1}}{\partial a_i} \cdot \frac{\partial a_i}{\partial W_i}
$$

Each term $\frac{\partial a_{j+1}}{\partial a_j}$ represents the Jacobian matrix of the activations of one layer with respect to the activations of the previous one. If the magnitudes of these terms are consistently greater than 1, their product will grow exponentially as we propagate backward, leading to an "explosion."

The weight update rule is given by:

$$
W_{new} = W_{old} - \eta \frac{\partial L}{\partial W_{old}}
$$

When the gradient $\frac{\partial L}{\partial W_{old}}$ is enormous, the weight update becomes a massive leap in the weight space. This can cause the optimizer to drastically overshoot any optimal minima, leading to an unstable training process where the loss function fluctuates erratically and fails to converge. In worst-case scenarios, this can even result in numerical overflow, producing `NaN`  values in your weights and loss.

## The Data Scale

The scale of the input data is a primary contributor to this problem. Large input values lead to large activation values, which in turn can produce large gradients.

Here's a visualization of what a batch of unnormalized image data might look like as a matrix. The wide range of values is a clear issue.

```
[[ 23, 180,  45, 210,  90],
 [150,  88, 200,  12, 240],
 [ 75, 190,   5, 160, 220],
 [110,  30, 255,  95,  60],
 [ 20, 170, 130,  40, 115]]
```

## Mitigation Strategies

Fortunately, several effective techniques exist to combat exploding gradients.

### 1. Data Normalization

Scaling input data to a smaller range ensures that the initial activations and subsequent gradients remain manageable. A common technique is Min-Max scaling:

$$
X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

For image data with pixel values from 0 to 255, this simplifies to dividing by 255. This simple step confines the input values to the range [0, 1].

Here's the same data matrix after normalization.

```
[[0.090, 0.706, 0.176, 0.824, 0.353],
 [0.588, 0.345, 0.784, 0.047, 0.941],
 [0.294, 0.745, 0.020, 0.627, 0.863],
 [0.431, 0.118, 1.000, 0.373, 0.235],
 [0.078, 0.667, 0.510, 0.157, 0.451]]
```

### 2. Gradient Clipping

This is a direct intervention that forcibly constrains the magnitude of the gradients. During the backward pass, if the L2 norm of the gradient vector $g$ exceeds a predefined threshold $\tau$, the gradient is scaled down:

$$
\text{if } ||g|| > \tau, \quad g \leftarrow \frac{\tau}{||g||} g
$$

This ensures that no single update step is excessively large, preventing the optimizer from making drastic jumps in weight space.

### 3. Weight Regularization

Weight regularization (specifically L2 regularization, or weight decay) can also help. It adds a penalty term to the loss function, which discourages the weights from growing too large:

$$
L_{reg} = L_{original} + \lambda \sum_{i} W_i^2
$$

By keeping the weights small, regularization indirectly helps to keep the gradients in a reasonable range.

## The Results: Convergence

Implementing normalization was the key. My model's accuracy immediately jumped from under 20% to over 90%. The hypothetical learning curves tell the story.

**Before Normalization:** The loss is erratic, and accuracy is flat. A clear sign of unstable training.
![Learning Curve Before Normalization](https://jethroodeyemi.github.io/files/2025_11_17_post/learning_curve_before.png)

**After Normalization:** The loss steadily decreases, and accuracy climbs smoothly towards convergence.
![Learning Curve After Normalization](https://jethroodeyemi.github.io/files/2025_11_17_post/learning_curve_after.png)

## Conclusion

The exploding gradients problem is a fundamental challenge in deep learning, rooted in the mathematics of backpropagation. While it can be a frustrating roadblock, it is well-understood. By normalizing input data, employing gradient clipping, and using weight regularization, we can effectively mitigate this issue and ensure stable training of deep neural networks.