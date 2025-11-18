---
title: "The Unreasonable Effectiveness of Scale"
date: 2025-11-17
permalink: /posts/2025/11/scaling-laws-in-ai/
tags:
  - machine learning
  - deep learning
  - large language models
  - scaling laws
  - AI
---

Scaling laws describe the relationship between a model's performance and the scale of three key ingredients: the number of model parameters, the size of the dataset, and the amount of computational power used for training. The core finding is that as you increase these resources, the model's performance improves in a predictable, power-law fashion.

Here, I dive into the technical details of these scaling laws, their implications, and the nuances discovered by subsequent research.

## Performance as a Power-Law

The work research from OpenAI in 2020, "Scaling Laws for Neural Language Models," [1] established that the cross-entropy loss of a model follows a power-law relationship with model size (N), dataset size (D), and the amount of training compute (C).

The test loss $L$ can be modeled as a function of each of these factors:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}
$$
$$
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
$$
$$
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

Where:
- $N$ is the number of model parameters (excluding embeddings).
- $D$ is the size of the training dataset in tokens.
- $C$ is the amount of compute used for training in FLOPs.
- $N_c, D_c, C_c$ and $\alpha_N, \alpha_D, \alpha_C$ are constants determined by fitting the power-law to experimental data. The exponents ($\alpha$) are typically small positive values (e.g., ~0.05-0.08).

This means that if you plot the model's loss against any of these three factors on a log-log scale, you get a straight line. This implies that you can train smaller models and extrapolate their performance to predict how a much larger, more expensive model will perform before you even start training it.

## The Irreducible Loss

The power-law doesn't continue to zero. There is a floor to the loss, an "irreducible" component $L_\infty$, which represents the inherent entropy or randomness of the data itself. The true loss is better modeled as the sum of a scalable term and this irreducible term:

$$
L(N, D) = \left( \frac{N_c}{N} \right)^{\alpha_N} + \left( \frac{D_c}{D} \right)^{\alpha_D} + L_\infty
$$

This formula shows that performance is ultimately limited by both model size and dataset size. If you only increase the model size ($N \to \infty$), the loss will be bottlenecked by the dataset term. Conversely, if you only increase the dataset size ($D \to \infty$), the loss will be limited by the model's capacity.

## Compute-Optimal Scaling

The original scaling laws treated these factors somewhat independently. However, a 2022 paper from DeepMind, "Training Compute-Optimal Large Language Models" [2], provided some refinement.

The researchers found that for a fixed computational budget, the best performance is not achieved by training the largest possible model. Instead, **model size and dataset size should be scaled in equal proportion.**

They proposed that for every doubling of model size, the training dataset size should also be doubled. This was a significant departure from the trend at the time, which favored creating enormous models and training them on comparatively smaller datasets. For example, the Chinchilla paper suggested that GPT-3 was "undertrained"—it was too large for the amount of data it was trained on and would have performed better if it were a smaller model trained on more data.

The Chinchilla scaling laws predict that for a given compute budget $C$, the optimal model size $N_{opt}$ and dataset size $D_{opt}$ are:

$$
N_{opt}(C) \propto C^a \quad \text{and} \quad D_{opt}(C) \propto C^b \quad \text{where } a+b \approx 1
$$



## Implications and Future Directions

The discovery of scaling laws has had a monumental impact on the field of AI:

1.  **Predictability and Engineering:** They turn the art of model training into more of a science. Organizations can now make informed, quantitative decisions about resource allocation, predicting the return on investment for increasing compute or data.
2.  **The Primacy of Scale:** The laws suggest that many performance gains are not from clever architectural tweaks but simply from scaling up. This has fueled the race for more computational power and larger datasets.
3.  **Emergent Abilities:** One of the most fascinating consequences of scale is the emergence of abilities that are not present in smaller models. Capabilities like multi-step reasoning, instruction following, and in-context learning appear to manifest only after a certain threshold of scale is crossed.


## References
[1] Kaplan, J., McCandlish, S., Henighan, T.J., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. ArXiv, abs/2001.08361.
[2] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, and Laurent Sifre. 2022. Training compute-optimal large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 2176, 30016–30030.
