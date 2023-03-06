# Consistency Models

<!-- #region -->
<p align="center">
<img  src="contents/_ct_sample_2step_30.png">
</p>
<!-- #endregion -->

> 30 Epoch, Consistency Model with 2 step. Using $t_1 = 2, t_2 = 80$.

<!-- #region -->
<p align="center">
<img  src="contents/_ct_sample_5step_30.png">
</p>
<!-- #endregion -->

> 30 Epoch, Consistency Model with 5 step. Using $t_i \in \{5, 10, 20,40, 80\}$.




Unofficial Implementation of Consistency Models  ([paper](https://arxiv.org/abs/2303.01469)) in pytorch.

Three days ago, legendary man [Yang Song](https://yang-song.net/) released entirely new set of generative model, called consistency models. There are yet any implementations, so here is my attempt at it.




Mainly implements consistency training:

$$
L(\theta) = \mathbb{E}[d(f_\theta(x + t_{n + 1}z, t_{n + 1}), f_{\theta_{-}}(x + t_n z, t_n))]
$$

And sampling:

$$
\begin{align}
z &\sim \mathcal{N}(0, I) \\
x &\leftarrow x + \sqrt{t_n ^2 - \epsilon^2} z \\
x &\leftarrow f_\theta(x, t_n) \\
\end{align}
$$

## Usage

```bash
python main.py
```

## Todo

- [x] EMA
- [ ] CIFAR10 Example
- [x] Samples are sooo fuzzy... try to get a crisp result.
- [ ] Consistency Distillation