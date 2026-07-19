# DARF: Diffusion-Adaptive Random Feature Model

**Generalization Bound for Diffusion Models using Adaptive Random Features**

Esha Saha¹, Giang Tran¹
¹ University of Waterloo, Canada
📧 esaha@uwaterloo.ca, giang.tran@uwaterloo.ca

[Paper (AAAI 2026 MATH4AI Workshop)](#citation) · [Code](https://github.com/esha-saha/darf)

---

## Overview

Diffusion models generate high-quality samples but are computationally expensive and largely uninterpretable, which makes them a poor fit for applications where a **theoretical guarantee** on the generated samples matters as much as (or more than) sample quality — e.g. simulating mesoscale climate maps.

**DARF (Diffusion-Adaptive Random Feature Model)** replaces the usual U-Net noise predictor with a *random feature* architecture: a single, fixed random hidden layer per timestep, combined with a small set of **trainable, timestep-adaptive weights** inspired by positional encoding. Because the hidden layer is random and fixed rather than fully learned, the resulting model is:

- **Parameter-efficient** — trains with under half the parameters of a comparable U-Net.
- **Interpretable** — the learned reverse process reduces exactly to a classical random feature model at each timestep, so existing random-feature approximation theory applies directly.
- **Theoretically grounded** — the paper derives an explicit, non-asymptotic bound on the total variation distance between the true and DARF-generated distributions, and shows how the number of random features $N$ controls that bound.

Numerically, DARF matches or outperforms a parameter-matched U-Net, a fully-connected variant, and a classical (non-adaptive) random feature model on image (Fashion-MNIST) and audio generation/denoising tasks — particularly under limited data and compute.

## Key Equations

**1. Noise predictor.** For timestep $k$, DARF predicts the noise added at that step using a random-feature layer $(\mathbf{W}, \mathbf{b})$ gated by trainable, timestep-adaptive weights $\boldsymbol\theta^{(1)}, \boldsymbol\theta^{(2)}$:

$$
p_\theta(\mathbf{x}_k, k) = \left(\sin(\mathbf{x}_k^{T}\mathbf{W} + \mathbf{b}^{T}) \odot \cos(\boldsymbol\tau_k^{T}\boldsymbol\theta^{(1)})\right)\boldsymbol\theta^{(2)}
$$

**2. Training objective.** Parameters $\theta = \{\boldsymbol\theta^{(1)}, \boldsymbol\theta^{(2)}\}$ are learned by regressing onto the injected noise $\epsilon_k$ across all $K$ timesteps, while $\mathbf{W}, \mathbf{b}$ stay fixed and random:

$$
L = \frac{1}{K}\sum_{k=1}^{K}\big\|\epsilon_k - p_\theta(\mathbf{x}_k, k)\big\|_2^{2}
$$

**3. Equivalence to a random feature model.** For fixed $k$, DARF's output reduces to a standard random feature expansion with feature map $\phi(\mathbf{x}_k;\boldsymbol\omega_i) = \sin(\mathbf{x}_k^{T}\boldsymbol\omega_i + b_i)$ and coefficients $c_{ij} = \cos(\theta^{(1)}_{ki})\,\theta^{(2)}_{ij}$ — this equivalence (formalized as $\mathcal{G}_{k,\omega} = \mathcal{F}_\omega$) is what lets classical random-feature approximation theory be imported into the diffusion setting.

**4. Random-feature approximation error.** With $N$ i.i.d. random features drawn from $\rho$, the best DARF approximation $f^{\sharp}$ of a target function $f^{*}$ satisfies, with probability at least $1-\delta$:

$$
\|f^{\sharp} - f^{*}\|_2 \;\le\; \frac{C\sqrt{d}}{\sqrt{N}}\left(1 + \sqrt{2\log\frac{1}{\delta}}\right)
$$

**5. Main generalization bound.** Combining the random-feature error above with the forward-process and SDE-discretization errors of (Chen et al. 2023), the total variation distance between the true distribution $q(\mathbf{x}_0)$ and DARF's generated distribution $p_\theta(\mathbf{x}_0, 0)$ satisfies, with probability at least $1-\delta$:

$$
TV\big(p_\theta(\mathbf{x}_0,0),\, q(\mathbf{x}_0)\big) \;\lesssim\;
\underbrace{\sqrt{KL(q(\mathbf{x}_0)\Vert\gamma)}\,e^{-T}}_{\text{forward-process error}}
\;+\;
\underbrace{(L\sqrt{d}\,h + Lm_2 h)\sqrt{T}}_{\text{SDE discretization error}}
\;+\;
\underbrace{\frac{C_2\sqrt{TKd}}{\sqrt{N}}\left(1+\sqrt{2\log\tfrac{1}{\delta}}\right)}_{\text{DARF approximation error}}
$$

The first two terms are architecture-independent and vanish with a suitable choice of horizon $T$ and step size $h$; the third term is DARF-specific and shows the approximation error decays as $\mathcal{O}(1/\sqrt{N})$, giving a principled way to choose the number of random features $N$ for guaranteed convergence.

## Results Summary

| Task | Comparison | Finding |
|---|---|---|
| Image generation (Fashion-MNIST) | DARF vs. U-Net, fully-connected variant (NN), classical random features (RF) | DARF achieves the lowest FID at matched parameter count under limited data (100 training images) |
| Image / audio denoising | DARF vs. NN, RF | DARF is comparable to or better than NN in PSNR, while remaining interpretable; RF fails to denoise effectively |
| Effect of timesteps | 100 vs. 1000 timesteps | More timesteps improve generative sample quality but not denoising quality |

**Limitation:** DARF's number of required features scales with $K \times d$ (timesteps × input dimension), so the model currently targets lower-dimensional distributions; scaling to high-resolution data and deeper (non-shallow) variants of DARF are noted as future work.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{saha2026darf,
  title     = {Generalization Bound for Diffusion Models using Adaptive Random Features (DARF)},
  author    = {Saha, Esha and Tran, Giang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  organization = {Association for the Advancement of Artificial Intelligence}
}
```

## Contact

For questions about the paper or code, please reach out to **Esha Saha** (esaha@uwaterloo.ca) or **Giang Tran** (giang.tran@uwaterloo.ca), University of Waterloo.
