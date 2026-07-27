# Adversarial Attacks on CIFAR-10 Classifiers

Coursework for **CSIT375 – AI and Cybersecurity**, University of Wollongong (2026).

Three targeted attacks against a pretrained VGG-11 (batch-norm) CIFAR-10 classifier, each
under a different threat model: a grey-box transfer attack, a universal perturbation, and an
adaptive attack that defeats a randomised defence.

All attacks are **targeted** (they must force one specific wrong class, not just any wrong
class) and all perturbations are bounded in $L_\infty$.

---

## Results

| Task | Threat model | Budget | Fooling rate | Marks |
|---|---|---|---|---|
| 1. Grey-box adversarial examples | No target gradients | $L_\infty \le 0.04$ | **82.0%** | 6 / 6 |
| 2. Universal adversarial perturbation | White-box | $L_\infty \le 0.06$ | **95.0%** | 5 / 5 |
| 3. Adaptive attack vs. randomised defence | White-box + known defence | $L_\infty \le 0.04$ | **94.0%** | 1 / 1 (bonus) |

Target model: `vgg11_bn`, 87.22% clean test accuracy on CIFAR-10.

---

## Task 1 — Grey-box targeted adversarial examples

The rules forbid backpropagating through the target. Only two facts about it are known: it is
a `vgg11_bn`, and it was trained with a specific known recipe.

**Approach — targeted transfer attack.**

1. **Surrogate ensemble.** Three models trained from scratch with the target's exact recipe
   (AdamW, lr 1e-3, weight decay 1e-5, batch 64, 50 epochs, normalised CIFAR-10, no
   augmentation): two `vgg11_bn` with different seeds plus one `vgg13_bn`. Matching the
   architecture and recipe puts their decision boundaries close to the target's; using several
   stops the perturbation from overfitting any single network.
2. **MI-DI-FGSM with the logit loss.** Momentum on the $L_1$-normalised gradient, input
   diversity (random resize to 24–31 px, pad back to 32 px, p = 0.7), 500 iterations, step size
   decayed from $\varepsilon/10$ to $\varepsilon/40$, projected onto the $L_\infty$ ball and
   $[0,1]$ every step.

   The **logit loss** matters here. Targeted cross-entropy saturates once the surrogates become
   confident — softmax → 1, gradients → 0 — so the back half of the run stops buying
   transferability. Maximising the raw target logit never saturates.
3. **Restarts.** Up to 6 runs, re-attacking only images that have not yet succeeded.

**What actually drove the result.** The first transfer run alone reached 81/100; five further
restarts added a single image. The remaining failures are consistent rather than random — those
images are genuinely hard to push to their assigned target class — so the result is
overwhelmingly a pure transfer effect.

Ablation (same surrogates, same budget):

| Configuration | Fooling rate |
|---|---|
| Targeted CE loss | 76% |
| \+ logit loss, + scale invariance | 77% |
| \+ logit loss, − scale invariance, + restarts | **82%** |

Scale invariance (averaging over $x$, $x/2$, $x/4$) is a standard ImageNet trick that *hurts*
on CIFAR: dividing a 32×32 image by 4 yields a near-black input far outside the surrogates'
BatchNorm statistics, so those copies contribute noise.

---

## Task 2 — Universal adversarial perturbation

One perturbation, shape `[1, C, H, W]`, that must push **every** image to class 3 (*cat*).

Broadcast a single `delta` over the whole batch and minimise cross-entropy toward the target
class averaged over all 100 images — that batch-wide averaging is exactly what makes the
perturbation universal rather than image-specific. Momentum, sign steps with a decaying step
size, and a clamp to $[-0.06, 0.06]$ every iteration.

Because cross-entropy is only a proxy for fooling rate and sign steps make the trajectory
non-monotone, the true fooling rate is measured every 10 iterations and the best iterate kept.

---

## Task 3 — Adaptive attack against a randomised defence

The defence crops a random 20–50% region, resizes back to 32×32, repeats 21 times, and returns
the **majority vote** as a one-hot vector. It reduces the Task 1 examples from 82% to 25%.

Two properties break a standard attack: the one-hot output has zero gradient everywhere, and
every forward pass sees a different crop.

**Approach — Expectation Over Transformation.** Don't attack the wrapper; attack the underlying
differentiable model *under the same random crops*, averaging the targeted loss over 48 samples
per step (backpropagated in chunks of 8 to bound memory). This estimates

$$\nabla_\delta \; \mathbb{E}_{t \sim T}\big[\mathcal{L}\big(f(t(x+\delta)),\, y_{\text{tgt}}\big)\big]$$

so the perturbation is optimised to survive *any* crop rather than one particular draw.

**Evaluating a stochastic defence is itself a trap.** A single 21-crop vote is a noisy
measurement, so keeping the best of many such scores selects whichever candidate got a lucky
draw. An early version did exactly that and reported 96% while the real rate was 94%; scoring
per image made it far worse (94.8% reported, 80% actual) because each image then got its own
long run of lucky draws — an image with a true survival probability of 0.7 will still hit a
perfect 5/5 in some round almost every time.

The fix is to compare **few** candidates and measure each **well**: 9 snapshots from the
converged half of the run, each scored over 20 independent defence runs, then a per-image pick
among those 9, and finally a re-measurement on 25 fresh runs the selection never saw. The
reported estimate (91.9%) then tracks the true rate (94.0%) instead of sitting above it.

---

## Repository contents

```
adversarial-attacks-cifar10.ipynb   # the full notebook, with outputs
README.md
.gitignore
```

## Not included, and why

| Excluded | Reason |
|---|---|
| `codebase/` | University-provided library (© 2026 University of Wollongong) |
| `assignment1_CSIT375.pdf`, blank notebook | University-provided task materials |
| `target_model/` (323 MB) | Provided pretrained weights; also over GitHub's 100 MB file limit |
| `data/`, `out/` | CIFAR-10 download and surrogate checkpoints, regenerated on run |

The notebook is therefore a **record of the work, not a runnable artefact** — reproducing it
needs the course materials above.

## Reproducing

1. Kaggle notebook, accelerator **GPU T4 ×2** (a P100 fails — current PyTorch builds dropped
   `sm_60` support), **Internet on** for the CIFAR-10 download.
2. Attach the course codebase and the target-model weights as datasets; update the
   `sys.path.append` calls and `target_ckpt_dir` / `TARGET_CKPT_DIR` to match.
3. Run all cells. First run trains three surrogates (~1.6 h); they are checkpointed, so later
   runs load them in seconds.

Runtimes on a T4: surrogate training ~28 min per `vgg11_bn`, Task 1 attack ~10 min, Task 2
~1 min, adaptive attack ~10 min.

---

## Notes

Submitted coursework, published as a portfolio record of my own work. The randomised defence
means the Task 3 fooling rate varies by a few percent between runs.
