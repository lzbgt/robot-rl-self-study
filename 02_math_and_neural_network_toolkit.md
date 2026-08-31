# 2. Math and Neural-Network Toolkit

Deep reinforcement learning combines probability, calculus, linear algebra,
and optimization. You do not need to master each subject before beginning. You
do need to know what the symbols mean, what shape each quantity has, and what
an equation asks the computer to do.

This chapter builds that minimum toolkit. Keep a pencil nearby: predicting a
number before running code is much more educational than only reading it.

### Where this toolkit came from

Deep reinforcement learning (RL) did not invent its mathematics. Bellman's dynamic programming uses
probability and recursive value equations; Robbins and Monro's 1951 stochastic
approximation explains learning from noisy samples; likelihood-ratio estimators
make policy gradients possible; and reverse-mode automatic differentiation
turns the chain rule into practical neural-network training. Modern libraries
compose these ideas at tensor scale.

The goal of this chapter is therefore not to survey all of calculus or linear
algebra. It is to build a trace from every symbol used later to an operation a
program performs, including its shape, units, estimate error, and failure mode.

## 2.1 Scalars, vectors, matrices, and tensors

A **scalar** is one number, such as the reward at one instant:

```math
r_t = 1.7.
```

A **vector** is an ordered list. A simple mobile-robot command might be

```math
c_t = [v_x, v_y, \omega_z] = [0.2, 0.0, -0.4].
```

The order is part of the interface. Swapping $v_x$ and $v_y$ does not merely
rename two values; it asks for a different physical motion.

A **matrix** is a rectangular array. A neural-network layer maps an input
vector $x$ to an output vector $z$ using a weight matrix $W$ and bias $b$:

```math
z = Wx + b.
```

If $x$ has 61 values and the layer has 512 neurons, then

```math
x\in\mathbb{R}^{61},\quad W\in\mathbb{R}^{512\times61},
\quad b\in\mathbb{R}^{512},\quad z\in\mathbb{R}^{512}.
```

Read $\mathbb{R}^{61}$ as “a vector of 61 real numbers.” The matrix shape says
512 rows, one per output, and 61 columns, one per input. Shape reasoning catches
many bugs before training.

A **tensor** is the programming term for a multidimensional numeric array. A
Microduck rollout tensor might have shape

```text
(time steps, parallel environments, observation values)
             (24, 4096, 61)
```

The word tensor is often used casually in deep-learning code; here it does not
require advanced tensor calculus.

### Units are an invisible part of every vector

Shape alone is not enough. A value might be radians, radians per second,
meters, normalized units, or a Boolean encoded as 0/1. Write interfaces as
both shape and meaning:

```text
base angular velocity: shape (3,), unit rad/s, expressed in body frame
projected gravity:      shape (3,), dimensionless, body frame
joint position delta:  shape (14,), unit rad, relative to HOME
```

This discipline matters because a numerically valid 61-vector with the wrong
units or frame still produces a physically wrong action.

### Dot products, norms, and batches

The **dot product** combines two equal-length vectors into one scalar:

```math
x^Ty=\sum_{i=1}^{n}x_i y_i.
```

It appears in every neural layer and many robot costs. If
$x=[1,2,-1]$ and $y=[3,0,4]$, then $x^Ty=3+0-4=-1$.

The Euclidean norm measures vector magnitude:

```math
\lVert x\rVert_2=\sqrt{x^Tx}
=\sqrt{\sum_i x_i^2}.
```

A squared tracking error $\lVert v-v^*\rVert_2^2$ therefore means “square each
component error and add.” It grows quadratically and makes one large miss more
expensive than several small ones with the same absolute sum. The $L_1$ norm,
$\lVert x\rVert_1=\sum_i|x_i|$, grows linearly and has a corner at zero. Which
one is used changes both optimization gradients and outlier sensitivity.

For a batch matrix $X\in\mathbb{R}^{B\times61}$ and a layer stored as
$W\in\mathbb{R}^{512\times61}$, frameworks commonly evaluate

```math
Z=XW^T+\mathbf{1}b^T,
```

so $Z\in\mathbb{R}^{B\times512}$. The bias is **broadcast**—logically copied
across the $B$ rows. A layer with 61 inputs and 512 outputs has
$61\times512+512=31{,}744$ learned scalar parameters.

## 2.2 Functions and parameters

A **function** maps an input to an output. A policy is a function:

```math
a_t = \pi_\theta(o_t).
```

- $o_t$ is the observation at time $t$;
- $a_t$ is the action;
- $\pi$ is the policy; and
- $\theta$ is the collection of trainable weights and biases.

The subscript $\theta$ means changing those parameters changes the function.
Training searches for parameters that make useful trajectories more likely.

A **hyperparameter** is chosen by the experimenter rather than learned by the
optimizer: learning rate, discount factor, network width, Proximal Policy
Optimization (PPO) clip value, or rollout length are examples.

## 2.3 Probability is how reinforcement learning (RL) represents uncertainty

Robots face uncertainty from sensor noise, contact variation, delayed state,
random commands, randomized simulation, and exploration.

A **random variable** is a numeric outcome of an uncertain process. We write

```math
X\sim p(x)
```

as “$X$ is sampled from probability distribution $p$.” A Gaussian (normal)
distribution is written

```math
X\sim\mathcal{N}(\mu,\sigma^2),
```

where $\mu$ is the mean and $\sigma$ is the standard deviation. Larger
$\sigma$ means more spread.

During training, a continuous policy commonly produces a mean action and a
standard deviation, then samples:

```math
a_t\sim\pi_\theta(\cdot\mid o_t)
     = \mathcal{N}(\mu_\theta(o_t),\mathrm{diag}(\sigma^2)).
```

Read this as: “given observation $o_t$, the network proposes the center of an
action distribution, and an action is sampled from it.” During deployment,
the deterministic mean is often used instead.

### Conditional probability

$p(a\mid o)$ means “the probability of action $a$ given observation $o$.” The
vertical bar means “conditioned on.” A policy must respond differently to a
tilting-left observation and a tilting-right observation, so it represents a
conditional distribution rather than one fixed action distribution.

Joint and conditional probability are related by the product rule:

```math
p(x,y)=p(x\mid y)p(y)=p(y\mid x)p(x).
```

Rearranging gives Bayes' rule:

```math
p(x\mid y)=\frac{p(y\mid x)p(x)}{p(y)}.
```

In state estimation, $p(x)$ is a prior belief about robot state, $p(y\mid x)$
is a sensor likelihood, and $p(x\mid y)$ is the posterior after observing the
measurement. A learned recurrent actor need not explicitly calculate this
fraction, but its history-processing role is related: infer hidden conditions
from their observable effects.

### Expectation

The **expectation** $\mathbb{E}[X]$ is a probability-weighted average. For a
six-sided die,

```math
\mathbb{E}[X] = \frac{1+2+3+4+5+6}{6}=3.5.
```

No roll produces 3.5. Expectation describes the long-run average, not a
guaranteed single result. Likewise, maximizing expected return does not
guarantee that every robot episode succeeds. Tail failures need explicit
measurement and safety handling.

### Variance

Variance measures spread around the mean:

```math
\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2].
```

Two policies can have the same mean return while one fails violently in 10% of
trials. Reporting only the mean hides that distinction.

For a finite distribution with outcomes $x_i$ and probabilities $p_i$:

```math
\mu=\sum_i p_i x_i,
\qquad
\mathrm{Var}(X)=\sum_i p_i(x_i-\mu)^2.
```

Standard deviation is $\sigma=\sqrt{\mathrm{Var}(X)}$ and has the original
unit. Variance of an angle measured in radians has units radian squared;
standard deviation is again in radians.

Two variables can vary together. Their covariance is

```math
\mathrm{Cov}(X,Y)=
\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])].
```

A diagonal Gaussian policy assumes action noise has zero cross-joint
covariance after conditioning on the observation. The neural mean can still
coordinate joints; the simplifying assumption concerns sampled residual noise.
Full-covariance policies can represent correlated exploration but require more
parameters and stable matrix operations.

### From population expectation to a sample estimate

The objective contains expectations over distributions we cannot enumerate.
With $N$ sampled values $x_1,\ldots,x_N$, the Monte Carlo estimate is

```math
\hat\mu=\frac{1}{N}\sum_{i=1}^{N}x_i.
```

If samples are independent with variance $\sigma^2$, then
$\mathrm{Var}(\hat\mu)=\sigma^2/N$ and the standard error scales as
$1/\sqrt{N}$. Four times as many independent samples roughly halves this
sampling uncertainty, not quarters it. Robot trajectory samples are correlated
within episodes, so treating every time step as independent can greatly
overstate confidence; evaluation usually aggregates by seed, episode, or
hardware trial.

## 2.4 Why logarithms appear in policy gradients

For independent events, probabilities multiply. Products of many numbers
smaller than one become numerically tiny. Logarithms turn products into sums:

```math
\log(xy)=\log x+\log y.
```

The log-probability of a sampled action is convenient to differentiate. A key
identity is

```math
\nabla_\theta \log \pi_\theta(a\mid o)
= \frac{\nabla_\theta\pi_\theta(a\mid o)}
       {\pi_\theta(a\mid o)}.
```

You do not need to memorize the fraction yet. The practical idea is that the
gradient can adjust the probability of actions that were actually sampled.
Positive advantage pushes their log-probability up; negative advantage pushes
it down.

For one scalar Gaussian action, the density is

```math
\pi(a\mid o)=
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left[-\frac{(a-\mu)^2}{2\sigma^2}\right].
```

Taking its logarithm turns multiplication into addition:

```math
\log\pi(a\mid o)=
-\frac{1}{2}
\left[
\frac{(a-\mu)^2}{\sigma^2}
+2\log\sigma+\log(2\pi)
\right].
```

Differentiate with respect to the mean:

```math
\frac{\partial}{\partial\mu}\log\pi(a\mid o)
=\frac{a-\mu}{\sigma^2}.
```

If the sampled action is above the mean, increasing the mean makes it more
likely; if it is below, the gradient points downward. Smaller $\sigma$ makes
the same deviation more surprising and produces a larger score magnitude.
This local derivative is the concrete mechanism behind “increase the
probability of an advantageous action.”

For an independent $d$-dimensional diagonal Gaussian, implementations sum one
such log-probability per action dimension. Run
[`examples/gaussian_policy_math.py`](examples/gaussian_policy_math.py) to
compare the analytic derivative with a finite difference and to calculate
Gaussian entropy.

## 2.5 Derivatives: sensitivity to change

The derivative of a scalar function says how its output changes for a small
input change. If

```math
y=x^2,
```

then

```math
\frac{dy}{dx}=2x.
```

At $x=3$, the derivative is 6. Increasing $x$ by approximately 0.01 increases
$y$ by approximately $6\times0.01=0.06$.

A **partial derivative** changes one input while holding others fixed. For

```math
f(x,y)=x^2+3xy,
```

```math
\frac{\partial f}{\partial x}=2x+3y,
\qquad
\frac{\partial f}{\partial y}=3x.
```

A **gradient** collects all partial derivatives:

```math
\nabla f =
\begin{bmatrix}
\partial f/\partial x\\
\partial f/\partial y
\end{bmatrix}.
```

The gradient points toward the locally steepest increase. To minimize a loss
$L(\theta)$, gradient descent takes a small step in the opposite direction:

```math
\theta_{new}=\theta_{old}-\alpha\nabla_\theta L,
```

where $\alpha$ is the **learning rate**, the step-size hyperparameter.

When both input and output are vectors, derivatives form a **Jacobian**. For
$f:\mathbb{R}^n\rightarrow\mathbb{R}^m$:

```math
J_{ij}=\frac{\partial f_i}{\partial x_j},
\qquad J\in\mathbb{R}^{m\times n}.
```

Robot kinematics uses a Jacobian to map joint velocity to end-effector
velocity. Neural-network backpropagation also multiplies Jacobian-vector
products, but automatic differentiation avoids materializing every huge
matrix.

### Numerical example

Let $L(w)=(w-4)^2$ and start at $w=1$.

```math
\frac{dL}{dw}=2(w-4)=-6.
```

With $\alpha=0.1$:

```math
w_{new}=1-0.1(-6)=1.6.
```

The update moves $w$ toward 4. A huge learning rate could overshoot; a tiny one
could make progress impractically slow.

## 2.6 The chain rule and backpropagation

Neural networks compose functions. Suppose

```math
u=wx,\qquad y=\tanh(u),\qquad L=(y-y^*)^2.
```

The **chain rule** multiplies local sensitivities along the path:

```math
\frac{dL}{dw}
=\frac{dL}{dy}\frac{dy}{du}\frac{du}{dw}.
```

Substitute each derivative:

```math
\frac{dL}{dw}
=2(y-y^*)\,(1-\tanh^2(u))\,x.
```

**Backpropagation** is an efficient application of this chain rule through a
computation graph. Automatic differentiation libraries such as PyTorch compute
it, but understanding the path helps diagnose detached tensors, exploding
gradients, saturation, and unintended objectives.

## 2.7 A neural network is a learned nonlinear function

A feed-forward layer is

```math
h=\phi(Wx+b),
```

where $\phi$ is an **activation function** applied element by element. Without
nonlinear activations, many stacked linear layers collapse to one linear map.

Common activations include:

| Activation | Formula or behavior | Practical note |
| --- | --- | --- |
| rectified linear unit (ReLU) | $\max(0,x)$ | cheap; negative side has zero gradient |
| exponential linear unit (ELU) | $x$ if positive, smooth negative saturation otherwise | used by the Microduck actor/critic |
| tanh | maps to $(-1,1)$ | useful for bounded outputs; can saturate |
| sigmoid | maps to $(0,1)$ | common for probabilities or gates |

The Microduck actor is conceptually

```text
61 observations -> 512 ELU -> 256 ELU -> 128 ELU -> 14 action means
```

The numbers 512, 256, and 128 are hidden-layer widths. This network is not a
database of motions. Its weights define a smooth high-dimensional mapping from
observations and commands to actions.

## 2.8 Loss, objective, reward, and return are different

These words are easy to mix up:

- **reward**: one scalar emitted by the environment at one step;
- **return**: a discounted sum of future rewards along a trajectory;
- **objective**: the quantity the algorithm conceptually wants to maximize;
- **loss**: a quantity the optimizer minimizes in code.

An implementation may minimize the negative policy objective, add a value loss,
and subtract an entropy bonus. Seeing a negative sign in code does not by
itself mean the robot is punished.

## 2.9 Stochastic gradient descent, minibatches, and epochs

Computing a gradient over all possible trajectories is impossible. RL estimates
it from sampled experience.

A **batch** is the collected set of samples. A **minibatch** is one subset used
for an optimizer step. An **epoch** is one pass through the batch. PPO can reuse
one on-policy rollout for several epochs, but too much reuse makes the updated
policy drift away from the policy that generated the data.

Optimizers such as Adam maintain moving estimates of gradient scale. Adam can
make training easier, but it does not repair a wrong reward, missing
observation, impossible target, or simulator error.

### Why a sampled gradient can still be useful

Let $g_i$ be a gradient contribution from sample $i$. A minibatch estimator is

```math
\hat g=\frac{1}{B}\sum_{i=1}^{B}g_i.
```

If the sampling procedure matches the objective, $\mathbb{E}[\hat g]$ can
equal the desired gradient even though any one minibatch is noisy. Stochastic
gradient descent follows these noisy directions repeatedly. In RL, the match
is delicate: trajectories come from a policy, adjacent steps are correlated,
and reusing old data changes the distribution. Algorithm design is partly the
art of constructing a gradient estimator whose bias and variance remain
manageable.

### Gradient clipping and parameter update are not action clipping

Global gradient-norm clipping rescales a parameter gradient when
$\lVert g\rVert_2$ exceeds threshold $c$:

```math
g_{used}=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right).
```

This limits an optimizer step caused by an unusually large batch. It does not
bound the robot action, motor current, or next network output. Action bounds,
target rate limits, and hardware supervisors are separate layers.

## 2.10 Normalization and numerical scale

Suppose one observation ranges around 0.01 and another around 1000. The same
weight scale treats them very differently. Observation normalization maintains
running mean $\mu$ and variance $\sigma^2$, then computes approximately

```math
\hat{o}=\frac{o-\mu}{\sqrt{\sigma^2+\epsilon}}.
```

$\epsilon$ is a small constant that avoids division by zero. Normalization can
improve optimization, but it becomes part of the learned input contract. If
training uses $\hat{o}$ and deployment feeds raw $o$, the actor receives a
different problem. That is why Microduck's export path bakes the normalizer
into Open Neural Network Exchange (ONNX).

Reward scaling also matters. Multiplying every reward by 100 leaves the
mathematical optimal policy unchanged in an ideal tabular setting, but it
changes gradient magnitude, value targets, clipping interactions, and numerical
behavior in a practical deep-RL implementation.

### Numerical stability is part of the algorithm

Mathematically equivalent expressions need not behave equally in finite
precision. For example, directly computing $\log(e^{x_1}+e^{x_2})$ can
overflow for large $x_i$. The stable log-sum-exp identity subtracts the
maximum $m$:

```math
\log\sum_i e^{x_i}
=m+\log\sum_i e^{x_i-m},
\qquad m=\max_i x_i.
```

Likewise, policy code stores `log_std`, adds epsilon before square roots, and
often computes probability ratios from differences of log-probabilities:

```math
\frac{\pi_{new}(a\mid s)}{\pi_{old}(a\mid s)}
=\exp[\log\pi_{new}(a\mid s)-\log\pi_{old}(a\mid s)].
```

These are not cosmetic implementation tricks. A not-a-number (NaN) value in
one parallel environment can contaminate normalization statistics and then an
entire policy update.

## 2.11 Bias and variance: a recurring tradeoff

An estimator has **bias** when its average estimate is systematically shifted.
It has **variance** when estimates fluctuate strongly across samples.

- Monte Carlo return uses a complete sampled future: low modeling bias, often
  high variance.
- A one-step temporal-difference target bootstraps from a learned estimate:
  potentially more bias, usually lower variance.
- Generalized Advantage Estimation introduces $\lambda$ to move along this
  tradeoff.

“Unbiased” does not automatically mean “better.” A very noisy gradient may
need so much data that a slightly biased, lower-variance estimator learns more
reliably.

### Approximation, estimation, and optimization error

When a learned policy fails, separate three sources:

1. **approximation error**: the chosen network/function family cannot
   represent the needed mapping;
2. **estimation error**: finite, noisy, or poorly covered data cannot identify
   the best representable mapping; and
3. **optimization error**: the optimizer did not find good parameters even for
   the available data and model class.

Adding a larger network targets approximation error but can worsen estimation
or optimization. Collecting diverse resets targets coverage, not network
capacity. Lowering a learning rate targets optimization dynamics, not a missing
camera input. This decomposition prevents “tune the neural network” from
becoming a universal but untestable explanation.

## 2.12 A tiny gradient check

For a differentiable scalar function, compare the analytic derivative with a
finite difference:

```python
def f(w: float) -> float:
    return (w - 4.0) ** 2

w = 1.0
eps = 1e-5
finite_difference = (f(w + eps) - f(w - eps)) / (2.0 * eps)
analytic = 2.0 * (w - 4.0)
print(finite_difference, analytic)
```

They should be close to $-6$. Finite differences are slow in large networks,
but the idea is valuable for checking a new reward derivative or small custom
operation.

## 2.13 Exercises

1. A batch has shape `(24, 4096, 61)`. How many observation scalars does it
   contain?
2. A Gaussian policy has $\mu=0.3$ and $\sigma=0.1$. Is action 0.31 more or less
   typical than action 0.8? Why?
3. Compute one gradient-descent step for $L(w)=(w+2)^2$, starting at $w=1$
   with learning rate 0.25.
4. Why can two policies with equal expected return have different hardware
   safety risk?
5. What breaks if an exported policy receives joint angles in degrees while
   training used radians?
6. Explain backpropagation without using the phrase “the computer learns.”
7. Why does an observation normalizer belong in the deployment contract?
8. A linear layer maps 61 inputs to 512 outputs. Derive its parameter count,
   including bias. What is the output batch shape for 4,096 observations?
9. Compute the dot product and Euclidean norm of $x=[3,4]$. Give one robot
   interpretation of each.
10. A return distribution is $0$ with probability 0.1 and $10$ with
    probability 0.9. Compute its expectation, variance, and standard deviation.
    Why does the expectation alone hide an important fact?
11. For a scalar Gaussian with $\mu=0$, $\sigma=0.5$, and sampled action
    $a=0.25$, compute $\partial\log\pi/\partial\mu$. If advantage is positive,
    which way will a policy-gradient update tend to move $\mu$?
12. Assuming independent samples, by what factor must the sample count grow to
    reduce standard error by a factor of three? Why may robot time steps not
    satisfy the independence assumption?
13. A gradient is $g=[6,8]$ and global norm threshold is 5. Compute the clipped
    gradient. Does this bound the motor command?
14. Classify each proposed fix by its primary target: add history for hidden
    backlash; collect more rough-terrain resets; widen a network that cannot
    represent a discontinuous gate; lower a diverging optimizer step.

The small bandit example in
[`examples/bandit_incremental_mean.py`](examples/bandit_incremental_mean.py)
uses expectation, sampling, and an incremental mean without any neural network.
Run it before continuing with
[Bellman methods from tables to a Deep Q-Network
(DQN)](03_bellman_and_value_learning.md).

## 2.14 Folded solutions

<details>
<summary>Show answers to Section 2.13</summary>

1. Multiply every axis:

   ```math
   24\times4096\times61=5{,}996{,}544
   ```

   These are scalar entries, not independent trajectories; the batch still has
   24 correlated time positions per environment.
2. Action 0.31 is only $0.01/0.1=0.1$ standard deviations from the mean. Action
   0.8 is $(0.8-0.3)/0.1=5$ standard deviations away, so 0.31 is vastly more
   typical under that Gaussian.
3. The gradient at $w=1$ is $2(w+2)=6$. Gradient descent gives
   $w_{new}=1-0.25(6)=-0.5$. A direct check is:

   ```python
   w = 1.0
   learning_rate = 0.25
   gradient = 2.0 * (w + 2.0)
   w -= learning_rate * gradient
   assert w == -0.5
   ```

4. Equal means do not imply equal distributions. One policy may score
   consistently while another mixes excellent trials with rare destructive
   failures. Report quantiles, failure categories, interventions, and raw
   trials in addition to expected return.
5. One radian is about 57.3 degrees. Feeding degree-valued numbers into a model
   trained on radians changes its input scale and distribution, often driving
   normalized features far outside training experience and producing unsafe
   actions.
6. Backpropagation applies the chain rule from the final loss toward earlier
   operations. It multiplies each downstream sensitivity by each local
   derivative, accumulating how a small change in every weight would change
   the loss. The optimizer then uses those gradients to update the weights.
7. The actor learned a function of normalized observations, not raw sensor
   numbers. Mean, variance, clipping, epsilon, order, and units therefore form
   part of the deployed function and must travel with or be embedded in the
   exported model.
8. There are $61\times512=31{,}232$ weights and 512 biases, for 31,744
   parameters. A batch $X\in\mathbb{R}^{4096\times61}$ produces
   $Z\in\mathbb{R}^{4096\times512}$.
9. The dot product with itself is $x^Tx=3^2+4^2=25$; the norm is
   $\sqrt{25}=5$. A dot product between unit body-up and world-up measures
   orientation alignment. A norm can measure the magnitude of a velocity or
   tracking-error vector.
10. The mean is $0.1(0)+0.9(10)=9$. The variance is

    ```math
    0.1(0-9)^2+0.9(10-9)^2=8.1+0.9=9,
    ```

    so the standard deviation is 3. The mean of 9 conceals a 10% complete-
    failure rate, which could be unacceptable on hardware.
11. The score is $(a-\mu)/\sigma^2=0.25/0.25=1$. With positive advantage,
    gradient ascent tends to increase the action's log-probability, moving the
    mean upward toward the sampled value (subject to other samples/loss terms).
12. Standard error scales as $1/\sqrt{N}$. Reducing it by three requires
    $3^2=9$ times as many independent samples. Consecutive robot steps share
    state, command, terrain, and episode conditions, so they are correlated;
    the effective independent sample count is smaller than the transition
    count.
13. The original norm is $\sqrt{6^2+8^2}=10$. Scale by $5/10$ to obtain
    $g_{used}=[3,4]$. This bounds the parameter-update gradient norm only. Motor
    commands require their own action transform, range/rate limits, and safety
    checks.
14. History primarily targets partial observability/representation;
    rough-terrain resets target estimation coverage; widening the network
    targets approximation capacity; lowering the step targets optimization
    stability. Each diagnosis should still be tested because the categories
    can interact.

</details>
