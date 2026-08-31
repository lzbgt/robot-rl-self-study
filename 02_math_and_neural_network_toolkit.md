# 2. Math and Neural-Network Toolkit

Deep reinforcement learning combines probability, calculus, linear algebra,
and optimization. You do not need to master each subject before beginning. You
do need to know what the symbols mean, what shape each quantity has, and what
an equation asks the computer to do.

This chapter builds that minimum toolkit. Keep a pencil nearby: predicting a
number before running code is much more educational than only reading it.

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

</details>
