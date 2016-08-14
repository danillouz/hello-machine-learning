# Classification
Examples of classification problems are:

- Email: spam, not spam
- Online transactions: fraudulent, not fraudulent

With a **Binary Classification Problem**, the output
`y` can take on 2 values:

- `0`: _Negative Class_ (absence)
- `1`: _Positive Class_ (presence)

With a **Multi-class Classification Problem**, the output
`y` can take on more than 2 values.

**Logistic Regression** is a classification algorithm that
is applied to settings where the output `y` is a discrete
value, i.e. either `1` or `0`.

# Hypothesis Representation
The hypothesis should satisfy:
```
0 ≤ hθ(x) ≤ 1
```

The new form uses the _Sigmoid Function_, also called the
_Logistic Function_ which can be described as:

```
hθ(x) = g(θTx)

z = θTx

g(z) = 1 / (1 + e^−z)
```

The function `g(z)`, maps any real number to the (`0`, `1`)
interval, making it useful for transforming an arbitrary
valued function into a function better suited for
classification.

`hθ` will give the probability that the output is `1`.
For example, `hθ(x)=0.7` gives the probability of `70%`
that the output is 1.

```
hθ(x) = P(y = 1 | x ;θ)

hθ(x) = 1 − P(y = 0 ∣x ;θ)
```

# Decision Boundary
The decision boundary is the line that separates the area
where `y = 0` and where `y = 1`. It is created by our
hypothesis function.

```
hθ(x) ≥ 0.5 → y = 1
hθ(x) < 0.5 → y = 0
```

The way our logistic function `g` behaves is that when its
input is greater than or equal to zero, its output is greater
than or equal to 0.5:

```
g(z) ≥ 0.5
	when
z ≥ 0

hθ(x) = g(θTx) ≥ 0.5
	when
θTx ≥ 0

θTx ≥ 0 ⇒ y = 1
θTx < 0 ⇒ y = 0
```

# Cost Function
We cannot use the same cost function that we use for linear
regression because the _Logistic Function_ will cause the
output to be wavy, causing many local optima. In other words,
it will not be a convex function.

Instead, the _cost function_ for _logistic regression_ looks
like:

```
if y = 1
	Cost(hθ(x),y) = −log(hθ(x))

if y = 0
	Cost(hθ(x),y) = −log(1−hθ(x))
```

The more our hypothesis is _off_ from y, the _larger_ the
cost function output.

If `hθ(x) = y`, i.e. ff our hypothesis is equal to `y`, then
our cost is `0`:
```
Cost(hθ(x), y) = 0
```

If `y = 0`, i.e. if our correct answer `y` is `0`, then the
cost function will be `0` if our hypothesis function also
outputs `0`. If our hypothesis approaches `1`, then the cost
function will approach infinity:
```
Cost(hθ(x), y) → ∞

and

hθ(x) → 1
```

If `y = 1`, i.e. if our correct answer `y` is `1`, then the
cost function will be `0` if our hypothesis function outputs
`1`. If our hypothesis approaches `0`, then the cost function
will approach infinity:
```
Cost(hθ(x), y) → ∞

and

hθ(x) → 0
```

# Simplified Cost Function
```
Cost(hθ(x), y) = −y * log(hθ(x)) − (1 − y) * log(1 − hθ(x))
```

A vectorized implementation is:
```
h = g(Xθ)

J(θ) = 1/m * ( −yT log(h) − (1 − y)T log(1 − h) )
```

# Gradient Descent
```
θ := θ − α/m XT ( g(X * θ) − y )

where y is a vector and X and θ are Matrices
```

# Multi-class Classification: One-vs-all
We are basically choosing one class and then lumping all the
others into a single second class. We do this repeatedly,
applying binary logistic regression to each case, and then
use the hypothesis that returned the highest value as our
prediction.

# Regularization: the problem of overfitting
**Underfitting** or **high bias** is the situation where the
hypothesis algorithm **doesn't fit** the training data very
well. It is usually caused by a function that is too simple
or uses too few features.

**Overfitting** or **high variance** is the situation where
the hypothesis algorithm **fits** the training data very
well, but **doesn't generalize** well to predict new data.
It is usually caused by a complicated function that creates
a lot of unnecessary curves and angles unrelated to the data.

_This terminology is applied to both linear and logistic
regression._

There are two main options to address the issue of overfitting:

1. Reduce the number of features.
	* Manually select which features to keep.
	* Use a model selection algorithm.
2. Regularization
	* Keep all the features, but reduce the parameters `θj`.

_Regularization works well when we have a lot of slightly
useful features._

# Regularization: Cost Function
If the hypothesis function causes overfitting, we can
reduce the weight that some of the terms in our function
carry, by increasing their cost, i.e. increasing `θ`
parameters.

The `λ` is the regularization parameter, which determines
how much the costs of the `θ` parameters are inflated.

If `λ` is chosen to be too large, it may smooth out the
function too much and cause underfitting.
