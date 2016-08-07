# What is Machine Learning?
> Arthur Samuel: "the field of study that gives computers
the ability to learn without being explicitly programmed."

> Tom Mitchell: "A computer program is said to learn from
experience E with respect to some class of tasks T and
performance measure P, if its performance at tasks in T, as
measured by P, improves with experience E."

In general, any ML problem can be assigned to one of two
broad classifications:

## Supervised Learning
An algorithm is given a data set in which the _right answers_
were given. This way we already know what our correct output
should look like, having the idea that there is a relationship
between the input and the output.

Supervised learning problems are categorized into:

### Regression problems
Trying to _predict_ results within a _continuous output_.
Meaning that we are trying to map input variables to some
continuous function.

> Regression: statistical process for estimating the
relationships among variables.

> **Continuous** data is **measured** and can take any value
within a range. For example a person's height.

### Classification problems
Trying to _predict_ results in a _discrete output_. Meaning
that we are trying to map input variables into discrete
categories.

> Classification: identifying to which of a set of categories
a new observation belongs, on the basis of a training set of
data containing observations whose category membership is
known.

> **Discrete** data is **counted** and can only take certain
values. For example the number of students in a class (you
can't have _half_ a student).

## Unsupervised Learning
Allows to approach problems with little or no idea what the
results should look like. Structure can be derived by
_clustering_ the data based on relationships among the
variables in the data.

With unsupervised learning there is no feedback based on the
prediction results, i.e., there is no _teacher_ to correct
you.

# Model: Linear Regression with one variable
Linear regression with one variable is also known as **univariate
linear regression**.

In regression problems, we are taking _input_ variables and
trying to fit the _output_ onto a _continuous_ expected result
function.

_Univariate linear regression is used when you want to predict
a single output value `y` from a single input value `x`._

## Hypothesis function
- m; number of training examples
- x; input
- y; output

`Training Set (m) -> Learning Algorithm -> hypothesis (h)`

`Input (x) -> h -> output (y)`

`hθ(x) = θ0 + θ1x`

## Cost function
The accuracy of the `h` function can be measured by using a
_cost function_ (`J`). This takes an average of all the results
of the hypothesis inputs compared to the actual outputs:

`J(θ0,θ1) = 1/2m m∑i=l (yi−yi)^2 = 1/2m m∑i=l (hθ(xi)−yi)^2`

The training data set is scattered on the `x-y` plane. We are
trying to make straight line (defined by `hθ(x)` function)
which passes through this scattered set of data. Our objective
is to choose a `θ0` and `θ1` so that our `hθ(x)` is close to
the `y` of our training data set.

# Gradient descent
Automatically improves the parameters of a hypothesis
function.

Imagine that we graph our `h` function based on its fields
`θ0` (x-axis), `θ1` (y-axis) and the cost function (z axis).
The points on the graph will be the result of the cost function
using the hypothesis with those specific theta parameters.

We will know that we have succeeded when our cost function is
at the very bottom of the pits in our graph, i.e. when its
value is the minimum.

The way we do this is by taking the derivative (the tangential
line to a function) of our cost function. The slope of the
tangent is the derivative at that point and it will give us
a direction to move towards. We make steps down that derivative
by the parameter `α`, called the learning rate.

The gradient descent algorithm is:

```
repeat until convergence {
	`θj := θj − α * (∂/∂ * θj) * J(θ0, θ1)`
}
```

Where `j=0,1` represents the feature index number. Intuitively,
this could be thought of as:

```
repeat until convergence {
	θj :=θj − α [slope of tangent aka derivative in j dimension]
}
```

## (Batch) Gradient Descent for Linear Regression
Each step of gradient descent uses **all** the training
examples.
