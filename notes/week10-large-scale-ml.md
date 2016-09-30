# Learning with Large Datasets
We mainly benefit from a very large dataset when our
algorithm has high variance when `m` is small. Recall that
if our algorithm has _high bias_, more data will **not**
have any benefit.

Datasets can often approach such sizes as `m = 100,000,000`.
In this case, our gradient descent step will have to make a
summation over all one hundred million examples. We will
want to try to avoid this.

# Stochastic Gradient Descent
Stochastic gradient descent is an alternative to classic
(or batch) gradient descent and is more efficient and
scalable to large data sets.

This algorithm will only try to fit one training example
at a time. This way we can make progress in gradient
descent without having to scan all `m` training examples
first.

Stochastic gradient descent will be unlikely to converge
at the global minimum and will instead wander around it
randomly, but usually yields a result that is close
enough.

Stochastic gradient descent will usually take `1-10`
passes through your data set to get near the global
minimum.

# Mini-Batch Gradient Descent
Mini-batch gradient descent can sometimes be even faster
than stochastic gradient descent. Instead of using all `m`
examples as in batch gradient descent, and instead of
using only 1 example as in stochastic gradient descent, we
will use some in-between number of examples `b`.

# Stochastic Gradient Descent Convergence
How do we choose the learning rate `α` for stochastic
gradient descent? Also, how do we debug stochastic
gradient descent to make sure it is getting as close as
possible to the global optimum?

One strategy is to plot the average cost of the hypothesis
applied to every `1000` or so training examples. We can
compute and save these costs during the gradient descent
iterations.

With a smaller learning rate, it is possible that you may
get a slightly better solution with stochastic gradient
descent. That is because stochastic gradient descent will
oscillate and jump around the global minimum, and it will
make smaller random jumps with a smaller learning rate.

If you increase the number of examples you average over to
plot the performance of your algorithm, the plot's line
will become smoother.

With a very small number of examples for the average, the
line will be too noisy and it will be difficult to find
the trend.

One strategy for trying to actually converge at the global
minimum is to slowly decrease `α` over time. For example:
```
α = const1 / (iterationNumber + const2)
```

However, this is not often done because people don't want
to have to fiddle with even more parameters.

# Online Learning
With a continuous stream of users to a website, we can run
an endless loop that gets `(x,y)`, where we collect some
user actions for the features in `x` to predict some
behavior `y`.

You can update `θ` for each individual `(x,y)` pair as you
collect them. This way, you can adapt to new pools of
users, since you are continuously updating theta.

# Map Reduce and Data Parallelism
We can divide up batch gradient descent and dispatch the
cost function for a subset of the data to many different
machines so that we can train our algorithm in parallel.

Your learning algorithm is _MapReduceable_ if it can be
expressed as computing sums of functions over the training
set. Linear regression and logistic regression are easily
parallelizable.

For neural networks, you can compute forward propagation
and back propagation on subsets of your data on many
machines. Those machines can report their derivatives back
to a _master_ server that will combine them.
