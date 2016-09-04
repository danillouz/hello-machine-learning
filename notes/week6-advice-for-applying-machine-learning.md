# Evaluate a learning algorithm
Errors in your predictions can be troubleshooted by:

- Getting more training examples.
- Trying smaller sets of features.
- Trying additional features.
- Trying polynomial features.
- Increasing or decreasing `λ`.

_Don't just pick one of these avenues at random_.

# Evaluate a hypothesis
A hypothesis may have low error for the training examples but
still be inaccurate (overfitting).

With a given dataset of training examples, we can split up
the data into two sets, i.e. a _training set_ and a _test
set_.

The new procedure using these two sets is then:

- Learn `Θ` and minimize `Jtrain(Θ)` using the training set.
- Compute the test set error `Jtest(Θ)`.

# Model Selection
Just because a learning algorithm fits a training set well,
that does not mean it is a good hypothesis.

The error of your hypothesis as measured on the data set with
which you trained the parameters will be lower than any other
data set.

In order to choose the model of your hypothesis, you can test
each degree of polynomial and look at the error result.

## Without the Validation Set
_This is a bad method, do not use it._

1. Optimize the parameters in `Θ` using the training set for each polynomial degree.
2. Find the polynomial degree `d` with the least error using the test set.
3. Estimate the generalization error also using the test set with `Jtest( Θ(d) )`.

In this case, we have trained one variable `d`, or the degree
of the polynomial, using the test set. This will cause our
error value to be greater for any other set of data.

## Use of the CV set
To solve this, we can introduce a third set, the Cross
Validation Set, to serve as an intermediate set that we can
train `d` with. Then our test set will give us an accurate,
non-optimistic error.

One example way to break down our dataset into the three
sets is:

- Training set: 60%
- Cross validation set: 20%
- Test set: 20%

We can now calculate three separate error values for the
three different sets.

## With the Validation Set
_This method presumes we do not also use the CV set for
regularization._

1. Optimize the parameters in `Θ` using the training set for each polynomial degree.
2. Find the polynomial degree `d` with the least error using the cross validation set.
3. Estimate the generalization error using the test set with `Jtest( Θ(d) )`.

This way, the degree of the polynomial `d` has not been
trained using the test set.

# Diagnosing Bias vs. Variance
The training error will tend to decrease as we increase the
degree `d` of the polynomial.

At the same time, the cross validation error will tend to
decrease as we increase `d` up to a point, and then it will
increase as `d` is increased, forming a convex curve.

## High bias; underfitting
_high training error_ and _high cross validation error_:

```
JCV(Θ) ≈ Jtrain(Θ)
```

## High variance; overfitting
_low training error_ and _high cross validation error_:

```
JCV(Θ) >> Jtrain(Θ)
```

# Regularization and Bias/Variance
Instead of looking at the degree `d` contributing to
bias/variance, now we will look at the regularization
parameter `λ`:

- Large `λ` means high bias (underfitting); both `Jtrain(Θ)` and `JCV(Θ)` will be high.

- Intermediate `λ` means 'just right'; `Jtrain(Θ)` and `JCV(Θ)` are somewhat low and `Jtrain(Θ) ≈ JCV(Θ)`.

- Small `λ` means high variance (overfitting); `Jtrain(Θ)` is low and `JCV(Θ)` is high.

In order to choose the model and the regularization `λ`, we
need to:

1. Create a list of lambda.
2. Select a lambda to compute.
3. Create a model set like degree of the polynomial or others.
4. Select a model to learn `Θ`.
5. Learn the parameter `Θ` for the model selected, using `Jtrain(Θ)` with `λ` selected (this will learn `Θ` for the next step).
6. Compute the train error using the learned `Θ` (computed with `λ` ) on the `Jtrain(Θ)` without regularization or `λ = 0`.
7. Compute the cross validation error using the learned `Θ` (computed with `λ`) on the `JCV(Θ)` without regularization or `λ = 0`.
8. Do this for the entire model set and lambdas, then select the best combo that produces the lowest error on the cross validation set.
9. Now if you need visualize to help you understand your decision, you can plot to the figure with: `λ x Cost Jtrain(Θ)` and `λ x Cost JCV(Θ)`.
10. Now using the best combo `Θ` and `λ`, apply it on `Jtest(Θ)` to see if it has a good generalization of the problem.
11. To help decide the best polynomial degree and `λ` to use, we can diagnose with the learning curves.

# Learning Curves
If a learning algorithm is suffering from high bias, getting
more training data will not (by itself) help much.

If a learning algorithm is suffering from high variance,
getting more training data is likely to help.

# What to do when
- Getting more training examples
	* Fixes high variance
- Trying smaller sets of features
	* Fixes high variance
- Adding features
	* Fixes high bias
- Adding polynomial features
	* Fixes high bias
- Decreasing `λ`
	* Fixes high bias
- Increasing `λ`
	* Fixes high variance

# Diagnosing Neural Networks
- A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
- A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase `λ`) to address the overfitting.

_Using a single hidden layer is a good starting default. You
can train your neural network on a number of hidden layers
using your cross validation set_.

# Machine Learning System Design
Precision:

```
True Positives / Total number of predicted positives = True Positives / True Positives + False positives
```

Recall:

```
True Positives / Total number of actual positives = True Positives / True Positives + False negatives
```

# How much data should we train on
In certain cases, an inferior algorithm, if given enough
data, can outperform a superior algorithm with less data.

We must choose our features to have enough information. A
useful test is: _Given input x, would a human expert be able
to confidently predict y?_

Rationale for large data: if we have a low bias algorithm
(many features or hidden units making a very complex function),
then the larger the training set we use, the less we will
have overfitting (and the more accurate the algorithm will
be on the test set).
