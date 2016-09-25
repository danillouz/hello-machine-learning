# Anomaly Detection
We define a model `p(x)` that tells us the probability the
example is not anomalous. We also use a threshold `ϵ`
(epsilon) as a dividing line so we can say which examples
are anomalous and which are not.

A very common application of anomaly detection is detecting
fraud:

- `x(i)` are the features of user `i`s activities
- Model `p(x)` from the data
- Identify unusual users by checking which have `p(x) < ϵ`

_If our anomaly detector is flagging too many anomalous
examples, then we need to decrease our threshold `ϵ`._

# Developing and Evaluating an Anomaly Detection System
To evaluate our learning algorithm, we take some labeled
data, categorized into anomalous and non-anomalous examples:
```
y = 0 if normal
y = 1 if anomalous
```

Among that data, take a large proportion of good,
non-anomalous data for the training set on which to train
`p(x)`.

Then, take a smaller proportion of mixed anomalous and
non-anomalous examples (you will usually have many more
non-anomalous examples) for your cross-validation and test
sets.

For example, we may have a set where `0.2%` of the data is
anomalous. We take `60%` of those examples, all of which are
good, `y=0`, for the training set. We then take `20%` of the
examples for the cross-validation set (with `0.1%` of the
anomalous examples) and another `20%` from the test set
(with another `0.1%` of the anomalous).

In other words, we split the data `60/20/20` (training/CV/test)
and then split the anomalous examples `50/50` between the CV
and test sets.

## Algorithm evaluation
Fit model `p(x)` on training set `{ x(1), … , x(m) }`.

On a cross validation/test example `x`, predict:
```
If p(x) < ϵ (anomaly), then y = 1

If p(x) ≥ ϵ (normal), then y = 0
```

Possible evaluation metrics:
- True positive, false positive, false negative, true negative
- Precision/recall
- F1 score

_Note that we use the cross-validation set to choose
parameter `ϵ`._

# Anomaly Detection vs. Supervised Learning
Use anomaly detection when:
- We have a very small number of positive examples (`y=1 ... 0-20` examples is common) and a large number of negative (`y=0`) examples.
- We have many different types of anomalies and it is hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous examples we've seen so far.

Use supervised learning when:
- We have a large number of both positive and negative examples. In other words, the training set is more evenly divided into classes.
- We have enough positive examples for the algorithm to get a sense of what new positives examples look like. The future positive examples are likely similar to the ones in the training set.

# Choosing What Features to Use
The features will greatly affect how well your anomaly
detection algorithm works.

We can check that our features are gaussian by plotting a
histogram of our data and checking for the bell-shaped curve.

Some transforms we can try on an example feature `x` that
does not have the bell-shaped curve are:
- `log(x)`
- `log(x+1)`
- `log(x+c)` for some constant
- `x√`
- `x1/3`

There is an error analysis procedure for anomaly detection
that is very similar to the one in supervised learning.

Our goal is for `p(x)` to be large for normal examples and
small for anomalous examples.

One common problem is when `p(x)` is similar for both types
of examples. In this case, you need to examine the anomalous
examples that are giving high probability in detail and try
to figure out new features that will better distinguish the
data.

In general, choose features that might take on unusually
large or small values in the event of an anomaly.

# Content Based Recommendations
We can introduce two features, `x1` and `x2` which represents
how much romance or how much action a movie may have (on a
scale of 0−1).

One approach is that we could do linear regression for every
single user.

# Collaborative Filtering
It can be very difficult to find features such as "amount of
romance" or "amount of action" in a movie. To figure this
out, we can use feature finders.

We can let the users tell us how much they like the different
genres, providing their parameter vector immediately for us.
