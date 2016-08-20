# Non-linear Hypotheses
Performing linear regression with a complex set of data with
many features is very unwieldy.

We can approximate the growth of the number of new features
we get with all quadratic terms with `O(n^2/2)`. And if you
wanted to include all cubic terms in your hypothesis, the
features would grow asymptotically at `O(n^3)`.

Neural networks offers an alternate way to perform ML when
we have complex hypotheses with many features.

# Neural Networks
Neural networks are limited imitations of how our own brains
work. They've had a big recent resurgence because of advances
in computer hardware.

There is evidence that the brain uses only _one learning
algorithm_ for all its different functions.

# Model Representation
At a very simple level, _neurons_ are basically computational
units that take _input (dendrites)_ as electrical input
(called spikes) that are channeled to _outputs (axons)_.

In neural networks, we use the same logistic function as in
classification:

```
1 / (1 + e^−θT * x)
```

However, in neural networks, this is sometimes called the
_sigmoid (logistic) activation function_. And the `θ`
parameters are sometimes instead called _weights_.

The _first layer_ is called the _input layer_ and the _final
layer_ the _output layer_, which gives the final value
computed on the hypothesis.

We can have _intermediate layers_ of nodes between the input
and output layers called the _hidden layer_. These are labeled
as `a` superscript `j` (layer) subscript `i` (unit) and called
_activation units_. Where `Θ` is the matrix of weights
controlling function mapping from layer `j` to layer `j + 1`.

If network has `sj` units in layer `j` and `sj + 1` units in
layer `j + 1`, then `Θ(j)` will be of dimension:
```
sj + 1 * (sj + 1)
```
