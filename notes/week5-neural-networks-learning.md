# Cost Function
- L: total number of layers in the network
- sl: number of units (not counting bias unit) in layer l
- K: number of output units/classes

In neural networks, we may have many output nodes. We denote
`hΘ(x)k` as being a hypothesis that results in the _kth_
output.

# Backpropagation Algorithm
_Backpropagation_ is neural-network terminology for minimizing
our cost function, just like what we were doing with gradient
descent in logistic and linear regression.

That is, we want to minimize our cost function `J` using an
optimal set of parameters in theta.

# Gradient Checking
_Gradient checking_ will assure that our _backpropagation_
works as intended.

# Random Initialization
Initializing all theta weights to zero does not work with
neural networks. When we _backpropagate_, all nodes will
update to the same value repeatedly. instead we have to
randomly initialize our weigths.

# Putting it all together
First, pick a network architecture; choose the layout of
your neural network, including how many hidden units in each
layer and how many layers total:

- Number of input units; dimension of features `x(i)``
- Number of output units; number of classes
- Number of hidden units per layer; usually more are better (must balance with cost of computation as it increases with more hidden units)
- Defaults: 1 hidden layer. If more than 1 hidden layer, then the same number of units in every hidden layer.

Training a Neural Network:

1. Randomly initialize the weights
2. Implement forward propagation to get `hθ( x(i) )`
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on
every training example:

```
for i = 1:m,
   Perform forward propagation and backpropagation using example ( x(i), y(i) )
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```
