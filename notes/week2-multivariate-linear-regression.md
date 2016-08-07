# Model: Linear Regression with multiple variables
Linear regression with multiple variables is also known as
**Multivariate linear regression** and its hypothesis can be
represented as:

`hθ(x) = θ0 + θ1x1 + θ1x2 + ... + θ1xn`

# Feature Normalization
When dealing with problems that have multiple features, e.g.
_multivariate linear regression_, the _gradient descent_ can
converge faster in a lot fewer iterations when the different
features take on similar ranges of values. **Reason being
that fewer iterations are required.**

Two techniques to help with this are:
- **Feature scaling**; divide the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1.
- **Mean normalization**; subtract the average value for an input variable from the values for that input variable, resulting in a new average value for the input variable of just zero.

To implement both of these techniques, adjust your input
values as shown in this formula:

```
xi := (xi− μi) / si
```

- `xi`: feature
- `μi`: average input value for a feature
- `si`: input value range, i.e. max - min or standard deviation

For example when estimating the price of a house where you
have a feature `xi` that captures the age in a training set
where every house has an age between 30 and 50 years (`si`),
with an average `μi` of 38 years:

```
xi = (age of house - 38) / (50 - 30)
```

# Debugging
Plot `J` of `θ` as a function of the number of iterations
to make sure _gradient descent_ is working correctly by
seeing a decreasing line.

If, for example the line is _increasing_, this means a
smaller _learning rate_ `α` should be used. However note
that if `α` is _too small_, gradient descent might be _slow_
to converge.

# Polynomial Regression
Allows to use a quadratic, cubic or square root function as
a model to better fit your training data.

# Normal Equation
The _Normal Equation_ is a method of finding the optimum
_theta_ without iteration:
```
θ=(X^T * X)^−1 X^T * y
```

_There is no need to do feature scaling with the normal
equation._

| GRADIENT DESCENT | NORMAL EQUATION |
| :---: | :---: |
| Needs `α` | Does **not** need `α` |
| Many iterations are required | **No** iteration are required |
| Fast for large `n` | Slow for large `n` |

With the normal equation, computing the inversion has
complexity O(n3). For a very large number of features `n`,
the normal equation will be slow. In practice, when `n`
exceeds `10.000` it might be a good to use an iterative
process.

# Octave tutorial
- comments: `%`
- not equals: `~=`
- suppress output: `;`

## Print Commands
```octave
sprintf('hallo %0.2f', 2.2222222) 		% ans = hallo 2.22

disp(sprintf('hallo %0.4f', 2.2222222)  % hallo 2.2222
```

## Vectors and Matrices
```octave
% 4 x 2 Matrix
A = [ 1 2; 3 4; 5 6; 7 8; ]

% 3 x 1 vector
v = [ 1; 2; 3 ]

% ones(rows, columns)
B = ones(1, 3) % 1 1 1

% Identity Matrix
eye(4) % 4 x 4 Identity Matrix
```

- get the size of a Matrix: `size(Matrix)`
- get the size of a vector: `length(vector)`

## Loading, displaying, moving and computing data
- load a file as variable: `load featureX.dat`
- print workspace data: `who` or `whos`
- clear data: `clear featureX`
- save data to a file: `save hello.mat v`
	* make it human readable with `-ascii`
- Access Matrix element by index: `A(row, column);`
	* use `:` to get all: `A(:,column);`
	* use `:` to get `from:to`: `A(1:3, 1:2)`
- Transpose Matrix: `A';`
- Inverse Matrix: `pinv(A);`

## Plotting data
- plot a function: `plot(x, h);`
- plot multiple functions: `hold on`
- name x-axis: `xlabel(string);`
- name y-axis: `ylabel(string);`
- add legend: `legend(string, string, ..);`
- add title: `title(string);`
- save image: `print -dpng 'image.png'`
- divide plot in 1x2 grid:
	* `subplot(1, 2, 1); plot(t, y1);`
	* `subplot(1, 2, 2); plot(t, y2);`
- set axis range: `axis([ x-range-from x-range-to y-range-from y-range-to])`
- clear figure: `clf`

# Cost function J implementation
```
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% Prediction vector 'h' contains all of the hypothesis
% values. One for each training example, i.e. for each
% row of X, which is the product of X and theta
h = X * theta;

% Error vector computes the difference between the
% hypothesis and y, i.e. % the error for each training
% example
e = h - y;

% Squared error vector
eSqr = e.^2;

% Summed squared error vector
eSum = sum(eSqr)

% Calc the Cost value J
J = 1 / (2 * m) * eSum;

end
```

# Gradient Descent with one variable implementation
```octave
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

	% The hypothesis is a vector, formed by multiplying the
	% X matrix and the theta vector. X has size (m x n),
	% and theta is (n x 1), so the product is (m x 1). That's
	% good, because it's the same size as 'y'. Call this hypothesis
	% vector 'h'.
	h = X * theta;

	% The "errors vector" is the difference between the 'h'
	% vector and the 'y' vector.
	e = h - y;

	% The change in theta (the "gradient") is the sum of the
	% product of X and the "errors vector", scaled by alpha
	% and 1/m. Since X is (m x n), and the error vector is
	% (m x 1), and the result you want is the same size as
	% theta (which is (n x 1), you need to transpose X before
	% you can multiply it by the error vector.
	eSum = X' * e;
	gradient = alpha * (1 / m) * eSum;

	% Subtract this "change in theta" from the original value
	% of theta.
	theta = theta - gradient;

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
```
