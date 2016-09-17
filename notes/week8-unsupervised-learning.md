# Unsupervised Learning
Unsupervised learning is contrasted from supervised learning
because it uses an unlabeled training set rather than a
labeled one.

# K-Means Algorithm
The K-Means Algorithm is the most popular and widely used
algorithm for automatically grouping data into coherent
subsets:

1. Randomly initialize two points in the dataset called the cluster centroids.
2. Cluster assignment; assign all examples into one of two groups based on which cluster centroid the example is closest to.
3. Move centroid; compute the averages for all the points inside each of the two cluster centroid groups, then move the cluster centroid points to those averages.
4. Re-run (2) and (3) until we have found our clusters.

# Random Initialization
There's one particular recommended method for randomly
initializing your cluster centroids:

1. Have `K < m`. That is, make sure the number of your clusters is less than the number of your training examples.
2. Randomly pick `K` training examples.
3. Set `μ1, ... μk` equal to these `K` examples.

_K-means can get stuck in local optima. To decrease the
chance of this happening, you can run the algorithm on many
different random initializations. In cases where `K < 10` it
is strongly recommended to run a loop of random
initializations._

# Choosing the Number of Clusters
Choosing `K` can be quite arbitrary and ambiguous.

## The elbow method
Plot the cost `J` and the number of clusters `K`. The cost
function should reduce as we increase the number of clusters,
and then flatten out. Choose `K` at the point where the cost
function starts to flatten out.

However, fairly often, the curve is very gradual, so there's
no clear elbow.

Note: `J` will always decrease as `K` is increased. The one
exception is if K-means gets stuck at a bad local optimum.

Another way to choose `K` is to observe how well k-means
performs on a downstream purpose. In other words, you choose
`K` that proves to be most useful for some goal you're trying
to achieve from using these clusters.

# Dimensionality Reduction

## Data Compression
We may want to reduce the dimension of our features if we
have a lot of redundant data.

To do this, we find two highly correlated features, plot
them, and make a new line that seems to describe both
features accurately. We place all the new features on this
single line.

Doing dimensionality reduction will reduce the total data we
have to store in computer memory and will speed up our
learning algorithm.

_In dimensionality reduction, we are reducing our features
rather than our number of examples. Our variable `m` will
stay the same size; `n`, the number of features each example
from `x(1)` to `x(m)` carries, will be reduced._

## Visualization
It is not easy to visualize data that is more than three
dimensions.

We can reduce the dimensions of our data to 3 or less in
order to plot it.

We need to find new features, `z1, z2` (and perhaps `z3`)
that can effectively summarize all the other features.

# Principal Component Analysis
The most popular dimensionality reduction algorithm is PCA.

## Problem formulation
Given two features, `x1` and `x2`, we want to find a single
line that effectively describes both features at once.
We then map our old features onto this new line to get a new
single feature.

The goal of PCA is to reduce the average of all the distances
of every feature to the projection line. This is the
_projection error_.

## PCA is not linear regression
In linear regression, we are minimizing the squared error
from every point to our predictor line. These are vertical
distances.

In PCA, we are minimizing the shortest distance, or shortest
orthogonal distances, to our data points.

More generally, in linear regression we are taking all our
examples in `x` and applying the parameters in `Θ` to
predict `y`.

In PCA, we are taking a number of features `x1, x2, ... xn`,
and finding a closest common dataset among them. We aren't
trying to predict any result and we aren't applying any
theta weights to the features.

# Advice for Applying PCA
The most common use of PCA is to speed up supervised learning.

_Bad use of PCA: trying to prevent overfitting._
