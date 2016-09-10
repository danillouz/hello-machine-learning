# Support Vector Machine
SVM is yet another type of supervised machine learning
algorithm. It is sometimes cleaner and more powerful.

A useful way to think about SVMs is to think of them as
Large Margin Classifiers.

```
If y = 1, we want ΘTx ≥ 1 (not just ≥ 0)
If y = 0, we want ΘTx ≤ −1 (not just < 0)
```

The distance of the decision boundary to the nearest example
is called the _margin_. Since SVMs maximize this margin, it
is often called a Large Margin Classifier.

Increasing and decreasing `C` is similar to respectively
decreasing and increasing `λ`, and can simplify our decision
boundary.

# Kernels
Kernels allow us to make complex, non-linear classifiers
using Support Vector Machines.

# Logistic Regression vs SVMs
- If `n` is large (relative to `m`), then use logistic regression, or SVM without a kernel (the linear kernel).
- If `n` is small and `m` is intermediate, then use SVM with a Gaussian Kernel.
- If `n` is small and `m` is large, then manually create/add more features, then use logistic regression or SVM without a kernel.
