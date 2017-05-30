Optimization Methods
========

Stable Linear-time optimization in APT models
--------

The asset return is modeled as  
> R = X*f + e

where 
* R is n-dimensional vector containing returns in excess of the risk-free rate.
* X is a n*p matrix that is known as factors.
* f is coefficients corresponding to given factors, this vector is not observable
	and must be obtained via statistical inference. ( The estimator of f
	is called __factor return__)
* e is error term with mean zero and covariance matrix D.

#### Goal
> 1. Solve f from above model, which is well-defined and numerically stable 
	in the presence of colinearity in X. Also the computational complexity
	should be linear with repect to the number of assets.
> 2. Use Markowitz mean-variance optimization to find optimal portfolio.
> 3. Extend with trading costs.

#### Solution
1. If X'X is invertible, the model is said to be __identifiable__. We call a model 
	__barely identifiable__, if X`X is close to be violated.
	
	* For identifiable model, factor return can be estimated by OLS.
	* For unidentifiable model, factor return can be estimated by Ridge Regression.

2. The Markowitz problem is defined as 
	> min f(h) = exposures term + idiosyncratic variance

	Intuitively, an optimal portfolio h must minimize the idiosyncratic variance.
	Based on this intuition, an explicit formula is given with the help of Moore-
	Penrose pseudoinverse.
	The computational complexity proved to be linear in the number of assets.
	Moreover, this number is about 
	> 6np^2 + 20p^3 flops

3. To include certain simple trading cost given in quadratic from of portfolio h,
	the following problem need to be solved
	> min f(h) = exposures term + idiosyncratic variance + trading cost
	
	which has the same mathematical structure as original problem in 1. Thus,
	a similar approach can be used to derive a explicit formula for the optimal 
	portfolio.

#### Remarks
1. This model works well on both identifiable and unidentifiable cases.
2. The complexity scales as p^3 for fixed n.
3. Under certain assumptions on return predictability and trading costs,
	the dynamically optimal portfolio sequence is given by a linear combination of
	past optimal portfolios and the "aim portfolio" that are weighted sum of 
	future Markowitz portfolios.
4. If the alpha factors are statistically indenpendent from risk factors,
	then a portfolio __neutral__ to all risk factors can be obtained.



