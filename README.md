
# Robust Adaptive Metropolis (RAM) Algorithm for parameter estimation in linear models

The Robust Adaptive Metropolis Algorithm dynamically adjusts its proposal distribution during sampling, facilitating better exploration of the target distribution and increasing acceptance rates. This adaptability improves efficiency by aligning proposals more effectively with the distribution's characteristics, offering advantages over the static proposal distributions of conventional Metropolis-Hastings methods.

## The model
A linear model can be expressed as\
$\huge{y = {X}{\theta} + \epsilon} $  
$\huge{\epsilon \sim N(0, \sigma^{2})}$ 

## The RAM algorithm

* Draw $\theta^{0}$ from an initial distribution $p_0(\theta)$, and initialize $S_0$ to be the lower triangular cholesky factor of the initial covariance matrix $\Sigma_0$.
* For MCMC samples $i = 1, 2, \ldots, N$, do:
    * Sample a candidate point\
      $\huge{\theta^* = \theta_{i-1} + S_{i-1}r_i }$\
      where\
      $\huge{r_i \sim N(0, 1)} $ 
    * Compute the acceptance probability $(\alpha)$:\
      $\huge{\alpha_i = min\\{1,R\\}}$\
      $\huge{R = \exp(\phi(\theta_{i-1}) - \phi(\theta^*))}$
      


## References

[1] [Särkkä S, Svensson L. Bayesian filtering and smoothing. Cambridge university press; 2023 May 31.](https://books.google.co.in/books?hl=en&lr=&id=utXBEAAAQBAJ&oi=fnd&pg=PP1&dq=bayesian+filtering+and+smoothing&ots=GX-dLQ7sTN&sig=aZTp8fQkWR6yzu1NrCQUvIWnYeA&redir_esc=y#v=onepage&q=bayesian%20filtering%20and%20smoothing&f=false) 

[2] [Vihola, M. Robust adaptive Metropolis algorithm with coerced acceptance rate. Stat Comput 22, 997–1008 (2012).](https://doi.org/10.1007/s11222-011-9269-5)

## See also
- [Metropolis Hastings algorithm for parameter estimation in Linear Models](https://github.com/debrup-sarkar/Metropolis-Hastings-algorithm-for-parameter-estimation-in-linear-models/blob/main/README.md)
