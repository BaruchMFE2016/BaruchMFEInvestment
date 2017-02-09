__author__ = 'Derek Qi'
# Date: 2/2/2017

import numpy as np
import cvxopt as opt
from cvxopt import solvers, matrix, sparse, spmatrix

solvers.options['show_progress'] = False  # Turn off progress printing


def optimizer(w_old, alpha, sigma, gamma=1, lambd=0, L=-1, U=1, dlt=1):
    """
    Optimizer given the projected alpha and sigma with a given utility function. Gives long-only portfolio
    :param w_old: numpy array n-by-1, weight of the last time period
    :param alpha: numpy array, n-by-1, projected asset alpha
    :param sigma: numpy array, n-by-n, projected asset covariance matrix, needs to be symmetric positive semi-definite
    :param gamma: volatility preference coefficient, needs to be positive
    :param lambd: transaction cost coefficient, needs to be positive
    :param L, U: Consistent lower bound and upper bound for each assets, L needs to be negative and U needs to be positive, for non-binding parameters use L=-1, U=1
    :param dlt: maximum change of positions from last time period, need to be positive, if non-binding use dlt=1
    :param hasshort: bool, long-only or long-short
    :return: numpy array, n-by-1, the optimized portfolio weight vector
    """

    alpha_m, sigma_m = matrix(alpha), matrix(sigma)
    N_INS = alpha_m.size[0]

    # Tool matices
    zero_m = matrix(0.0, (N_INS, N_INS))
    I = matrix(np.eye(N_INS))
    neg_I = - matrix(np.eye(N_INS))

    # matrices in quadratic function 1/2 x'Px + q'x
    q = matrix([-1 * alpha_m, matrix(lambd, (N_INS, 1))])
    P = matrix([[gamma * sigma_m, zero_m], [zero_m, zero_m]])

    # define the constraints
    # equalities Ax = b
    # sum of all the weights equal to 1
    A = matrix([[matrix(1.0, (1, N_INS))], [matrix(0.0, (1, N_INS))]])
    b = opt.matrix(1.0)

    # inequalities Gx <= h
    G_lst = []
    h_lst = []

    # Constraints

    # w_i > 0
    G_lst.append(matrix([neg_I, zero_m]).T)
    h_lst.append(matrix(0.0, (N_INS, 1)))

    # L_i < w_i < U_i
    G_lst.append(matrix([I, zero_m]).T)
    h_lst.append(matrix(U, (N_INS, 1)))

    G_lst.append(matrix([neg_I, zero_m]).T)
    h_lst.append(matrix(-L, (N_INS, 1)))

    # w_i_old - delta < w_i < w_i_old + delta
    G_lst.append(matrix([I, zero_m]).T)
    h_lst.append(matrix(w_old) + matrix(dlt, (N_INS, 1)))

    G_lst.append(matrix([neg_I, zero_m]).T)
    h_lst.append(matrix(-w_old) + matrix(dlt, (N_INS, 1)))

    # |w_i - w_i_old| < beta_i
    G_lst.append(matrix([I, neg_I]).T)
    h_lst.append(matrix(w_old))

    G_lst.append(matrix([neg_I, neg_I]).T)
    h_lst.append(matrix(-w_old))

    # beta_i > 0
    G_lst.append(matrix([matrix(0.0, (N_INS, N_INS)), neg_I]).T)
    h_lst.append(matrix(0.0, (N_INS, 1)))

    # Stacking together
    G = matrix([G for G in G_lst])
    h = matrix([h for h in h_lst])

    solution = solvers.qp(P, q, G, h, A, b)
    w_opt = solution['x'][:N_INS]
    w_opt = np.array(w_opt)
    return w_opt


def optimizerlongshort(w_old, alpha, sigma, gamma=1, lambd=0, L=-1, U=1, dlt=1, Lev=2):
    """
    Optimizer given the projected alpha and sigma with a given utility function. This optimizer gives the long-short portfolio.
    :param w_old: numpy array n-by-1, weight of the last time period
    :param alpha: numpy array, n-by-1, projected asset alpha
    :param sigma: numpy array, n-by-n, projected asset covariance matrix, needs to be symmetric positive semi-definite
    :param gamma: volatility preference coefficient, needs to be positive
    :param lambd: transaction cost coefficient, needs to be positive
    :param L, U: Consistent lower bound and upper bound for each assets, L needs to be negative and U needs to be positive, for non-binding parameters use L=-1, U=1
    :param dlt: maximum change of positions from last time period, need to be positive, if non-binding use dlt=1
    :param Lev: total level of leverage, if enable short
    :param hasshort: bool, long-only or long-short
    :return: numpy array, n-by-1, the optimized portfolio weight vector
    """

    alpha_m, sigma_m = matrix(alpha), matrix(sigma)
    N_INS = alpha_m.size[0]

    # Tool matices
    zero_m = matrix(0.0, (N_INS, N_INS))
    I = matrix(np.eye(N_INS))
    neg_I = - matrix(np.eye(N_INS))

    # matrices in quadratic function 1/2 x'Px + q'x
    q = matrix([-1 * alpha_m, matrix(lambd, (N_INS, 1)), matrix(0.0, (N_INS, 1))])
    P = matrix([[gamma * sigma_m, zero_m, zero_m], [zero_m, zero_m, zero_m], [zero_m, zero_m, zero_m]])

    # define the constraints
    # equalities Ax = b
    # sum of all the weights equal to 1
    A = matrix([[matrix(1.0, (1, N_INS))], [matrix(0.0, (1, 2 * N_INS))]])
    b = opt.matrix(1.0)

    # inequalities Gx <= h
    G_lst = []
    h_lst = []

    # Constraints

    # sum of |w_i| < Lev
    G_lst.append(matrix([[matrix(0.0, (1, 2 * N_INS))], [matrix(1.0, (1, N_INS))]]))
    h_lst.append(float(Lev))

    # All the absolute values are positive
    G_lst.append(matrix([zero_m, zero_m, neg_I]).T)
    h_lst.append(matrix(0.0, (N_INS, 1)))

    # And they are absolute values
    G_lst.append(matrix([I, zero_m, neg_I]).T)
    h_lst.append(matrix(0.0, (N_INS, 1)))

    G_lst.append(matrix([neg_I, zero_m, neg_I]).T)
    h_lst.append(matrix(0.0, (N_INS, 1)))

    # L_i < w_i < U_i
    G_lst.append(matrix([I, zero_m, zero_m]).T)
    h_lst.append(matrix(U, (N_INS, 1)))

    G_lst.append(matrix([neg_I, zero_m, zero_m]).T)
    h_lst.append(matrix(-L, (N_INS, 1)))

    # w_i_old - delta < w_i < w_i_old + delta
    G_lst.append(matrix([I, zero_m, zero_m]).T)
    h_lst.append(matrix(w_old) + matrix(dlt, (N_INS, 1)))

    G_lst.append(matrix([neg_I, zero_m, zero_m]).T)
    h_lst.append(matrix(-w_old) + matrix(dlt, (N_INS, 1)))

    # |w_i - w_i_old| < beta_i
    G_lst.append(matrix([I, neg_I, zero_m]).T)
    h_lst.append(matrix(w_old))

    G_lst.append(matrix([neg_I, neg_I, zero_m]).T)
    h_lst.append(matrix(-w_old))

    # beta_i > 0
    G_lst.append(matrix([zero_m, neg_I, zero_m]).T)
    h_lst.append(matrix(0.0, (N_INS, 1)))

    # Stacking together
    G = matrix([G for G in G_lst])
    h = matrix([h for h in h_lst])

    solution = solvers.qp(P, q, G, h, A, b)
    w_opt = solution['x'][:N_INS]
    w_opt = np.array(w_opt)
    return w_opt


if __name__ == "__main__":
    np.random.seed(123)
    n_assets = 4
    n_obs = 1000

    ret_vec = np.random.randn(n_assets, n_obs)
    alpha = np.mean(ret_vec, axis=1)
    sigma = np.cov(ret_vec)

    w = np.random.rand(n_assets)
    w /= sum(w)  # initial weights

    x_opt = optimizerlongshort(w_old=w, alpha=alpha, sigma=sigma)

    print(x_opt)
    port_ret = np.dot(alpha, x_opt)  # return of the optimized portfolio
    temp = np.dot(x_opt.T, np.linalg.cholesky(sigma))
    port_var = np.dot(temp, temp.T)  # variance of the optimized portfolio
    sharpe = port_ret / port_var  # simple sharpe ratio with 0 interest rate
    print('Sharpe ratio:', sharpe[0][0])
